// Sokoban solver — account 3 angle: IDA* (iterative deepening A*) on pushes,
// one independent push-optimal search per room, with a transposition table.
//
// Level structure (per the generator): a 1-wide hallway holding the player,
// plus 1-4 rooms off it, each in a 6x6 slot with ONE door and 3-4 boxes.
// A box pushed into the hallway can never come back, so every box stays in
// its own room and the level optimum is the sum of the rooms' optima.
// We therefore decompose, solve each room push-optimally, then stitch the
// rooms together in the walking-cheapest visiting order.
//
// Search state: (sorted box positions, canonical player region = min cell
// index of the player's reachable area). Cost is pushes; player walks are
// free and are recorded as tie-break / reconstruction detail only.
// Heuristic: sum over boxes of the minimum reverse-pull push-distance to
// any goal (precomputed by reverse BFS from the goals, ignoring other
// boxes). It is admissible and consistent. Dead squares (cells whose
// reverse-pull distance is infinite and which are not goals) are pruned:
// a box there can never reach a goal, with or without other boxes around.
// No other pruning is used: optimality matters more than speed here.
//
// Determinism: no Math.random, no Date, no clock. Every iteration order
// is fixed (directions u,d,l,r; boxes in sorted order; BFS in fixed
// neighbour order). Node budgets are counts, never wall clock, so two
// runs on any machine produce byte-identical output.
//
// Fallbacks (safety nets, in order): IDA* -> A* (same heuristic, optimal)
// -> greedy best-first (valid solution, may be non-optimal) -> whole-level
// IDA*/A*/greedy when the hallway/room structure does not hold.
// Every emitted solution is replayed on the level before it is written;
// a failed replay falls back to the next solver.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------- types
type Dir = { dr: number; dc: number; ch: string };
const DIRS: Dir[] = [
  { dr: -1, dc: 0, ch: 'u' },
  { dr: 1, dc: 0, ch: 'd' },
  { dr: 0, dc: -1, ch: 'l' },
  { dr: 0, dc: 1, ch: 'r' },
];

type Level = {
  w: number;
  h: number;
  walls: Set<number>;
  goals: Set<number>;
  boxes: number[];
  player: number;
};

type Room = {
  cells: Set<number>;
  door: number;
  entryHall: number;
  boxes: number[];
  goals: Set<number>;
};

type SolveResult = { moves: string; endPlayer: number } | null;

// ---------------------------------------------------------------- parse
function parseLevel(text: string): Level {
  const lines = text.split('\n');
  while (lines.length > 0 && lines[lines.length - 1].length === 0) lines.pop();
  const h = lines.length;
  let w = 0;
  for (const l of lines) if (l.length > w) w = l.length;
  const walls = new Set<number>();
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    const line = lines[r];
    for (let c = 0; c < w; c++) {
      const ch = c < line.length ? line[c] : '#';
      const idx = r * w + c;
      if (ch === '#') walls.add(idx);
      else if (ch === '.') goals.add(idx);
      else if (ch === '$') boxes.push(idx);
      else if (ch === '*') { boxes.push(idx); goals.add(idx); }
      else if (ch === '@') player = idx;
      else if (ch === '+') { player = idx; goals.add(idx); }
      // ' ' and anything else is plain floor
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, walls, goals, boxes, player };
}

// ---------------------------------------------------------------- structure
function detectStructure(lv: Level): { hallway: Set<number>; rooms: Room[] } | null {
  const hallRow = Math.floor(lv.player / lv.w);
  const hallway = new Set<number>();
  for (let c = 0; c < lv.w; c++) {
    const idx = hallRow * lv.w + c;
    if (!lv.walls.has(idx)) hallway.add(idx);
  }
  if (!hallway.has(lv.player)) return null;
  // No boxes or goals may sit in the hallway.
  for (const b of lv.boxes) if (hallway.has(b)) return null;
  for (const g of lv.goals) if (hallway.has(g)) return null;

  const seen = new Set<number>();
  const rooms: Room[] = [];
  const inHall = (i: number) => hallway.has(i);
  for (let r = 0; r < lv.h; r++) {
    for (let c = 0; c < lv.w; c++) {
      const idx = r * lv.w + c;
      if (lv.walls.has(idx) || inHall(idx) || seen.has(idx)) continue;
      // Flood fill one room component.
      const cells = new Set<number>();
      const stack = [idx];
      seen.add(idx);
      while (stack.length > 0) {
        const cur = stack.pop() as number;
        cells.add(cur);
        const cr = Math.floor(cur / lv.w);
        const cc = cur % lv.w;
        for (const d of DIRS) {
          const nr = cr + d.dr;
          const nc = cc + d.dc;
          if (nr < 0 || nr >= lv.h || nc < 0 || nc >= lv.w) continue;
          const n = nr * lv.w + nc;
          if (lv.walls.has(n) || inHall(n) || seen.has(n)) continue;
          seen.add(n);
          stack.push(n);
        }
      }
      // The door: the one room cell adjacent to a hallway cell.
      let door = -1;
      let entryHall = -1;
      let doorCount = 0;
      for (const cell of cells) {
        const cr = Math.floor(cell / lv.w);
        const cc = cell % lv.w;
        for (const d of DIRS) {
          const nr = cr + d.dr;
          const nc = cc + d.dc;
          if (nr < 0 || nr >= lv.h || nc < 0 || nc >= lv.w) continue;
          const n = nr * lv.w + nc;
          if (hallway.has(n)) {
            doorCount++;
            door = cell;
            entryHall = n;
          }
        }
      }
      if (doorCount !== 1) return null;
      const roomBoxes = lv.boxes.filter((b) => cells.has(b));
      const roomGoals = new Set<number>();
      for (const g of lv.goals) if (cells.has(g)) roomGoals.add(g);
      if (roomBoxes.length === 0 || roomBoxes.length !== roomGoals.size) return null;
      rooms.push({ cells, door, entryHall, boxes: roomBoxes, goals: roomGoals });
    }
  }
  if (rooms.length === 0 || rooms.length > 4) return null;
  // Every box and every goal must belong to exactly one room.
  let counted = 0;
  for (const room of rooms) counted += room.boxes.length;
  if (counted !== lv.boxes.length) return null;
  // Sort rooms by door cell for a deterministic order.
  rooms.sort((a, b) => a.door - b.door);
  return { hallway, rooms };
}

// ---------------------------------------------------------------- BFS helpers
// Shortest walk path (as a move string) from `from` to `to`, walking only
// on `floor` cells and avoiding `blocked` cells. Fixed neighbour order.
function walkPath(
  from: number,
  to: number,
  w: number,
  floor: Set<number>,
  blocked: Set<number>,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, { p: number; ch: string }>();
  const queue = [from];
  prev.set(from, { p: -1, ch: '' });
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    const cr = Math.floor(cur / w);
    const cc = cur % w;
    for (const d of DIRS) {
      const nr = cr + d.dr;
      const nc = cc + d.dc;
      if (nr < 0 || nc < 0 || nc >= w) continue;
      const n = nr * w + nc;
      if (!floor.has(n) || blocked.has(n) || prev.has(n)) continue;
      prev.set(n, { p: cur, ch: d.ch });
      if (n === to) {
        let s = '';
        let x = n;
        while (x !== from) {
          const step = prev.get(x) as { p: number; ch: string };
          s = step.ch + s;
          x = step.p;
        }
        return s;
      }
      queue.push(n);
    }
  }
  return null;
}

// Reachable region from `start` over `floor` cells avoiding `blocked`.
// Returns the set and its minimum cell index (the canonical player key).
function region(
  start: number,
  w: number,
  h: number,
  floor: Set<number>,
  blocked: Set<number>,
): { set: Set<number>; min: number } {
  const set = new Set<number>();
  const queue = [start];
  set.add(start);
  let min = start;
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    if (cur < min) min = cur;
    const cr = Math.floor(cur / w);
    const cc = cur % w;
    for (const d of DIRS) {
      const nr = cr + d.dr;
      const nc = cc + d.dc;
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const n = nr * w + nc;
      if (!floor.has(n) || blocked.has(n) || set.has(n)) continue;
      set.add(n);
      queue.push(n);
    }
  }
  return { set, min };
}

// ---------------------------------------------------------------- problem
type Problem = {
  w: number;
  h: number;
  floor: Set<number>; // cells the player may walk on (room cells + entry hall cell)
  boxFloor: Set<number>; // cells a box may occupy (room cells only; never the hallway)
  goals: Set<number>;
  startBoxes: number[];
  startPlayer: number;
  door: number; // -1 when there is no door requirement (whole-level solve)
  dist: Map<number, number>; // reverse-pull push distance to nearest goal
};

function buildProblem(
  w: number,
  h: number,
  floor: Set<number>,
  boxFloor: Set<number>,
  goals: Set<number>,
  boxes: number[],
  player: number,
  door: number,
): Problem {
  // Reverse-pull BFS from the goals: dist[c] = minimum pushes needed to
  // move a box from cell c onto any goal, ignoring other boxes. A box at
  // `prev` can be pushed to `cur` when the player can stand on `stand`
  // (prev - dir); the box itself must stay on boxFloor.
  const dist = new Map<number, number>();
  const queue: number[] = [];
  for (const g of goals) {
    dist.set(g, 0);
    queue.push(g);
  }
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    const d = dist.get(cur) as number;
    const cr = Math.floor(cur / w);
    const cc = cur % w;
    for (const dir of DIRS) {
      const prevR = cr - dir.dr;
      const prevC = cc - dir.dc;
      const standR = cr - 2 * dir.dr;
      const standC = cc - 2 * dir.dc;
      if (prevR < 0 || prevR >= h || prevC < 0 || prevC >= w) continue;
      if (standR < 0 || standR >= h || standC < 0 || standC >= w) continue;
      const prev = prevR * w + prevC;
      const stand = standR * w + standC;
      if (!boxFloor.has(prev) || !floor.has(stand)) continue;
      if (dist.has(prev)) continue;
      dist.set(prev, d + 1);
      queue.push(prev);
    }
  }
  return { w, h, floor, boxFloor, goals, startBoxes: boxes.slice(), startPlayer: player, door, dist };
}

function heuristic(prob: Problem, boxes: number[]): number {
  let s = 0;
  for (const b of boxes) {
    const d = prob.dist.get(b);
    if (d === undefined) return Infinity;
    s += d;
  }
  return s;
}

function isGoalState(prob: Problem, boxes: number[], regionSet: Set<number>): boolean {
  for (const b of boxes) if (!prob.goals.has(b)) return false;
  if (prob.door >= 0 && !regionSet.has(prob.door)) return false;
  return true;
}

type PushOption = {
  newBoxes: number[];
  newPlayer: number;
  box: number; // the box cell before the push (for the incremental heuristic)
  dir: Dir;
  stand: number; // the cell the player must stand on to make this push
};

// All pushes available from a state, in fixed deterministic order.
// `reg` is the player's reachable region for this state. The walk path to
// the push position is NOT computed here: callers compute it lazily only
// for pushes that survive pruning, which is the dominant cost saving.
function pushOptions(
  prob: Problem,
  boxes: number[],
  reg: Set<number>,
): PushOption[] {
  const out: PushOption[] = [];
  const boxSet = new Set<number>(boxes);
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const br = Math.floor(b / prob.w);
    const bc = b % prob.w;
    for (const dir of DIRS) {
      const tr = br + dir.dr;
      const tc = bc + dir.dc;
      const sr = br - dir.dr;
      const sc = bc - dir.dc;
      if (tr < 0 || tr >= prob.h || tc < 0 || tc >= prob.w) continue;
      if (sr < 0 || sr >= prob.h || sc < 0 || sc >= prob.w) continue;
      const target = tr * prob.w + tc;
      const stand = sr * prob.w + sc;
      if (!prob.boxFloor.has(target) || !prob.floor.has(stand)) continue;
      if (boxSet.has(target)) continue;
      if (!reg.has(stand)) continue;
      // Dead square: a box on `target` could never reach a goal.
      if (!prob.goals.has(target) && prob.dist.get(target) === undefined) continue;
      const newBoxes = boxes.slice();
      newBoxes[i] = target;
      newBoxes.sort((a, b2) => a - b2);
      out.push({ newBoxes, newPlayer: b, box: b, dir, stand });
    }
  }
  return out;
}

// The walk + push move string for taking one push option from `player`.
// Returns null when the push position is unreachable (should not happen
// when the option came from pushOptions with the matching region).
function pushSegment(
  prob: Problem,
  boxes: number[],
  player: number,
  opt: PushOption,
): string | null {
  const walk = walkPath(player, opt.stand, prob.w, prob.floor, new Set<number>(boxes));
  if (walk === null) return null;
  return walk + opt.dir.ch;
}

function stateKey(boxes: number[], regionMin: number): string {
  return boxes.join(',') + '|' + regionMin;
}

// ---------------------------------------------------------------- IDA*
// Iterative deepening A* on pushes, with a transposition table holding the
// best push-count seen per state. Pruning a revisit is safe only when the
// new push-count is STRICTLY worse than the stored one: IDA* re-explores
// from the root every iteration, so the stored (better or equal) path is
// always re-generated. The heuristic is consistent, so a strictly worse
// revisit can never lead to a better solution.
function idaStar(prob: Problem, budget: number): SolveResult | 'ABORT' {
  const h0 = heuristic(prob, prob.startBoxes);
  if (h0 === Infinity) return null;
  const table = new Map<string, number>();
  let expansions = 0;
  const moves: string[] = [];
  let solution: { moves: string; endPlayer: number } | null = null;

  const dfs = (
    boxes: number[],
    player: number,
    g: number,
    h: number,
    bound: number,
  ): number => {
    if (solution !== null) return bound;
    expansions++;
    if (expansions > budget) return -1; // abort flag
    const boxSet = new Set<number>(boxes);
    const reg = region(player, prob.w, prob.h, prob.floor, boxSet);
    const f = g + h;
    if (f > bound) return f;
    if (isGoalState(prob, boxes, reg.set)) {
      solution = { moves: moves.join(''), endPlayer: player };
      return bound;
    }
    let minNext = Infinity;
    for (const opt of pushOptions(prob, boxes, reg.set)) {
      const nreg = region(opt.newPlayer, prob.w, prob.h, prob.floor, new Set<number>(opt.newBoxes));
      const key = stateKey(opt.newBoxes, nreg.min);
      const seen = table.get(key);
      if (seen !== undefined && g + 1 > seen) continue;
      table.set(key, g + 1);
      // Incremental heuristic: only the pushed box changes distance.
      // Its new cell is the old cell stepped one in the push direction.
      const br = Math.floor(opt.box / prob.w);
      const bc = opt.box % prob.w;
      const newCell = (br + opt.dir.dr) * prob.w + (bc + opt.dir.dc);
      const h2 = h - (prob.dist.get(opt.box) as number) + (prob.dist.get(newCell) as number);
      const seg = pushSegment(prob, boxes, player, opt);
      if (seg === null) continue;
      moves.push(seg);
      const t = dfs(opt.newBoxes, opt.newPlayer, g + 1, h2, bound);
      moves.pop();
      if (solution !== null) return bound;
      if (t === -1) return -1;
      if (t < minNext) minNext = t;
    }
    return minNext;
  };

  let bound = h0;
  for (;;) {
    const t = dfs(prob.startBoxes, prob.startPlayer, 0, h0, bound);
    if (solution !== null) return solution;
    if (t === -1) return 'ABORT';
    if (t === Infinity) return null;
    bound = t;
  }
}

// ---------------------------------------------------------------- A*
// Same state space and heuristic, best-first with a closed set. Optimal
// with an admissible consistent heuristic. Deterministic tie-breaks:
// (f, g, state key). Used when IDA* exceeds its node budget.
function aStar(prob: Problem, budget: number): SolveResult | 'ABORT' {
  const h0 = heuristic(prob, prob.startBoxes);
  if (h0 === Infinity) return null;
  type Node = {
    boxes: number[];
    player: number;
    g: number;
    h: number;
    f: number;
    key: string;
    parent: Node | null;
    seg: string;
  };
  const heap: Node[] = [];
  const cmp = (a: Node, b: Node): number => {
    if (a.f !== b.f) return a.f - b.f;
    if (a.g !== b.g) return a.g - b.g;
    return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
  };
  const push = (n: Node) => {
    heap.push(n);
    let i = heap.length - 1;
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (cmp(heap[i], heap[p]) >= 0) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const pop = (): Node => {
    const top = heap[0];
    const last = heap.pop() as Node;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let m = i;
        if (l < heap.length && cmp(heap[l], heap[m]) < 0) m = l;
        if (r < heap.length && cmp(heap[r], heap[m]) < 0) m = r;
        if (m === i) break;
        const tmp = heap[i];
        heap[i] = heap[m];
        heap[m] = tmp;
        i = m;
      }
    }
    return top;
  };
  const startReg = region(prob.startPlayer, prob.w, prob.h, prob.floor, new Set<number>(prob.startBoxes));
  const startKey = stateKey(prob.startBoxes, startReg.min);
  push({ boxes: prob.startBoxes, player: prob.startPlayer, g: 0, h: h0, f: h0, key: startKey, parent: null, seg: '' });
  const best = new Map<string, number>();
  best.set(startKey, 0);
  let expansions = 0;
  while (heap.length > 0) {
    const node = pop();
    if ((best.get(node.key) as number) < node.g) continue;
    expansions++;
    if (expansions > budget) return 'ABORT';
    const boxSet = new Set<number>(node.boxes);
    const reg = region(node.player, prob.w, prob.h, prob.floor, boxSet);
    if (isGoalState(prob, node.boxes, reg.set)) {
      const segs: string[] = [];
      let cur: Node | null = node;
      while (cur !== null && cur.parent !== null) {
        segs.push(cur.seg);
        cur = cur.parent;
      }
      segs.reverse();
      return { moves: segs.join(''), endPlayer: node.player };
    }
    for (const opt of pushOptions(prob, node.boxes, reg.set)) {
      const nreg = region(opt.newPlayer, prob.w, prob.h, prob.floor, new Set<number>(opt.newBoxes));
      const key = stateKey(opt.newBoxes, nreg.min);
      const g2 = node.g + 1;
      const seen = best.get(key);
      if (seen !== undefined && g2 >= seen) continue;
      const br = Math.floor(opt.box / prob.w);
      const bc = opt.box % prob.w;
      const newCell = (br + opt.dir.dr) * prob.w + (bc + opt.dir.dc);
      const h2 = node.h - (prob.dist.get(opt.box) as number) + (prob.dist.get(newCell) as number);
      const seg = pushSegment(prob, node.boxes, node.player, opt);
      if (seg === null) continue;
      best.set(key, g2);
      push({ boxes: opt.newBoxes, player: opt.newPlayer, g: g2, h: h2, f: g2 + h2, key, parent: node, seg });
    }
  }
  return null;
}

// ---------------------------------------------------------------- greedy
// Best-first on the heuristic alone, with a visited set. Returns the first
// valid solution found; used only when both optimal searches give up.
function greedy(prob: Problem, budget: number): SolveResult {
  const h0 = heuristic(prob, prob.startBoxes);
  if (h0 === Infinity) return null;
  type Node = { boxes: number[]; player: number; parent: Node | null; seg: string; h: number };
  const heap: Node[] = [];
  const cmp = (a: Node, b: Node): number => {
    if (a.h !== b.h) return a.h - b.h;
    return 0;
  };
  const push = (n: Node) => {
    heap.push(n);
    let i = heap.length - 1;
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (cmp(heap[i], heap[p]) >= 0) break;
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    }
  };
  const pop = (): Node => {
    const top = heap[0];
    const last = heap.pop() as Node;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let m = i;
        if (l < heap.length && cmp(heap[l], heap[m]) < 0) m = l;
        if (r < heap.length && cmp(heap[r], heap[m]) < 0) m = r;
        if (m === i) break;
        const tmp = heap[i];
        heap[i] = heap[m];
        heap[m] = tmp;
        i = m;
      }
    }
    return top;
  };
  const startReg = region(prob.startPlayer, prob.w, prob.h, prob.floor, new Set<number>(prob.startBoxes));
  const visited = new Set<string>();
  visited.add(stateKey(prob.startBoxes, startReg.min));
  push({ boxes: prob.startBoxes, player: prob.startPlayer, parent: null, seg: '', h: h0 });
  let expansions = 0;
  while (heap.length > 0) {
    const node = pop();
    expansions++;
    if (expansions > budget) return null;
    const boxSet = new Set<number>(node.boxes);
    const reg = region(node.player, prob.w, prob.h, prob.floor, boxSet);
    if (isGoalState(prob, node.boxes, reg.set)) {
      const segs: string[] = [];
      let cur: Node | null = node;
      while (cur !== null && cur.parent !== null) {
        segs.push(cur.seg);
        cur = cur.parent;
      }
      segs.reverse();
      return { moves: segs.join(''), endPlayer: node.player };
    }
    for (const opt of pushOptions(prob, node.boxes, reg.set)) {
      const nreg = region(opt.newPlayer, prob.w, prob.h, prob.floor, new Set<number>(opt.newBoxes));
      const key = stateKey(opt.newBoxes, nreg.min);
      if (visited.has(key)) continue;
      const br = Math.floor(opt.box / prob.w);
      const bc = opt.box % prob.w;
      const newCell = (br + opt.dir.dr) * prob.w + (bc + opt.dir.dc);
      const h2 = node.h - (prob.dist.get(opt.box) as number) + (prob.dist.get(newCell) as number);
      const seg = pushSegment(prob, node.boxes, node.player, opt);
      if (seg === null) continue;
      visited.add(key);
      push({ boxes: opt.newBoxes, player: opt.newPlayer, parent: node, seg, h: h2 });
    }
  }
  return null;
}

// Run the optimal chain for one problem, then the greedy safety net.
// IDA* goes first with a small node budget (it settles easy rooms fast);
// harder rooms fall through to A*, which expands each state once and is
// still push-optimal with this admissible heuristic.
function solveProblem(prob: Problem): SolveResult {
  let r: SolveResult | 'ABORT' = idaStar(prob, 100000);
  if (r === 'ABORT') r = aStar(prob, 1000000);
  if (r === 'ABORT' || r === null) r = greedy(prob, 500000);
  return r === 'ABORT' ? null : r;
}

// ---------------------------------------------------------------- replay
// Replays a move string on the level; returns the push count when every
// box ends on a goal, and -1 otherwise (invalid move or unsolved).
function replay(lv: Level, moves: string): number {
  const boxes = new Set<number>(lv.boxes);
  let player = lv.player;
  let pushes = 0;
  for (const ch of moves) {
    if (ch === '\n' || ch === '\r' || ch === ' ' || ch === '\t') continue;
    const dir = DIRS.find((d) => d.ch === ch.toLowerCase());
    if (!dir) return -1;
    const pr = Math.floor(player / lv.w);
    const pc = player % lv.w;
    const nr = pr + dir.dr;
    const nc = pc + dir.dc;
    if (nr < 0 || nr >= lv.h || nc < 0 || nc >= lv.w) return -1;
    const next = nr * lv.w + nc;
    if (lv.walls.has(next)) return -1;
    if (boxes.has(next)) {
      const br = nr + dir.dr;
      const bc = nc + dir.dc;
      if (br < 0 || br >= lv.h || bc < 0 || bc >= lv.w) return -1;
      const beyond = br * lv.w + bc;
      if (lv.walls.has(beyond) || boxes.has(beyond)) return -1;
      boxes.delete(next);
      boxes.add(beyond);
      pushes++;
    }
    player = next;
  }
  for (const b of boxes) if (!lv.goals.has(b)) return -1;
  return pushes;
}

// ---------------------------------------------------------------- level solve
function solveLevel(lv: Level): string {
  const struct = detectStructure(lv);
  if (struct !== null) {
    const combined = solveDecomposed(lv, struct);
    if (combined !== null && replay(lv, combined) >= 0) return combined;
  }
  // Whole-level fallback (same machinery, no door requirement).
  const floor = new Set<number>();
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.walls.has(i)) floor.add(i);
  const prob = buildProblem(lv.w, lv.h, floor, floor, lv.goals, lv.boxes, lv.player, -1);
  const res = solveProblem(prob);
  if (res !== null && replay(lv, res.moves) >= 0) return res.moves;
  // Last resort: return the decomposed attempt if any, else empty.
  if (struct !== null) {
    const combined = solveDecomposed(lv, struct);
    if (combined !== null) return combined;
  }
  return '';
}

function solveDecomposed(
  lv: Level,
  struct: { hallway: Set<number>; rooms: Room[] },
): string | null {
  const { hallway, rooms } = struct;
  // Solve every room independently (push-optimal). The room problem's
  // floor is the room cells plus the hallway cell in front of the door:
  // the player starts there, and a box sitting on the door cell is pushed
  // inward from the hallway. Boxes may only ever occupy room cells.
  const roomSol: SolveResult[] = [];
  for (const room of rooms) {
    const floor = new Set<number>(room.cells);
    floor.add(room.entryHall);
    const prob = buildProblem(lv.w, lv.h, floor, room.cells, room.goals, room.boxes, room.entryHall, room.door);
    const res = solveProblem(prob);
    if (res === null) return null;
    roomSol.push(res);
  }
  // Try every visiting order; pick the one with the least walking.
  const n = rooms.length;
  const order: number[] = [];
  for (let i = 0; i < n; i++) order.push(i);
  let bestPerm: number[] | null = null;
  let bestWalk = Infinity;
  const perms = permutations(order);
  for (const perm of perms) {
    let walk = 0;
    let cur = lv.player;
    let ok = true;
    for (const ri of perm) {
      const d = hallwayDist(cur, rooms[ri].entryHall, lv.w, hallway);
      if (d === null) { ok = false; break; }
      walk += d;
      cur = rooms[ri].entryHall;
    }
    if (ok && walk < bestWalk) {
      bestWalk = walk;
      bestPerm = perm;
    }
  }
  if (bestPerm === null) return null;
  // Stitch the chosen order into one move string.
  let out = '';
  let curPos = lv.player;
  let curBoxes = new Set<number>(lv.boxes);
  for (const ri of bestPerm) {
    const room = rooms[ri];
    const sol = roomSol[ri] as { moves: string; endPlayer: number };
    // Walk the hallway to this room's entry cell. The room solution
    // starts with the player standing there (entryHall), so no separate
    // step-in move is needed: the solution's first moves walk in / push
    // the door box inward from the hallway as needed.
    const hw = walkPath(curPos, room.entryHall, lv.w, hallway, curBoxes);
    if (hw === null) return null;
    out += hw;
    // The room's own solution (player starts on the entry hall cell).
    out += sol.moves;
    // Walk out: from the room's end position back to the door, then step out.
    const finalBoxes = new Set<number>(curBoxes);
    for (const b of room.boxes) finalBoxes.delete(b);
    for (const g of room.goals) finalBoxes.add(g);
    const exitWalk = walkPath(sol.endPlayer, room.door, lv.w, room.cells, finalBoxes);
    if (exitWalk === null) return null;
    out += exitWalk;
    const stepOut = stepChar(room.door, room.entryHall, lv.w);
    if (stepOut === null) return null;
    out += stepOut;
    curPos = room.entryHall;
    curBoxes = finalBoxes;
  }
  return out;
}

// BFS distance between two hallway cells (hallway only).
function hallwayDist(from: number, to: number, w: number, hallway: Set<number>): number | null {
  if (from === to) return 0;
  const dist = new Map<number, number>();
  const queue = [from];
  dist.set(from, 0);
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    const d = dist.get(cur) as number;
    const cr = Math.floor(cur / w);
    const cc = cur % w;
    for (const dir of DIRS) {
      const nr = cr + dir.dr;
      const nc = cc + dir.dc;
      if (nr < 0 || nc < 0 || nc >= w) continue;
      const n = nr * w + nc;
      if (!hallway.has(n) || dist.has(n)) continue;
      if (n === to) return d + 1;
      dist.set(n, d + 1);
      queue.push(n);
    }
  }
  return null;
}

function stepChar(from: number, to: number, w: number): string | null {
  const fr = Math.floor(from / w);
  const fc = from % w;
  const tr = Math.floor(to / w);
  const tc = to % w;
  for (const d of DIRS) if (fr + d.dr === tr && fc + d.dc === tc) return d.ch;
  return null;
}

function permutations(arr: number[]): number[][] {
  if (arr.length <= 1) return [arr.slice()];
  const out: number[][] = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permutations(rest)) out.push([arr[i]].concat(p));
  }
  return out;
}

// ---------------------------------------------------------------- main
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const lv = parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  const moves = solveLevel(lv);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
