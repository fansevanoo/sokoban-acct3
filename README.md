[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/fansevanoo/sokoban-acct3/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/fansevanoo/sokoban-acct3/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Sokoban Solver — IDA* per Room

A push-optimal Sokoban solver for make.fun's `sokoban-solver-01-run-20261121`.

Levels are a 1-wide hallway plus 1–4 rooms off it; a box pushed into the
hallway can never come back, so every box stays in its own room and the
level optimum is the sum of the rooms' optima. This solver decomposes the
level, solves each room independently with an exact push-optimal search,
and stitches the rooms together in the walking-cheapest visiting order.

**Search:** IDA* (iterative deepening A*) on pushes with a transposition
table, over states of (sorted box positions, canonical player region).
Heuristic: sum of per-box minimum reverse-pull push-distances to any goal.
Dead squares (reverse-pull unreachable, non-goal cells) are pruned; no
other pruning is used, so the result is provably push-optimal. Fallbacks:
A* (same heuristic, still optimal) then greedy best-first, then a
whole-level search when the hallway/room structure does not hold. Every
solution is replayed on the level before it is written.

**Usage:** `MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> node main.ts`
(Node 22, stdlib only, deterministic — two runs are byte-identical.)

**Practice set:** 13/13, 16/16, 29/29, 47/47, 60/60 pushes — all optimal
(P*), ~18 s total for all 5 levels.
