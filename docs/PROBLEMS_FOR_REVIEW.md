# Current Problems With Our AlphaZero Quoridor Training

This document summarizes the issues we're facing so someone (or an LLM) can help diagnose and suggest fixes.

---

## The Setup

- **Game**: Quoridor on 9×9 board, 2 players, 10 walls each
- **Architecture**: AlphaZero (ResNet 12 blocks, 256 channels, 20M params)
- **MCTS**: 800 simulations per move, PUCT selection, Dirichlet noise
- **Training**: Self-play → train network → arena (new vs old) → accept if >55% win rate
- **Infrastructure**: C++ game engine + MCTS via pybind11, GPU (A10G) for neural net inference
- **Game cap**: 500 moves (draw if nobody wins)

---

## Problem 1: Games are too long (250-300 moves average)

Even after pre-training the model to walk straight toward the goal, self-play games average 250 moves. This makes each iteration take 30-40 minutes instead of the 5 minutes we'd expect if games were 30-40 moves.

**Why games are long**: Both players rush forward but then get stuck in wall battles. They place walls, go around them, place more walls, creating long detours. The model knows "move toward goal" but doesn't know "place walls efficiently and stop."

**What we want**: Games should end in 20-40 moves once the model is good. Top Quoridor players finish games in 15-25 moves.

---

## Problem 2: Arena games all draw (0W/0L/40D)

When the trained model plays against the previous version, every game draws at 500 moves. Neither can beat the other decisively.

**Why**: Both models play nearly identically (trained on same data). They mirror each other's rushing strategy, place walls symmetrically, and neither breaks through.

**Current workaround**: Auto-accept for first 5 iterations (skip arena). But this means we have no quality gate — we don't know if the model is actually improving.

---

## Problem 3: Model learns to predict moves but doesn't improve play quality

Policy loss drops nicely (2.5 → 0.25) but the model doesn't play better. It gets good at predicting what MCTS outputs — but MCTS with a bad network outputs bad moves. So it learns to be accurately bad.

**The circular problem**:
1. Bad network → MCTS produces mediocre policies
2. Train network on mediocre policies → network predicts mediocre moves well
3. Next iteration: same quality MCTS → same quality data → no improvement

---

## Problem 4: Value signal is weak

Most games draw (value = partial credit ≈ small number near 0). The network rarely sees +1.0 (won) or -1.0 (lost). Without strong value signal, it can't learn "this position is winning/losing."

**Partial credit formula**: `value = (opponent_BFS_dist - my_BFS_dist) / 60.0` — gives [-0.5, 0.5] based on path distance. But this is a weak gradient compared to actual wins/losses.

---

## Problem 5: Wall placement strategy doesn't emerge

After 33+ iterations in one run, the model still dumps all walls randomly or not at all. It hasn't learned:
- WHEN to place walls (timing)
- WHERE to place walls (on opponent's shortest path)
- How many walls to save for later

---

## What We've Tried

| Approach | Result |
|----------|--------|
| Start from random, 400 sims | Stuck at 50/50 forever |
| Start from random, 800 sims | Same — MCTS overrides network signal |
| Curriculum sims (200 → 800) | Didn't help |
| Auto-accept first 5 iterations | Model accepted but doesn't improve |
| C++ game cap 100 moves | Bug: every game drew, zero signal |
| C++ game cap 500 moves | Games too long, mostly draws |
| Supervised pre-training (BFS shortest path) | ✓ WORKS — model walks straight, wins arena iteration 1 |
| Larger network (6→12 blocks) | No difference when training signal is bad |
| BFS partial credit for draws | Helps value learning slightly |

---

## What We Think Might Work (Haven't Tried)

1. **Shorter game cap (50-100 moves)** with stronger partial credit — forces decisive outcomes faster
2. **Reward shaping**: value = 1.0 - (moves_taken / max_moves) for wins — incentivizes fast wins
3. **Asymmetric training**: pre-train one player to place walls, other to navigate — creates diversity
4. **Opponent pool**: play against all past versions, not just the latest — prevents "mirror match draws"
5. **Auxiliary losses**: predict BFS distance as a side task — gives additional learning signal
6. **Progressive board size**: start on 5×5 with 3 walls, master it, then move to 9×9
7. **Forced wall placement**: require at least 1 wall placed per game in early iterations — forces model to learn wall play
8. **Temperature annealing on walls**: higher temperature for wall actions specifically — explore more wall positions

---

## Key Metrics (Latest Run, Iteration 4)

- Self-play game length: 250 moves avg
- Policy loss: 0.25 (good prediction accuracy, bad play quality)  
- Value loss: 0.01-0.05
- Arena: all draws (40/40)
- Time per iteration: ~35 minutes
- Positions per iteration: ~25,000
- Total positions seen: ~100,000

---

## Problem 6: Local Optimum — Model Stuck at "Rush Forward"

**Date**: 2026-07-20 (latest issue)

After implementing wall filtering, tempo fixes, γ-discounting, 60-move cap, asymmetric games, and data augmentation, the model is stuck in a local optimum.

**Symptom**: Policy loss flat at 0.88-0.91 across iterations. Game length flat at 46-49 moves. No improvement iteration to iteration. Model just rushes forward (pre-trained behavior) and self-play data only reinforces "rushing works."

**Root cause**: The pre-trained model learned "rush forward" so strongly that:
- Self-play: both sides rush → game ends in ~48 moves → training data says "rushing = good"
- Walls never get explored because rushing beats random/bad walls
- Network never sees evidence that walls are useful
- Circular: no wall data → no wall learning → no wall data

**What we tried that DIDN'T break the local optimum**:
- Wall filtering in MCTS (helps MCTS pick good walls, but network never asks for walls)
- Asymmetric games 20% (one player has 0 walls — but the 10-wall player still just rushes)
- Higher learning rate
- More iterations (just confirms rushing is good)
- Auto-accept (accepts models that are same quality, doesn't drive improvement)

**Options we're considering to force wall exploration**:

1. **Forced wall moves**: In 30% of self-play games, first 3-5 moves MUST be wall placements (from filtered candidates). Creates training data showing "what happens when walls exist" — some games the wall-placer wins.

2. **Higher exploration noise**: Increase Dirichlet ε from 0.25 to 0.5 for first 20 iterations. MCTS occasionally picks wall moves even when network says "rush."

3. **Two-phase training**: Phase 1 (done) = rush. Phase 2 = freeze rushing weights, train only wall decision head. Like teaching driving straight first, then turns.

4. **Immediate wall bonus**: When a wall increases opponent BFS by 3+, give +0.2 value bonus to that position regardless of game outcome. Direct reward for effective wall placement.

5. **Opponent modeling**: In arena, pit current model against a "wall-heavy" heuristic bot that places 5 walls then rushes. If model can't navigate walls, it loses → learns to handle walls.

6. **Curriculum of opponents**: Instead of self-play only, mix in games against:
   - Pure rusher (no walls)
   - Wall spammer (places 5 random walls then rushes)
   - Strategic waller (places walls on opponent shortest path then rushes)
   Model must beat all three → forced to learn both rushing AND wall navigation.

---

## The Core Question

How do we break the model out of the "rush forward" local optimum and get it to:
1. Discover that well-timed walls can win games faster than pure rushing
2. Learn WHEN to place walls (timing) vs WHEN to move (tempo)
3. Actually improve in playing strength each iteration

The model CAN walk straight (pre-training proved that). The model CAN identify good wall positions (wall filtering in MCTS finds them). The gap is: the NETWORK never learns to REQUEST wall moves because it never sees evidence that walls lead to winning.

This is a classic exploration-exploitation problem: the model exploits "rushing" and never explores "walls" because any single wall attempt looks worse than rushing in the short term.
