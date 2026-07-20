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

## The Core Question

How do we get the model to:
1. Play shorter, decisive games (30-40 moves, not 250)
2. Learn wall placement timing and positioning
3. Actually improve in playing strength each iteration (not just prediction accuracy)

The model CAN walk straight (pre-training proved that). The gap is going from "walks straight" to "uses walls strategically to win faster."
