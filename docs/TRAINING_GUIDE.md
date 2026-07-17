# How the AI Learns to Play Quoridor

A complete guide to understanding the AlphaZero training process for this project.

---

## Table of Contents

1. [The Big Idea](#the-big-idea)
2. [The Three Components](#the-three-components)
3. [The Training Loop Step by Step](#the-training-loop-step-by-step)
4. [What Each Metric Means](#what-each-metric-means)
5. [How the Neural Network Works](#how-the-neural-network-works)
6. [How MCTS Works](#how-mcts-works)
7. [Why Batching Makes It Fast](#why-batching-makes-it-fast)
8. [What to Expect During Training](#what-to-expect-during-training)
9. [How to Monitor Training](#how-to-monitor-training)
10. [File Structure Explained](#file-structure-explained)
11. [Cost and Time Estimates](#cost-and-time-estimates)
12. [Glossary](#glossary)

---

## The Big Idea

The AI starts knowing **absolutely nothing** about Quoridor. It doesn't know the goal, it doesn't know walls are useful, it doesn't even know which direction to move. It makes completely random moves.

Through playing **thousands of games against itself**, it gradually discovers what works and what doesn't. No human strategy is programmed in. The system learns entirely from experience.

This is called **AlphaZero** — the same approach DeepMind used to master Chess, Go, and Shogi in 2017.

### The Core Loop (Plain English)

```
1. AI plays a bunch of games against itself (badly at first)
2. It looks at which moves led to wins and which led to losses
3. It updates its "brain" to prefer winning moves
4. It plays more games (slightly better now)
5. Repeat 50 times
```

After enough repetitions, the random player becomes a strategic one.

---

## The Three Components

### 1. The Neural Network ("The Brain")

A deep neural network that looks at a board position and answers two questions:

| Question | Output | Example |
|----------|--------|---------|
| "What moves look good here?" | **Policy** — probability for each of 209 possible actions | "Move forward: 60%, Place wall left: 15%, ..." |
| "Who's winning?" | **Value** — a number from -1 to +1 | "+0.7 = I'm probably winning" |

At the start, the network outputs random garbage. After training, it gives useful predictions.

**Architecture**: ResNet (Residual Network) — the same family of networks used for image recognition. It has 6 "blocks" of computation with 128 channels each, totaling 2.4 million learnable parameters.

### 2. MCTS ("The Thinking Process")

MCTS = Monte Carlo Tree Search. Instead of just taking the neural network's raw suggestion, the AI **thinks ahead** by simulating possibilities.

Think of it like this: before making a move in chess, you consider "if I go here, they might go there, then I could..."

The AI does this systematically:
```
For 200 simulations:
  1. Start from current position
  2. Walk down a "tree" of possibilities
     - Choose moves that the network thinks are good (exploitation)
     - Also try moves that haven't been explored yet (exploration)
  3. Reach a new position → ask the network "who's winning here?"
  4. Walk back up the tree, updating scores

After 200 simulations:
  The move that was visited most = the best move
```

**Key insight**: MCTS + a mediocre network > the network alone. Even a bad network gives useful signal when you search ahead.

### 3. Self-Play ("The Practice Games")

The AI plays against a **copy of itself**. Both sides use the same network + MCTS. This creates training data:

For every position in every game, we save:
- **State**: what the board looked like
- **MCTS Policy**: what the thinking process concluded (better than the raw network output)
- **Outcome**: who actually won the game

This data teaches the network: "From this position, MCTS figured out these moves were good, and the game ended with this result."

---

## The Training Loop Step by Step

Each "iteration" has 3 phases:

### Phase 1: Self-Play (90% of compute time)

```
Input:  Current neural network
Output: ~5,000 training positions (state, policy, value)

What happens:
  - Play 100 games of AI vs itself
  - Each move: run MCTS for 200 simulations to pick the best move
  - Record every position + what MCTS concluded + who eventually won
  - Early in game: add randomness to explore different strategies
  - Later in game: play the best moves found
```

**Why it's slow**: 100 games × ~50 moves × 200 simulations × neural network call = millions of computations.

### Phase 2: Training (5% of compute time)

```
Input:  All accumulated training positions (could be 50,000+)
Output: Updated neural network weights

What happens:
  - Feed positions to the network in batches of 64
  - For each batch:
    - Network predicts policy + value
    - Compare to what MCTS actually found (policy target)
    - Compare to who actually won (value target)
    - Adjust weights to reduce the gap (backpropagation)
  - Repeat for 10 passes ("epochs") over all data
```

**The losses being minimized**:
- **Policy loss**: "Your move predictions should match what MCTS concluded after deep thinking"
- **Value loss**: "Your win/loss predictions should match actual game outcomes"

### Phase 3: Arena (5% of compute time)

```
Input:  New model (just trained) vs Old model (before training)
Output: Decision: accept new model or reject it

What happens:
  - Play 20 games: new model vs old model
  - Alternate who plays first (to be fair)
  - If new model wins > 55% of decisive games → ACCEPT
  - If not → REJECT and keep old model
```

**Why this exists**: Training can sometimes make the model worse (catastrophic forgetting). The arena ensures we only keep improvements.

---

## What Happens Each Iteration (Detailed)

### [1/3] Self-play (the slowest part)

The AI plays 100 games against itself. Both sides use 800 MCTS simulations per move (thinking deeply before each move). Each game produces training examples.

**What a training example is**:
- Board position (what it saw) — 12×9×9 tensor
- MCTS policy (what deep thinking concluded) — 209 probabilities
- Value (who won, or BFS partial credit if draw) — single number [-1, +1]

**Why self-play is slow**: 100 games × ~100 moves × 800 sims × neural net call per sim. Even with C++, that's millions of operations.

**What makes games shorter over time**: As the model learns to play toward the goal, games end in 30-40 moves instead of 100+. This means later iterations are faster than early ones.

### [2/3] Training (the fast part)

The neural network studies all the examples (10 passes/epochs):

**Policy loss** = "How wrong are my move predictions?"
- Network predicts move probabilities
- Compare to what MCTS concluded after 800 sims of deep thinking
- Adjust parameters to make predictions match MCTS
- Think of it as: MCTS is the slow careful teacher, the network is the student learning to make instant good decisions

**Value loss** = "How wrong are my win/loss predictions?"
- Network predicts who's winning from any board position
- Compare to actual game outcomes (or BFS distance for draws)
- Adjust parameters to be more accurate
- This is what lets the network "feel" whether a position is good or bad

**Why it's fast** (30-60 seconds): Pure matrix multiplication on GPU. No game logic, no search — just feeding boards through the network and adjusting 20M weights.

### [3/3] Arena (the quality gate)

New model plays 20 games against the old model. Tests if training actually made it better.

- Alternates who plays Red/Blue (fairness)
- Uses C++ MCTS for both sides
- Win >55% of decisive games → accepted as new "best model"
- Otherwise → rejected, keep old model, try again next iteration

**Why this matters**: Training can sometimes make a model worse (it learns one thing but forgets another). The arena prevents regressions — the model can only get stronger, never weaker.

**What "decisive" means**: Games where someone actually won (not draws). If all games draw, win rate = 50% = rejected. This is why draws were a problem before.

### Draw Rules

- Game cap: **500 moves**. If nobody wins in 500 moves → draw.
- Draw partial credit: based on **BFS shortest path distance** to goal (not straight-line distance)
  - Player with shorter path to goal gets positive value
  - Player with longer path gets negative value
  - This teaches: "being walled off = bad, clear path = good"
- Early iterations: most games end around 80-100 moves (random play eventually stumbles to goal)
- Later iterations: games end in 20-40 moves (intentional, efficient play)
- Draws become rare once both models learn to play purposefully

---

## What Each Metric Means

### Policy Loss (should decrease over training)

How wrong the network's raw move predictions are compared to MCTS's conclusions.

| Value | Meaning |
|-------|---------|
| 4.0-5.0 | Random predictions (untrained) |
| 2.0-3.0 | Starting to learn basic patterns |
| 1.0-2.0 | Good predictions, MCTS only makes small corrections |
| 0.5-1.0 | Excellent — network nearly matches MCTS |

### Value Loss (should decrease over training)

How wrong the win/loss predictions are compared to actual outcomes.

| Value | Meaning |
|-------|---------|
| 0.8-1.0 | Basically guessing randomly |
| 0.3-0.5 | Learning who's likely to win |
| 0.1-0.3 | Good position evaluation |
| < 0.1 | Very accurate — knows who's winning |

### Arena Win Rate (should be > 55%)

How often the newly trained model beats the previous version.

| Value | Meaning |
|-------|---------|
| 50% | No improvement (random chance) |
| 55-60% | Modest improvement (accepted) |
| 60-70% | Strong improvement |
| 70%+ | Major leap in strength |
| < 50% | Got worse (rejected) |

### Average Game Length (should decrease)

How many moves an average game takes.

| Moves | What's happening |
|-------|-----------------|
| 150-200 | Random play — wandering around |
| 80-120 | Learning to move toward goal, but inefficient |
| 40-60 | Direct path awareness, some detours |
| 20-35 | Efficient play with wall strategy |
| 15-25 | Expert level — shortest path + perfect walls |

---

## How the Neural Network Works

### Input: Board Encoding (12 planes of 9×9)

The network sees the board as 12 "layers" stacked on top of each other, each 9×9:

```
Plane 0:  Where is my pawn?          (one-hot: 1 at my position, 0 elsewhere)
Plane 1:  Where is opponent's pawn?   (one-hot)
Plane 2:  How many walls do I have?   (all cells = walls_left/10)
Plane 3:  How many walls do they have? (all cells = their_walls/10)
Plane 4:  Where are horizontal walls? (1 where wall exists, padded to 9×9)
Plane 5:  Where are vertical walls?   (1 where wall exists)
Plane 6-7: (reserved)
Plane 8:  My goal row                 (all 1s on the row I'm trying to reach)
Plane 9:  Opponent's goal row         (all 1s on their target row)
Plane 10: Where can my pawn legally move? (1 on reachable cells)
Plane 11: Am I player 1?             (all 1s or all 0s)
```

This encoding gives the network everything it needs to understand the position.

### Output: Policy + Value

**Policy head** (209 numbers that sum to 1):
```
Actions 0-80:    Move pawn to each of the 81 cells (9×9)
Actions 81-144:  Place horizontal wall at each of 64 intersections (8×8)
Actions 145-208: Place vertical wall at each of 64 intersections (8×8)
```

Most of these are illegal on any given turn. We mask out illegal moves before picking.

**Value head** (single number between -1 and +1):
```
-1.0: I'm definitely losing
 0.0: Even position
+1.0: I'm definitely winning
```

### How It Learns

The network adjusts its 2.4 million parameters using **gradient descent**:

1. Show it a board position
2. It predicts policy + value
3. Compare to the "correct" answer (MCTS policy + game outcome)
4. Compute the error (loss)
5. Adjust each parameter slightly to reduce the error
6. Repeat millions of times

---

## How MCTS Works

### The Tree

MCTS builds a tree of possible futures:

```
                    Current Position
                   /     |      \
              Move A   Move B   Move C
              /   \      |       |
           A→D  A→E    B→F     C→G
           ...  ...    ...     ...
```

Each node stores:
- **N** = how many times we've visited this move
- **W** = total value accumulated (sum of game outcomes)
- **Q** = average value (W/N)
- **P** = prior probability (what the network initially thought)

### Selection Formula (PUCT)

When choosing which branch to explore:

```
Score = Q(s,a) + c × P(s,a) × √(N_parent) / (1 + N(s,a))
         ↑               ↑
    exploitation      exploration
    "pick moves       "also try moves
     that worked"      we haven't explored"
```

- **Exploitation** (Q): prefer moves that led to good results
- **Exploration** (P × √N / (1+N)): also try moves the network likes but we haven't fully explored

The constant **c** (=1.5) controls the balance. Higher = more exploration.

### Dirichlet Noise

At the root node only, we add random noise to the prior probabilities:

```
P_noisy = 0.75 × P_network + 0.25 × Dirichlet(0.3)
```

This forces the AI to occasionally try moves it wouldn't normally consider, preventing it from getting stuck in one strategy. Only applied during self-play training, not during real play.

### Temperature

Controls how random the move selection is:

```
Temperature = 1.0 (first 15 moves): Sample from visit counts proportionally
                                     → explores different openings
Temperature = 0.1 (after 15 moves): Almost always pick the most-visited move
                                     → plays strongest move found
```

---

## Why Batching Makes It Fast

### The Problem

The neural network runs on a GPU (Graphics Processing Unit). GPUs have thousands of cores that work in parallel — they're designed to process many things at once.

But our MCTS calls the network **one position at a time**:

```
Simulation 1: send 1 board to GPU → GPU processes it → get result (GPU 99% idle)
Simulation 2: send 1 board to GPU → GPU processes it → get result (GPU 99% idle)
...
```

The GPU finishes each tiny job in microseconds, then sits idle while the CPU prepares the next one. Result: **4% GPU utilization**.

### The Solution

Run 32 games simultaneously. When they all need the network evaluated, batch them:

```
32 games all reach a leaf node simultaneously →
Stack all 32 board states into one tensor →
ONE GPU call processes all 32 at once →
Distribute results back to each game's tree
```

The GPU takes barely longer to process 32 states than 1 (it has thousands of cores). But we've eliminated 31 round-trips of overhead.

Result: **50-80% GPU utilization**, **10-20x faster** overall.

---

## What to Expect During Training

### Iteration 1 (The Slowest)

- Games last ~200 moves (random play, hits the move cap)
- Policy loss starts at ~4.0
- Value loss starts at ~0.8
- Takes longest because games are so long
- After training: model barely improves but learns "moving is better than standing still"

### Iterations 2-5: "Learning to Walk"

- Games shorten to 100-150 moves
- Model discovers moving towards the goal is good
- Still doesn't understand walls
- Win rate in arena: ~55-60%

### Iterations 5-15: "The Rushing Phase"

- Games drop dramatically to 30-60 moves
- Model runs in a straight line to the goal
- Starts to notice walls exist but doesn't use them well
- Policy loss drops to ~2.0

### Iterations 15-30: "Wall Discovery"

- Model starts placing walls to block opponent
- Games stabilize at 25-35 moves
- First signs of "strategy" — blocking shortest path
- Value loss drops below 0.3 (can predict outcomes)

### Iterations 30-50: "Strategic Play"

- Wall economy emerges (saving walls for key moments)
- Model learns to read the opponent's position
- Opening patterns develop
- Games take 20-30 moves with intentional wall play

---

## How to Monitor Training

### TensorBoard (charts)

```bash
# SSH tunnel from your machine:
ssh -i "D:\Downloads\vla-key-pair.pem" -L 6006:localhost:6006 ubuntu@<IP> "cd /home/ubuntu/barricade/engine && tensorboard --logdir runs"

# Then open: http://localhost:6006
```

Shows: loss curves, win rates, game lengths over time.

### Training Log (text)

```bash
ssh -i "D:\Downloads\vla-key-pair.pem" ubuntu@<IP> "tail -f /home/ubuntu/barricade/train.log"
```

Shows: iteration progress, loss values, arena results in real-time.

### Training Journal (after training)

```bash
# Copy the full journal to your machine:
scp -r -i "D:\Downloads\vla-key-pair.pem" ubuntu@<IP>:/home/ubuntu/barricade/engine/training_runs/ ./engine/training_runs/
```

Contains: config, per-iteration metrics, notable game replays, milestones, summary.

### Watch AI Play (live viewer)

```bash
# Start watch server on AWS:
ssh -i "D:\Downloads\vla-key-pair.pem" ubuntu@<IP> "cd /home/ubuntu/barricade/engine && python3 watch.py &"

# Tunnel it:
ssh -i "D:\Downloads\vla-key-pair.pem" -L 8001:localhost:8001 ubuntu@<IP>

# Open your frontend (localhost:5174) → "Watch AI"
```

---

## File Structure Explained

```
engine/
├── model.py              The neural network architecture
│                         - Input: 12×9×9 board encoding
│                         - Output: 209 move probabilities + 1 value
│                         - 2.4 million parameters
│
├── mcts.py               Original MCTS (sequential, 1 game at a time)
│                         - Used for: watch server, arena, CPU training
│
├── parallel_mcts.py      Batched MCTS (multiple games, GPU-optimized)
│                         - Used for: self-play on GPU (10-20x faster)
│
├── self_play.py          Plays games to generate training data
│                         - Auto-detects GPU → uses parallel version
│                         - Records replays for the journal
│
├── train.py              The main training loop orchestrator
│                         - Calls self-play → training → arena → repeat
│                         - Logs to TensorBoard + journal
│
├── arena.py              Pits new model vs old model
│                         - 20 games, alternating colors
│                         - >55% win rate = accept
│
├── journal.py            Auto-documents everything during training
│                         - Saves replays, detects milestones
│                         - Generates summary.md at the end
│
├── replay.py             Replay file analysis tools
│                         - Step through games frame by frame
│                         - Generate text descriptions of games
│
├── watch.py              WebSocket server for live viewing
│                         - Plays AI games in real-time
│                         - Broadcasts to browser viewers
│
├── ai_player.py          Wraps trained model for the game API
│                         - Loads checkpoint, uses MCTS to pick moves
│
├── game.py               The actual game rules
│                         - Board, moves, walls, pathfinding
│                         - BFS ensures walls never fully trap a player
│
├── dashboard.py          Serves training metrics as JSON API
│
└── training_runs/        Output of training
    └── run_001/
        ├── config.json       Hyperparameters used
        ├── journal.jsonl     Per-iteration metrics
        ├── milestones.json   Achievement list
        ├── summary.md        Auto-generated report
        ├── replays/          Notable game recordings
        └── checkpoints/      Model snapshots
```

---

## Cost and Time Estimates

### On a g5.xlarge (A10G GPU, ~$1/hr)

| Setting | Time per iteration | Total (50 iter) | Cost |
|---------|-------------------|-----------------|------|
| 100 games, 200 sims (parallel) | ~10-15 min | ~8-12 hours | ~$10-15 |
| 200 games, 400 sims (parallel) | ~30-45 min | ~25-37 hours | ~$25-40 |
| 50 games, 100 sims (parallel) | ~5-8 min | ~4-7 hours | ~$5-8 |

### Why First Iteration Is Slowest

The random model plays 200-move games (max cap). A trained model plays 20-30 move games. So iteration 1 is literally 10x slower than iteration 20+. The training accelerates itself.

---

## Glossary

| Term | Meaning |
|------|---------|
| **AlphaZero** | The algorithm: neural network + MCTS + self-play |
| **Arena** | Tournament between new and old model to check improvement |
| **Backpropagation** | The algorithm that adjusts neural network weights based on errors |
| **Batch** | Multiple inputs processed together in one GPU call |
| **Checkpoint** | Saved model weights at a point in time |
| **Dirichlet noise** | Random noise added to encourage exploration |
| **Epoch** | One full pass through all training data |
| **GPU** | Graphics Processing Unit — massively parallel processor |
| **Gradient descent** | Method of minimizing loss by adjusting parameters downhill |
| **Iteration** | One full cycle of self-play + train + arena |
| **Loss** | How wrong the network's predictions are (lower = better) |
| **MCTS** | Monte Carlo Tree Search — lookahead search algorithm |
| **Policy** | Probability distribution over all possible moves |
| **PUCT** | The formula used to balance exploration vs exploitation |
| **Replay buffer** | Accumulated training data from recent games |
| **ResNet** | Residual Network — type of deep neural network with skip connections |
| **Self-play** | AI playing against a copy of itself to generate data |
| **Simulation** | One traversal of the MCTS tree (select → evaluate → backprop) |
| **Temperature** | Controls randomness in move selection |
| **TensorBoard** | Visualization tool for training metrics |
| **Value** | The network's prediction of who's winning (-1 to +1) |
| **Win rate** | Percentage of games won in the arena |

---

## Further Reading

- [AlphaZero paper](https://arxiv.org/abs/1712.01815) — the original DeepMind paper
- [AlphaGo documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y) — shows the human vs AI match
- [MCTS explained](https://www.youtube.com/watch?v=UXW2yZndl7U) — visual explanation
- [Neural networks from scratch](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown series
