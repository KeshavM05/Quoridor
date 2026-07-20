# Optimization History

Complete record of every attempt to make the AI training work — what we tried, what failed, what worked, and why. Written so someone (or an LLM) can understand the full context and suggest what to try next.

---

## Timeline

### Day 1 (July 15): Initial Setup

**Goal**: Get AlphaZero training running on AWS GPU.

**What we built**:
- Python game engine (game.py): 9×9 Quoridor with walls, BFS pathfinding, jumps
- Neural network (model.py): ResNet 6 blocks, 128 channels, 2.4M params
- MCTS (mcts.py): PUCT selection, Dirichlet noise, 100-400 simulations
- Self-play, training loop, arena, all in Python
- AWS launch scripts (g5.xlarge, A10G GPU, $1/hr)

**Config**: 200 games, 400 sims, 50 iterations

**Result**: Instance launched, training started. After 19 hours, hadn't completed iteration 1.

**Root cause**: Python MCTS is absurdly slow. Each neural net call is fine (~2ms on GPU) but the Python overhead between calls (tree traversal, deepcopy of game state, BFS path validation) takes 95% of the time. GPU utilization: 4%.

---

### Day 2 (July 16): Speed Optimization

**Attempt 1: Parallel MCTS (batch GPU calls)**
- Idea: Run 32 games simultaneously, batch leaf evaluations into one GPU call
- Result: HUNG. The implementation had O(n²) deepcopy bug — each MCTS simulation stored a deepcopy in every tree node. With 200 sims, tree grows 200 levels deep, each level copies game state. Total: 20,100 copies per move.
- Abandoned.

**Attempt 2: Reduce settings**
- Dropped to 30 games, 50 sims
- Result: Iteration 1 completed in ~1 hour. But arena also took 5.5 hours (Python MCTS for arena games too).

**Attempt 3: C++ rewrite**
- Rewrote game engine + MCTS in C++ with pybind11 bindings
- Game state copy: memcpy ~200 bytes (vs Python deepcopy ~1ms)
- BFS: stack-allocated arrays (vs Python heap allocation)
- MCTS tree traversal: pure C++ (vs Python interpreter overhead)
- Batched leaf evaluation: collect 8-32 leaves, one Python callback for GPU
- Result: **46ms per move** (vs 2000ms in Python). 43x speedup.

**After C++ fix**: Self-play 100 games in 7 minutes (was 37 minutes). Arena 40 games in 2-5 minutes (was 5.5 hours).

---

### Day 2-3 (July 16-17): The 50/50 Problem

**Symptom**: Every arena result was exactly 50% — 20W/20L or 10W/10L. Model never accepted.

**What we tried**:
1. More arena games (20 → 40): Still 50/50, just with more data points confirming it
2. Lower threshold (55% → 50%): Would just accept random noise
3. Curriculum sims (200 early → 800 later): Didn't help
4. Auto-accept first 5 iterations: Model accepted but didn't improve afterward

**Diagnosis**: The model literally couldn't differentiate from random because:
- Policy loss dropped (looked good) but it was learning to predict random MCTS output accurately
- Value loss was 0.0000 — NO training signal about winning/losing

---

### Day 3 (July 17): The Game Cap Bug

**Discovery**: Every self-play game was EXACTLY 101 moves. Average game length: 101.0 for every iteration.

**Root cause**: The C++ self-play wrapper (`self_play_game_cpp()`) had `if move_count > 100: break` — the old cap from when we reduced it for Python speed. We updated the Python path to 500 but never updated the C++ path.

**Impact**: Every game drew at 101 moves. Value was always ~0 (BFS partial credit near zero because both players roughly equidistant). The model NEVER saw a win (+1) or loss (-1). It had no concept of what "winning" means.

**Fix**: Changed cap to 500 in both paths.

**After fix**: Games varied in length (some 50, some 400). Value loss became non-zero. But model still struggled because...

---

### Day 3-4 (July 17-20): Cold Start Bootstrap Problem

**Symptom**: Even with correct game cap, model stuck at 50/50 for 33 iterations.

**Diagnosis**: 
- Random network → random MCTS policies → random training data
- Network trained on random data → still random
- Circular: bad network → bad data → bad network (no escape)

**Why AlphaZero at DeepMind works from random**: They used 5000 TPUs playing millions of games. Sheer volume eventually breaks the cycle. With 100 games per iteration, we don't generate enough "accidental good moves" to bootstrap.

**What we tried**:
- 800 sims (more thinking): Didn't help — more sims with bad network = same quality output
- Bigger network (20M params): More capacity but same garbage in → garbage out
- Curriculum sims: Network prior too weak to matter even at 200 sims

---

### Day 4 (July 20): Supervised Pre-training Breakthrough

**Idea**: Don't start from random. First teach the model "move toward goal via shortest path" using a simple heuristic. THEN start self-play.

**Implementation**:
- Generate 2000 games where both players follow greedy BFS-shortest-path moves (using C++)
- Policy target: 85% on best forward move, 15% spread on other legal moves
- Value target: based on row distance to goal
- Train for 30 epochs (took 60 seconds total including data generation)

**Result**: 
- Model immediately walks straight toward goal
- First arena after self-play iteration 1: **20W / 0L / 20D (100% win rate)**
- First time ANY model was accepted legitimately (not auto-accept)

**Why this works**: Breaks the circular bootstrap. The model starts with a useful prior ("move forward"). MCTS with this prior produces meaningful policies. Training on meaningful policies → model improves. Positive feedback loop begins.

**Why it's not "cheating"**: We only teach "move toward goal." Wall strategy, jumps, timing, path manipulation — all discovered through self-play. The heuristic is weaker than what self-play will eventually produce.

---

### Day 4 (July 20): Current State

**Config**: 200 iterations, 100 games/iter, 800 sims, 12 blocks/256ch (20M params), pre-trained start

**Iteration 1**: 100% arena win rate. Games avg 250 moves. Self-play 38 min.
**Iteration 2-4**: All draws in arena (accepted via auto-accept first 5). Games still 250-300 moves.

**Current problem**: Games are too long. Model knows to rush forward but gets stuck in wall battles that drag to 250+ moves. Need games to be decisive in 30-50 moves.

---

## What's Working

- ✅ C++ engine (43x speedup over Python)
- ✅ Pre-training on BFS shortest path (breaks cold start)
- ✅ Training loop mechanics (self-play → train → arena all functional)
- ✅ Model walks toward goal (doesn't wander randomly)
- ✅ Arena with C++ (5 minutes instead of 5.5 hours)
- ✅ BFS partial credit for draws (provides some value signal)

## What's NOT Working

- ❌ Games too long (250 moves avg, want 30-50)
- ❌ Wall strategy not emerging (no learning signal for wall timing)
- ❌ Arena mostly draws after iteration 1 (models play identically)
- ❌ No clear improvement per iteration (policy loss low but play quality stagnant)

---

## Current Hypotheses for Why Games Are Long

1. **Both players rush forward and wall each other symmetrically** → creates mutual blockade → long detours
2. **No incentive to win FAST** → model happy to meander as long as it eventually wins
3. **800 sims makes both sides play "perfectly"** → neither makes mistakes → drawn-out battles
4. **Wall placement is random exploration** → walls don't help, just create obstacles for both sides
5. **BFS partial credit is too weak** → doesn't strongly reward positions that are close to winning

---

## Ideas Not Yet Tried (Ranked by Expected Impact)

### High Impact
1. **Win speed reward**: value = 1.0 - (moves/100) for wins. Winning in 15 moves = 0.85, winning in 80 moves = 0.20. Incentivizes fast decisive play.
2. **Game cap at 60 moves**: Forces games to end. With partial credit, model learns "be closer to goal at move 60" which directly incentivizes efficient play.
3. **Opponent pool**: Play against past versions (iter 1, 5, 10, etc.) not just the latest. Creates diversity — some opponents are weak (easy wins = strong signal), some are strong.
4. **Progressive training**: Start on 5×5 board (3 walls). Games end in 8-12 moves. Model learns strategy fast. Transfer to 9×9.

### Medium Impact
5. **Asymmetric games**: One player has 10 walls, other has 0. The walled player MUST learn to navigate. The wall-placer MUST learn where walls are effective.
6. **Auxiliary loss on BFS distance**: Side task — predict your own BFS distance. Forces the network to understand path structure.
7. **Reduce sims in self-play (200) but keep high in arena (800)**: Self-play with lower sims = more "mistakes" = more decisive games = stronger value signal. Arena keeps quality gate high.
8. **Increase exploration**: Higher Dirichlet noise, more temperature moves, to create diverse games.

### Lower Impact
9. **Larger replay buffer**: Keep 200K positions instead of 50K — more diverse training data.
10. **Learning rate schedule**: Start at 0.002, decay to 0.0001 over iterations.
11. **Data augmentation**: Mirror board horizontally (valid because board is symmetric). 2x training data.

---

## Hardware/Infrastructure

- **GPU**: NVIDIA A10G (24GB VRAM), using ~400MB
- **Instance**: g5.xlarge, us-east-1, ~$1/hr
- **CPU**: 4 vCPUs (mostly idle — C++ is fast)
- **Storage**: 100GB EBS
- **Total spent so far**: ~$120 (5 days running)
- **Budget remaining**: ~$19,800

---

## Code Architecture

```
Python (orchestration + GPU):
  train.py → calls generate_self_play_data() → calls quoridor_cpp
  Neural net inference stays in PyTorch (GPU)

C++ (game logic + MCTS tree):
  quoridor_cpp.mcts_search(game, sims, batch_size, temp, noise, eval_fn)
  - Traverses MCTS tree in C++
  - Collects batch of leaf states
  - Calls Python eval_fn (one GPU forward pass for whole batch)
  - Expands nodes + backpropagates in C++
  
  quoridor_cpp.QuoridorGame
  - Full game logic (moves, walls, BFS, jumps)
  - State copy = memcpy 200 bytes
```

---

## Key Insight We're Missing

The model needs to learn that **walls are a resource that creates asymmetry**. Right now both players rush forward and wall each other equally → symmetric game → no advantage → draw.

The breakthrough moment will be when the model discovers: "If I place THIS wall at THIS moment, my opponent's path becomes 10 moves longer but mine stays the same. I win the race."

That requires the model to:
1. Understand BFS path length (partially learned via pre-training)
2. Predict how a wall changes BOTH players' path lengths
3. Know when the path-length advantage justifies spending a turn on a wall instead of moving

This is a multi-step reasoning task that might need more training iterations OR a different reward signal that specifically rewards creating path-length asymmetry.
