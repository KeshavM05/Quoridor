# Why Training Is Slow (And What We Tried)

## The Simple Version

Imagine you need to play 30 chess games to learn from. Each game has ~150 moves. Before EACH move, you need to "think" 50 times (MCTS simulations). Each "thought" requires asking a brain (neural network) a question.

```
30 games × 150 moves × 50 thoughts = 225,000 brain questions

Each brain question takes ~2 milliseconds on GPU
225,000 × 2ms = 450 seconds = 7.5 minutes

That's just ONE batch of 10 games. Three batches = ~22 minutes for self-play alone.
```

Then after self-play, the Arena does 10 more games with the same math. Total iteration time: ~1 hour.

---

## Why the GPU Is Only 4% Utilized

Think of the GPU like a restaurant kitchen with 100 chefs.

**What we're doing**: We walk in, hand ONE plate to all 100 chefs, they finish it in 0.1 seconds, hand it back. Then we spend 2 seconds walking back to our table, deciding what to order next, walking back to the kitchen, handing them ONE plate again.

The chefs are idle 95% of the time waiting for us.

**What we SHOULD do**: Carry 32 plates at once, hand them ALL to the 100 chefs. They finish all 32 in 0.3 seconds (barely slower than 1). Now the chefs are busy 60% of the time.

The problem is our "walking back and forth" is the MCTS tree traversal on CPU. We can only figure out which plate to order next after we get the previous answer back.

---

## What Is Deepcopy and Why It Broke Parallel MCTS

### What is deepcopy?

In Python, `copy.deepcopy(game)` creates a complete independent clone of the game board. It copies every variable, every array, every piece of data so you have two separate games that don't affect each other.

This costs about **1 millisecond** per copy for our game (9×9 board + walls + positions).

### Why MCTS needs it

MCTS simulates "what if?" scenarios:
- "What if I move here?" → need to actually play that move on a COPY (not the real game)
- "Then what if they move there?" → need another copy to go deeper
- Each branch of the tree needs its own independent game state

### What went wrong in parallel_mcts.py

The ORIGINAL mcts.py (that works) does this:
```
Simulation 1: copy game → play 1 move → evaluate → done
Simulation 2: copy game → play 1 move → evaluate → done  (new leaf, depth 1)
Simulation 3: copy game → play 2 moves → evaluate → done (goes to depth 2)
...
Simulation 50: copy game → play ~10 moves → evaluate → done (depth ~10)
```

Each simulation copies once and replays a few moves. Total: ~50 copies. Fine.

The BROKEN parallel_mcts.py accidentally did this:
```
Simulation 1: copy at depth 0 → store copy in node
Simulation 2: copy at depth 0 → play 1 move → copy THAT at depth 1 → store
Simulation 3: copy at depth 0 → play 1 → copy at 1 → play 1 → copy at 2 → store
...
Simulation 200: copy × 200 nested levels deep
```

It was copying EVERY intermediate node's state permanently into the tree. That's:
- Simulation 1: 1 copy
- Simulation 2: 2 copies
- Simulation 3: 3 copies
- ...
- Simulation 200: 200 copies

Total: 1 + 2 + 3 + ... + 200 = **20,100 copies** per move.

At 1ms each = **20 seconds per move**. A 150-move game = **50 minutes per game**. 30 games = **25 hours** just for self-play. That's why it hung — it wasn't stuck in an infinite loop, it was just impossibly slow.

### Why we couldn't just fix it

The "replay from root" approach (copy once at root, replay all moves to reach the leaf) avoids storing copies in nodes. But it still does one copy + N move replays per simulation:

```
Simulation 50: copy root → replay 10 moves to reach leaf
```

On CPU this is fine (~0.5ms). But on our test it was still too slow because Python is slow at game logic (playing moves, checking walls, running BFS for path validation).

### The real solution (that production systems use)

Write the game engine and MCTS in **C++** (1000x faster than Python for this kind of code), and only call Python/GPU for the neural network evaluations. Our game's `play_move()` + `get_legal_moves()` + `_path_exists()` BFS are the actual bottleneck — they run in Python and take ~0.1ms each. In C++ they'd take 0.0001ms.

---

## What Actually Makes Training Faster (Ranked)

| Fix | Speedup | Why | Status |
|-----|---------|-----|--------|
| Fewer sims (50 instead of 400) | 8× | Directly fewer neural net calls | ✅ Done |
| Fewer games (30 instead of 200) | 7× | Fewer total games to play | ✅ Done |
| Cap game length (100 instead of 200) | 1.5-2× | Random models waste moves wandering | ✅ Done (just pushed) |
| C++ game engine | 10-50× | Python is slow at game logic | ❌ Major rewrite |
| Batched leaf evaluation (C++ MCTS) | 10-20× | Feed 32 states to GPU at once | ❌ Major rewrite |
| Multiple GPUs | 4× | Parallel games on separate GPUs | ❌ Different instance type |
| Smaller network | 2-3× | Faster inference per call | ❌ Sacrifices quality |

---

## Current State

**What's running on AWS right now**: Sequential MCTS, 30 games, 50 sims, ~1 hour per iteration.

**Will it work?** Yes. It'll produce a trained model. Just takes 50 hours for 50 iterations.

**Is this normal?** For a Python implementation, yes. Production AlphaZero at DeepMind used:
- 5000 TPUs for self-play (not 1 GPU)
- C++ MCTS (not Python)
- 40 million games (not 1500)
- 700,000 training steps (not 50)

We're doing a mini version that proves the concept and produces a playable AI. The model will be weaker than DeepMind's but will absolutely learn real Quoridor strategy.

---

## The Math of Our Current Training

```
Per iteration:
  Self-play: 30 games × 100 moves × 50 sims × 2ms = 300,000 × 2ms = 600s = 10 min
  Training:  5000 positions × 5 epochs × 0.1ms = 2.5s ≈ 0 min
  Arena:     10 games × 100 moves × 25 sims × 2ms = 25,000 × 2ms = 50s ≈ 1 min

Total per iteration: ~11 minutes (optimistic)
                     ~30-60 min (realistic, includes Python overhead, BFS, game logic)

50 iterations: 25-50 hours
Cost: $25-50 at $1/hr
```

The big difference between "optimistic" and "realistic" is Python overhead — the time between GPU calls where Python does tree traversal, game logic, BFS path checking, etc. That's the 95% that isn't the GPU.
