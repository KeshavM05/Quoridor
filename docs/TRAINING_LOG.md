# Training Deployment Log

Every issue, decision, and fix encountered while setting up and running training.

---

## Attempt Timeline

### Run 1 — Original settings (FAILED)
**Config**: 200 games, 400 simulations, sequential, us-east-2
**Issue**: vCPU quota was 0 for GPU instances in us-east-2
**Fix**: Switched to us-east-1 where quota was 32 vCPUs

### Run 2 — First successful launch
**Config**: 200 games, 400 simulations, sequential
**Instance**: g5.xlarge (A10G GPU), us-east-1
**Issues**:
1. **SSH key mismatch** — the key pair in AWS us-east-1 was from a previous import with a different public key than the local `.pem` file. Fixed by deleting and re-importing from the actual pem.
2. **Output buffering** — `nohup python3 -u train.py > log 2>&1 &` didn't flush output. Even with `-u` flag, the `print()` inside `generate_self_play_data()` only fires every 10 games, so log appeared stuck.
3. **Process ran as root** — the user-data script runs as root, creating files ubuntu user can't overwrite. Fixed with `chown -R ubuntu:ubuntu`.
4. **Training too slow** — 200 games × 400 sims × ~200 moves (random model) = 16 million neural net calls. Ran for 19 hours without completing iteration 1.

**Decision**: Kill and restart with lighter settings.

### Run 3-5 — Lighter settings (CRASHED/SLOW)
**Config**: 50 games, 100 sims → then 20 games, 50 sims
**Issues**:
1. Processes kept dying silently — turned out to be the `from main import move_to_notation` inside `self_play.py` — it works but the journal recording code path with `record_replays=True` slows things down.
2. Log still shows no updates for minutes at a time — confirmed this is just Python buffering + the fact that each game takes 2-4 minutes when playing 200 random moves.

### Run 6 — Parallel MCTS attempt (HUNG)
**Config**: 100 games, 200 sims, parallel batch_size=32
**Issue**: The `parallel_mcts.py` has a bug — `generate_self_play_data_parallel()` hangs indefinitely. The batch self-play loop likely has an infinite loop condition (games never finishing or getting stuck in the MCTS search).
**Decision**: Disabled parallel mode (`--no-parallel`), use sequential until parallel is debugged.

### Run 7-8 — Sequential with light settings (CURRENT)
**Config**: 30 games, 50 sims, sequential, `--no-parallel`
**Status**: Working! First 10/30 games completed after ~12 minutes. Expected total iteration 1 time: ~36 minutes.
**Why slow**: Random model plays ~150 moves per game. 30 games × 150 moves × 50 sims = 225,000 neural net forward passes. Each takes ~2ms sequentially on GPU = ~450 seconds ≈ 7-8 minutes per 10 games.

---

## Key Decisions Made

### 1. Region Choice: us-east-1
- **Why**: Had 32 vCPU quota for G-type instances (GPU). us-east-2 had 0.
- **Impact**: Needed to import key pair, create security group, find correct AMI for that region.

### 2. Instance Type: g5.xlarge
- **Why**: A10G 24GB GPU at ~$1/hr. Cheapest NVIDIA GPU instance good for training.
- **Tradeoff**: Could use p3.2xlarge (V100, $3/hr, faster) but A10G is sufficient.

### 3. AMI: Deep Learning Base OSS Nvidia Driver (Ubuntu 22.04)
- **Why**: Comes with NVIDIA drivers pre-installed. We install PyTorch + our deps on top.
- **Alternative**: Could use Deep Learning AMI with PyTorch pre-installed but those are bigger/slower to boot.

### 4. Training Settings: 30 games, 50 sims
- **Why**: Iteration 1 with a random model is brutally slow. Random models play 150-200 move games. Each MCTS simulation calls the neural network, and with sequential execution (one call at a time), each game takes 2-3 minutes.
- **Tradeoff**: Fewer games = less training data per iteration. But the model will still learn because:
  - 30 games × 150 moves = 4500 positions per iteration (enough to train on)
  - Once the model improves (games become 30-50 moves), we can increase settings
- **Future**: Once parallel_mcts.py is fixed, increase to 100-200 games.

### 5. Disabling Parallel MCTS
- **Why**: The batched implementation hangs. It has a bug in the game-completion detection or the MCTS traversal loop.
- **Impact**: GPU utilization stays at ~5% (most time is CPU doing tree traversal, GPU called one state at a time).
- **Fix needed**: Debug `parallel_mcts.py` — likely an infinite loop where games never terminate or the batch collection blocks.

### 6. Output Buffering
- **Why it happens**: Python buffers stdout when not connected to a terminal (as with `nohup`). Even `python3 -u` doesn't fully fix it when output goes through pipes.
- **Workaround**: Use `stdbuf -oL` or `PYTHONUNBUFFERED=1`. In practice, the print only fires every 10 games so you wait 10+ minutes between log updates regardless.
- **Real fix**: Write a progress file that updates every game (or use TensorBoard which writes to disk immediately).

---

## Issues Explained

### Why GPU Utilization Is Low (4%)
The MCTS algorithm is sequential by nature:
1. Walk down tree (CPU) → 2. Evaluate one leaf (GPU) → 3. Backpropagate (CPU) → repeat

The GPU call takes 0.1ms but the overhead of sending data + CPU work between calls takes 2ms. So the GPU is idle 95% of the time.

**Solution**: Batch multiple games' leaf evaluations into one GPU call (what parallel_mcts.py attempts).

### Why Iteration 1 Is So Slow
A random neural network outputs uniform probabilities. MCTS with random priors explores very inefficiently, and the game goes on for 150-200 moves before someone accidentally wins.

After iteration 1, the model learns "move towards goal = good" and games drop to 50-80 moves. This means iteration 2 is 3x faster than iteration 1, iteration 5 is 5x faster, etc.

### Why the Log Appears Stuck
The print statement `Self-play: 10/30 games, X positions` only fires every 10 games. With each game taking 2-3 minutes, you wait 20-30 minutes between log updates. The training IS running — it just doesn't report until the next batch of 10 completes.

---

## Current Status

**Training**: Run 008, Iteration 1, 10/30 games completed
**Instance**: i-068eb17d4f1f7fc5b at 54.221.43.44 (g5.xlarge, us-east-1)
**Expected**: Iteration 1 completes in ~35 min total, subsequent iterations faster
**Cost so far**: ~$1-2 (instance running ~1-2 hours)

---

## What Would Make This Faster

| Improvement | Speedup | Effort |
|-------------|---------|--------|
| Fix parallel_mcts.py (batch inference) | 10-20x | Debug infinite loop |
| Reduce simulations for iter 1 only (25 sims, ramp to 200) | 4-8x for iter 1 | Easy config change |
| Use multiple GPU processes (torch.multiprocessing) | 3-4x | Medium rewrite |
| Use a pre-trained model as starting point | Skip iter 1-5 | Need to train one first |
| C++ MCTS with Python bindings (like real AlphaZero) | 50-100x | Major effort |
