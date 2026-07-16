# Decision Log

Every problem encountered, decision made, and the reasoning behind it.

---

## Problem 1: Which cloud provider to use for training
**Date**: 2026-07-15  
**Context**: Need GPU for training, user has $20K AWS credits  
**Decision**: AWS (g5.xlarge with A10G GPU, ~$1/hr)  
**Reasoning**: User already has AWS credits and CLI configured. No point using another provider.  
**Alternatives considered**: Vast.ai (cheaper), Lambda Labs (simpler), local GPU (no NVIDIA GPU available)

---

## Problem 2: GPU instance quota was 0 in us-east-2
**Date**: 2026-07-15  
**Context**: `aws ec2 run-instances` failed with VcpuLimitExceeded  
**Decision**: Switch to us-east-1 (had 32 vCPU quota)  
**Reasoning**: Requesting quota increase takes hours/days. us-east-1 already had capacity.  
**Impact**: Had to import SSH key pair and create security group in new region.

---

## Problem 3: SSH key permission denied
**Date**: 2026-07-15  
**Context**: `ssh ubuntu@instance` rejected the key pair  
**Root cause**: Key pair was previously imported to us-east-1 with a different public key than the local .pem file  
**Fix**: Deleted old key pair, re-imported from the actual .pem file, launched new instance  
**Lesson**: Always verify key pair fingerprint matches before launching instances

---

## Problem 4: Windows SSH "bad permissions" on .pem file
**Date**: 2026-07-15  
**Context**: `ssh -i "D:/Downloads/vla-key-pair.pem"` failed with "UNPROTECTED PRIVATE KEY FILE"  
**Fix**: Copied to `~/.ssh/vla-key-pair.pem` with chmod 600  
**Alternative**: `icacls` command in PowerShell to restrict permissions on original file  
**Lesson**: Windows SSH requires strict file permissions same as Linux

---

## Problem 5: Training log appeared stuck (output buffering)
**Date**: 2026-07-15  
**Context**: `train.log` showed no updates for 30+ minutes  
**Root cause**: Python buffers stdout when piped through nohup. Even `-u` flag doesn't fully fix it when `print()` is inside nested functions.  
**Partial fix**: `PYTHONUNBUFFERED=1` + `stdbuf -oL`  
**Real issue**: The print statements only fire every 10 games. Each game takes 2-3 minutes → 20-30 min between visible updates.  
**Lesson**: Don't rely on stdout for progress monitoring. Write progress to a file or use TensorBoard (writes to disk immediately).

---

## Problem 6: Training process files owned by root
**Date**: 2026-07-15  
**Context**: user-data script runs as root, creates files ubuntu user can't overwrite  
**Fix**: `sudo chown -R ubuntu:ubuntu /home/ubuntu/barricade/`  
**Lesson**: Always add `chown` to user-data scripts, or run the training command as the ubuntu user explicitly.

---

## Problem 7: Original training settings way too slow (200 games × 400 sims)
**Date**: 2026-07-15  
**Context**: Iteration 1 ran for 19 hours without completing  
**Root cause math**: 200 games × 200 moves × 400 sims × 2ms/sim = 32,000 seconds = 8.9 hours just for self-play. Plus arena (another 8+ hours).  
**Decision**: Reduce to 30 games × 50 sims  
**New math**: 30 × 150 × 50 × 2ms = 450 seconds = 7.5 min per 10-game batch  
**Tradeoff**: Less training data per iteration, but still enough to learn (4000-5000 positions)  
**Result**: Iteration 1 completed in ~1 hour

---

## Problem 8: Parallel MCTS hung indefinitely
**Date**: 2026-07-16  
**Context**: `parallel_mcts.py` designed to batch GPU calls across multiple games  
**Root cause**: O(n²) deepcopy issue. Every MCTS simulation stored a `deepcopy(game_state)` in the tree node. With 200 simulations, tree grows 200 levels deep. Each level copies the game state. Total copies: 1+2+3+...+200 = 20,100 per move.  
**Attempted fix 1**: Replay from root (copy once, replay moves to reach leaf). Still too slow in Python — BFS path validation + legal move generation at every replayed move.  
**Decision**: Disable parallel MCTS, use sequential (which works). Accept ~1hr/iteration.  
**Real fix**: C++ rewrite of game engine + MCTS (in progress)

---

## Problem 9: WebSocket viewer not connecting
**Date**: 2026-07-16  
**Context**: Frontend showed "WebSocket connection failed" when connecting to watch server  
**Root cause**: Uvicorn was running WITHOUT a WebSocket library. Error in logs: "No supported WebSocket library detected"  
**Fix**: `pip install websockets`  
**Lesson**: FastAPI WebSocket endpoints require either `websockets` or `wsproto` package installed alongside uvicorn.

---

## Problem 10: SSH tunnel didn't forward WebSocket
**Date**: 2026-07-16  
**Context**: Tried `ssh -L 8001:localhost:8001` to tunnel AWS watch server to local browser  
**Root cause**: Actually it was the missing websockets library on the server side (Problem 9), not the tunnel itself  
**Fix**: Once websockets was installed locally, the viewer worked without any tunnel  
**Decision**: Run watch server locally instead of tunneling from AWS

---

## Problem 11: Games too long (150-200 moves with random model)
**Date**: 2026-07-16  
**Context**: Random model wanders aimlessly, games hit 200 move cap  
**Impact**: Each iteration takes 3-4× longer than necessary  
**Decision**: Cap game length at 100 moves (declare draw)  
**Reasoning**: If neither player wins in 100 moves, both are playing poorly — no useful training signal after that point. Drawing is correct.  
**Expected speedup**: ~2× for early iterations

---

## Problem 12: Need 10-50× speedup for practical training
**Date**: 2026-07-16  
**Context**: 50 iterations at 1 hr each = 50 hours. Want it in 1-5 hours.  
**Decision**: Rewrite game engine + MCTS in C++  
**Reasoning**: 
- Python game logic is the bottleneck (95% of time between GPU calls)
- `get_legal_moves()` calls BFS, checks 128 wall positions, each with path validation
- In C++ this is 100-1000× faster (no interpreter overhead, cache-friendly structs)
- pybind11 lets us call it from Python seamlessly
- Only the neural network stays in Python/PyTorch (GPU calls)
**Alternative considered**: Cython (partial speedup, easier), Rust (harder bindings), numpy vectorization (limited gains for tree structures)  
**Status**: In progress

---

## Architecture Decisions

### Why AlphaZero over simpler approaches?
- Minimax: branching factor ~130 (too high for depth > 2-3)
- Pure MCTS (no neural net): works but converges very slowly without learned priors
- AlphaZero: neural net guides MCTS → much stronger at same compute budget
- The user also specifically wanted to train a neural network as a learning experience

### Why ResNet architecture?
- Proven for board games (AlphaGo, AlphaZero, MuZero all use it)
- Residual connections help with gradient flow in deeper networks
- 6 blocks × 128 channels = 2.4M params: small enough to train fast, big enough to learn Quoridor strategy

### Why 12 input planes?
- Positions (2 planes): who is where
- Walls remaining (2 planes): resource tracking
- Wall positions (2 planes): board structure
- Goal rows (2 planes): what each player is trying to reach
- Legal moves (1 plane): valid actions
- Player identity (1 plane): whose turn
- This gives the network everything it needs without redundancy

### Why 209 output actions (not fewer)?
- 81 pawn positions + 64 H walls + 64 V walls = 209
- Could reduce by only outputting "relative moves" (up/down/left/right/jump = ~12 actions)
- But flat output over all positions works better with ResNet (proven in AlphaZero paper)
- Illegal moves are masked after prediction — network never picks an illegal move

### Why self-play over human data?
- No human game database exists for Quoridor at scale
- Self-play generates unlimited data
- No human bias — can discover strategies humans haven't found
- Same approach works regardless of game rules
