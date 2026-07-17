# Vision: Building the Best Quoridor AI

The goal: build the strongest Quoridor AI ever made, and document the entire journey as a portfolio/learning project.

No one has published a proper AlphaZero-trained Quoridor bot. Existing bots on barricade.gg use heuristic alpha-beta search. A well-trained neural MCTS model would beat them all.

---

## The Roadmap

### Phase 1: Foundation (current)
**Status**: Running on AWS

- ✅ C++ game engine + MCTS (100-1000x faster than Python)
- ✅ 12-block, 256-channel ResNet (20M params)
- ✅ 800 MCTS simulations per move
- ✅ Partial credit for draws (distance-to-goal signal)
- ✅ 500 move cap (games end naturally)
- ✅ Auto-documenting training journal
- ✅ Live game viewer
- 🔄 200 iterations running (~5 days)
- Expected result: model that plays purposeful Quoridor, moves toward goal, places some walls

### Phase 2: Scale Up
**Cost**: ~$200-300 | **Time**: 1-2 weeks

- 500+ iterations (more learning cycles)
- 200 games per iteration (more data per cycle)
- 1600 MCTS simulations (much deeper thinking during self-play)
- Self-play curriculum: start with fewer sims, ramp up as model improves
- Temperature annealing: more exploration early, exploitation later
- Expected result: strategic wall placement, shortest-path awareness, beating naive human players

### Phase 3: Refinements
**Cost**: ~$300-500 | **Time**: 2-4 weeks

- **Opening book extraction**: analyze the first 5-10 moves from thousands of games, save the best openings
- **Endgame solver**: when players are within 3-4 moves of goal, switch to perfect minimax (game is simple enough to brute-force at that depth)
- **Opponent pool**: play against all past model versions (not just the current one). Prevents forgetting old strategies.
- **Progressive widening**: don't waste MCTS simulations on obviously bad moves. Focus search on promising branches.
- **ELO tracking**: assign a rating to each checkpoint. Plot strength over time.
- **KataGo-style tricks**: dynamic komi, playout cap randomization, auxiliary policy targets
- Expected result: expert-level play, beating strong human players on barricade.gg

### Phase 4: Unbeatable
**Cost**: ~$500-1000 | **Time**: 1-2 months

- 1000+ iterations of training
- 1600 sims at inference time (1-2 seconds per move, but optimal play)
- Network distillation: train a smaller "fast" network from the big one (for real-time play)
- Exhaustive opening analysis (like chess opening theory)
- Formal verification of key endgame positions
- Expected result: strongest Quoridor bot in existence

---

## What Makes This a Portfolio Piece

### The Technical Story
1. "I built an AlphaZero system from scratch for Quoridor"
2. "I wrote a C++ game engine with pybind11 bindings for 100x speedup"
3. "I trained on AWS GPU instances, documented every decision"
4. "The AI discovered strategies no human taught it"

### The Artifacts
- Full source code (game engine, neural net, MCTS, training loop)
- Training progression (loss curves, ELO chart, game length over time)
- Game replays showing the AI evolving (random → rushing → strategic)
- Decision log explaining every tradeoff
- Live demo where anyone can play against it or watch it play

### The Narrative
- "Iteration 1: random moves, 200-move games"
- "Iteration 10: learned to move forward, 50-move games"
- "Iteration 50: started placing walls strategically"
- "Iteration 100: discovered opening theory on its own"
- "Iteration 200: beats every human I've tested against"

---

## Why This Would Be the Best

| Existing Quoridor bots | Our approach |
|------------------------|--------------|
| Alpha-beta with hand-crafted eval | Neural network learns its own eval |
| Search depth limited to 3-5 moves | MCTS searches 800+ simulations deep |
| Fixed strategy (coded by humans) | Discovers strategy through self-play |
| No learning, no improvement | Gets stronger with more training |
| Can be exploited by humans who know the eval | No fixed weaknesses to exploit |

No one has published a neural MCTS Quoridor bot trained at this scale. This would be the first.

---

## Compute Budget

| Phase | Iterations | Games | Sims | Time | Cost |
|-------|-----------|-------|------|------|------|
| 1 (current) | 200 | 20,000 | 800 | ~5 days | ~$120 |
| 2 | 500 | 100,000 | 1600 | ~2 weeks | ~$300 |
| 3 | 500 | 250,000 | 1600 | ~3 weeks | ~$500 |
| 4 | 1000 | 500,000 | 1600 | ~2 months | ~$1000 |
| **Total** | | | | | **~$2000** |

All well within the $20K AWS budget. We'd use 10% of available credits to build the strongest Quoridor AI ever made.

---

## Measuring Success

- **vs Random**: should win 100% by iteration 5
- **vs Heuristic bot** (shortest path + simple walls): should win 90%+ by iteration 50
- **vs Strong human**: should win 70%+ by iteration 100
- **vs barricade.gg top bots**: should win 60%+ by iteration 200
- **vs barricade.gg leaderboard players**: target top 10 by Phase 3
