# Research: State of the Art Quoridor AI (2025-2026)

Deep research conducted July 20, 2026. Sources: academic papers, open source repos, commercial products.

---

## Summary

The strongest known Quoridor AI is **Claustrophobia** (claustrophobia.dev) — an AlphaZero-style engine claiming superhuman play. The full 9×9 game remains unsolved (~10^20 wall configurations, branching factor ~104). The 5×5 board was fully solved in Feb 2026. The central challenge across ALL implementations is wall placement strategy.

---

## Strongest Known Implementations

### 1. Claustrophobia (claustrophobia.dev)
- **Type**: AlphaZero (policy-value network + MCTS)
- **Strength**: Claims superhuman (8 difficulty levels, "Pebble to Titan")
- **Training**: Pure self-play, no human data
- **Features**: Opening theory exploration from self-play games
- **Limitation**: No independent Elo verification or tournament results
- **Relevance to us**: This is exactly what we're building. Proof it's possible.

### 2. gorisanson/quoridor-ai (GitHub)
- **Type**: MCTS with heavy heuristic augmentation
- **Key innovation**: Filters wall placements to "probable" candidates only
  - Walls near pawns
  - Walls near existing walls
  - Leftmost/rightmost horizontal walls
- **Rollout policy**: Move along shortest path (70%), place random filtered wall (30%)
- **Performance**: 90% win rate vs Daniel Borowski's minimax AI with 60k rollouts
- **Relevance to us**: The wall filtering approach is the key technique we're missing

### 3. Grant Slatton — Solving 5×5 Quoridor (Feb 2026)
- **Method**: Proof-number search combined with alpha-beta
- **Result**: Fully solved 5×5 and most board configs with area ≤ 28
- **Key insight**: "Move generation is several orders of magnitude slower than chess due to BFS pathfinding"
- **Relevance**: Confirms BFS is THE bottleneck, validates our C++ approach

### 4. Failed AlphaZero Attempts (all have our same problems)
- **v-ade-r/QuoridorAI-AlphaZero**: "5-7h per iteration, still far from average-level, exhausts walls early"
- **xphoniex/alphazero-quoridor**: Reduced to 5×5 because 9×9 was too expensive
- **cryer/AlphaZero_Quoridor**: 5 residual blocks, 26×9×9 input, 140 actions, limited results

---

## What Actually Works (Ranked)

### 1. Wall Candidate Filtering (HIGHEST IMPACT)
**Instead of considering all ~128 wall placements, only consider walls that:**
- Increase opponent's BFS shortest path distance
- Are adjacent to existing walls
- Are near the opponent's pawn

This reduces branching factor from ~130 to ~15-20. Every wall considered during MCTS is meaningful, not random. The gorisanson bot specifically credits this as the reason it went from "poor performance" to 90% win rate.

### 2. BFS Path Distance as Primary Evaluation
The standard Quoridor evaluation:
```
Score = opponent_BFS_distance - my_BFS_distance + (my_walls - opponent_walls) * weight
```
Simple, effective, used by virtually all competitive Quoridor bots.

### 3. Guided Rollout Policy (for MCTS)
Instead of random rollouts, use:
- 70% probability: move along shortest BFS path
- 30% probability: place a filtered wall candidate
This makes every simulation informative.

### 4. AlphaZero Self-Play (expensive but ceiling is highest)
- Requires significant compute (5-7h per iteration on GPU)
- Eventually produces superhuman play (Claustrophobia claims this)
- Wall exhaustion is a known problem during training
- All open-source attempts remain weak — only Claustrophobia claims success

---

## The Wall Placement Problem

This is THE unsolved challenge. Every implementation struggles with it:

| Approach | How they handle walls | Result |
|----------|----------------------|--------|
| Pure MCTS | Consider all 128 positions | Poor (too many bad options dilute search) |
| Filtered MCTS | Only consider 5-15 useful positions | Good (90% vs minimax) |
| Minimax | BFS-diff eval, depth 2-3 | Decent but shallow |
| AlphaZero (untrained) | Network outputs from 209 actions | Dumps walls randomly |
| AlphaZero (well-trained) | Network learns which walls matter | Strong (Claustrophobia) |

**Key insight**: The AlphaZero approach CAN learn wall strategy, but it takes massive training. The shortcut used by MCTS bots (filtering) could be applied to our MCTS during self-play to accelerate learning.

---

## Implications for Our Project

### What we should implement:
1. **Filter wall candidates in MCTS** — only expand walls that increase opponent's BFS distance by ≥1. This is the single biggest improvement.
2. **The filtered walls become the training signal** — network learns "these are the walls worth considering" from MCTS data that only contains good walls.
3. **Keep the full 209-action output** — at inference time the network can still pick any wall, but training data only shows it walls that mattered.

### What we can skip:
- Solving the game (intractable for 9×9)
- Human game data (pure self-play works if training signal is good)
- Massive compute (wall filtering makes each simulation more useful = less compute needed)

---

## Architecture Comparison

| | Our System | Claustrophobia | gorisanson | Academic (Koirala 2024) |
|---|---|---|---|---|
| Algorithm | AlphaZero | AlphaZero | MCTS + heuristics | Minimax vs MCTS |
| Network | 12 blocks, 256ch, 20M | Unknown | None | None |
| Wall handling | Full 128 actions | Full (learned) | Filtered ~15 | BFS eval |
| Sims | 200 self-play | Unknown | 60,000 | 90-100 |
| Language | C++ + Python | Unknown | JavaScript | Python |
| Strength | Learning | Superhuman (claimed) | Strong | Moderate |

---

## Sources

1. claustrophobia.dev — Commercial AlphaZero Quoridor engine
2. github.com/gorisanson/quoridor-ai — MCTS with wall filtering
3. github.com/v-ade-r/QuoridorAI-AlphaZero — AlphaZero attempt (same problems as us)
4. github.com/xphoniex/alphazero-quoridor — 5×5 AlphaZero
5. github.com/cryer/AlphaZero_Quoridor — Architecture reference
6. github.com/suragnair/alpha-zero-general — Framework many build on
7. grantslatton.com/solving-quoridor — 5×5 solved (Feb 2026)
8. Koirala 2024 thesis (Charles University) — Minimax vs MCTS comparison
