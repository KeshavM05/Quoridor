# Barricade

A Quoridor-style strategic board game with an AlphaZero neural network AI.

## Project Structure

```
engine/              Python game engine + AI
  game.py            Core game logic (9x9 board, walls, BFS pathfinding)
  main.py            FastAPI server (REST API on port 8000)
  model.py           ResNet neural network (policy + value heads, 2.4M params)
  mcts.py            Monte Carlo Tree Search (PUCT + Dirichlet noise)
  self_play.py       Self-play game generation for training
  train.py           AlphaZero training loop with TensorBoard logging
  arena.py           Model vs model comparison
  ai_player.py       Inference wrapper for serving AI moves via API
  watch.py           WebSocket server for live AI self-play viewing (port 8001)
  dashboard.py       Training metrics API endpoint
  requirements.txt

frontend/            React (Vite) web UI
  src/App.jsx        Menu + game screens + move history
  src/WatchGame.jsx  Live AI game viewer component
  src/index.css      All styling (mobile-first, no-scroll)
  src/sounds.js      Web Audio sound effects
  src/main.jsx       Entry point
  index.html         PWA meta tags

reference/           Design reference material
  screenshots/       Target screenshots from barricade.gg + wrongway.app
  screenshot_final.mjs   Playwright script to recapture references
  wrongway_app_styles.css  (gitignored) extracted CSS from wrongway.app
  barricade_gg_styles.css  (gitignored) extracted CSS from barricade.gg
```

## Running Locally

### Start the game engine
```bash
cd engine
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

### Start the frontend
```bash
cd frontend
npm install
npm run dev
```

### Watch AI play itself
```bash
cd engine
python watch.py
# Then open frontend → click "Watch AI"
```

### Train the AI
```bash
cd engine

# Quick local test (CPU, ~5 min)
python train.py --iterations 3 --self-play-games 5 --simulations 30

# Full training (AWS GPU, ~2-4 hours)
python train.py --iterations 50 --self-play-games 200 --simulations 400 --device cuda
```

### View training metrics
```bash
# TensorBoard (after training)
tensorboard --logdir engine/runs
# → open localhost:6006
```

## Game Rules (Quoridor)

- 9×9 board, 2 players
- Player 1 (Red) starts at top center (row 0, col 4), races to row 8 (bottom)
- Player 2 (Blue) starts at bottom center (row 8, col 4), races to row 0 (top)
- Each turn: move one square (orthogonal) OR place a wall
- Walls are 2-cells long (horizontal or vertical), each player has 10
- **Walls cannot**: overlap, cross each other, or fully block a player's path to their goal
- Path validation: BFS runs after every wall placement to ensure both players can still reach their goal
- Jump: if adjacent to opponent, jump over them to the square behind
- Diagonal jump: if straight jump is blocked by wall/edge, can jump diagonally to either side
- First to reach opposite row wins

## Algebraic Notation (matching barricade.gg)

- **Columns**: a–i (left to right, maps to col 0–8)
- **Rows**: 1–9 (bottom to top; engine row 0 = notation row 9, row 8 = row 1)
- **Pawn move**: `e5` (column letter + row number)
- **Horizontal wall**: `hd4` (lowercase 'h' + top-left corner position)
- **Vertical wall**: `vf7` (lowercase 'v' + top-left corner position)
- **Move history format**: `1. e2 e8` (move number, red's move, blue's move)

## AI Architecture

### Neural Network (model.py)
- **Type**: ResNet with 6 residual blocks, 128 channels
- **Parameters**: 2.4 million
- **Input**: 12×9×9 planes encoding:
  - Planes 0-1: current/opponent pawn positions (one-hot)
  - Planes 2-3: walls remaining (normalized scalar broadcast)
  - Planes 4-5: horizontal/vertical walls on board
  - Planes 8-9: goal rows for each player
  - Plane 10: legal pawn move mask
  - Plane 11: current player identity
- **Output**:
  - Policy head: 209 actions (81 pawn moves + 64 H walls + 64 V walls)
  - Value head: scalar in [-1, 1] (win probability)

### MCTS (mcts.py)
- PUCT selection: Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
- Dirichlet noise at root for exploration (α=0.3, ε=0.25)
- Configurable simulation count (default 100, use 400+ for strong play)

### Training Loop (train.py)
1. **Self-play**: current model plays itself (with temperature + noise)
2. **Train**: update network on accumulated (state, policy, value) tuples
3. **Arena**: pit new model vs old model (alternating colors)
4. **Accept/reject**: new model must win >55% to replace the best model
5. Repeat for N iterations

### Observability
- **TensorBoard** (`engine/runs/`): loss curves, arena win rates, game lengths
- **Dashboard API** (`/dashboard/metrics`): JSON metrics per iteration
- **Live Viewer** (port 8001): watch AI play in real-time via WebSocket

## Wall Drag & Drop (UX reference from barricade.gg)

- Wall pieces in dock are **player-colored** (red on red's turn, blue on blue's)
- Dock shows a short horizontal bar and a tall vertical bar
- **Grab**: cursor changes to grabbing, piece tilts slightly
- **Drag over board**: a GREEN preview appears at the nearest valid snap point
- **Drop**: wall snaps to grid, animates in, wall count decrements
- **Invalid drop**: wall returns to dock (rubber-band animation)
- Walls snap to **intersection points** between cells (the grid lines)

## Decision Logging

Whenever a problem is encountered, a design decision is made, or a tradeoff is chosen, **document it in `docs/DECISION_LOG.md`** with:
- Date
- Context (what was happening)
- Root cause (if it's a bug)
- Decision (what we chose to do)
- Reasoning (WHY — the key part)
- Alternatives considered
- Impact/tradeoff

Also document architecture decisions (why X over Y) and ML decisions (why these hyperparams, why this size network, etc.) in the same file.

The user wants to learn from this project — every decision should be explained in plain language with analogies where helpful.

## Cost Estimates (AWS Training)

| Instance | GPU | Cost/hr | Speed | 50 iterations |
|----------|-----|---------|-------|---------------|
| g5.xlarge | A10G 24GB | ~$1/hr | ~20s/game | ~$80-100 |
| p3.2xlarge | V100 16GB | ~$3/hr | ~15s/game | ~$150-200 |
| g5.12xlarge | 4× A10G | ~$5/hr | parallel | ~$50-80 |

## Commands Reference

| Command | Description |
|---------|-------------|
| `npm run dev` | Frontend dev server (port 5173) |
| `npm run build` | Production build |
| `python -m uvicorn main:app --port 8000` | Game API server |
| `python watch.py` | Live self-play viewer (port 8001) |
| `python train.py --help` | Training options |
| `tensorboard --logdir runs` | View training metrics |
