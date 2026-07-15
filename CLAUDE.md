# Barricade

A Quoridor-style strategic board game with an AlphaZero neural network AI.

## Project Structure

```
engine/           Python game engine + AI
  game.py         Core game logic (9x9 board, walls, pathfinding)
  main.py         FastAPI server (REST API)
  model.py        ResNet neural network (policy + value heads)
  mcts.py         Monte Carlo Tree Search (PUCT + Dirichlet)
  self_play.py    Self-play game generation
  train.py        AlphaZero training loop
  arena.py        Model comparison (new vs old)
  ai_player.py    Inference wrapper for API
  requirements.txt

frontend/         React (Vite) web UI
  src/App.jsx     Menu + game screens
  src/index.css   All styling
  src/sounds.js   Web Audio sound effects
  src/main.jsx    Entry point

reference/        Target UI screenshots + CSS
  screenshots/    14 reference screenshots from barricade.gg and wrongway.app
  screenshot_final.mjs  Playwright script to recapture
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

### Train the AI
```bash
cd engine
python train.py --iterations 5 --self-play-games 10 --simulations 50
```

## Game Rules (Quoridor)

- 9×9 board, 2 players
- Player 1 (Red) starts at row 0 center, races to row 8
- Player 2 (Blue) starts at row 8 center, races to row 0
- Each turn: move one square OR place a wall
- Walls are 2-cells long (horizontal or vertical), 10 per player
- Walls cannot overlap, cross each other, or fully block a player's path
- Jump over adjacent opponent; diagonal jump if straight is blocked
- First to reach opposite side wins

## AI Architecture

- **Network**: ResNet with 6 residual blocks, 128 channels (2.4M params)
- **Input**: 12×9×9 planes encoding positions, walls, goals, legal moves
- **Output**: policy (209 actions) + value scalar [-1, 1]
- **Search**: PUCT-based MCTS with Dirichlet noise at root
- **Training**: self-play → train → arena → accept/reject

## Commands

- `npm run dev` — Start frontend dev server (port 5173)
- `npm run build` — Production build
- `python -m uvicorn main:app` — Start API server (port 8000)
- `python train.py` — Run training loop
- `python train.py --help` — Show training options
