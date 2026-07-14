# Barricade 1:1 Clone and AI Implementation Plan

We are building a 1:1 clone of `barricade.gg` and `wrongway.app` (a Quoridor-style game) combined with a Python-based AlphaZero AI engine.

## Current Progress
- **UI/UX Foundation**: The React (Vite) frontend has been set up with the exact PWA styling, gradients, grid systems, and animations extracted from the scraped source code.
- **Python Engine**: The core game logic (move validation, wall placement, BFS pathfinding) is written in `engine/game.py`.

## Proposed Architecture & Next Steps

To fully implement the game and connect the beautiful UI to the brain, I propose the following architecture:

### 1. The Game Server (Python API)
We need the React frontend to communicate with the Python logic. I will wrap `engine/game.py` in a **FastAPI** web server. 
- The React app will send moves to the API.
- The Python API will validate the move, update the board state, and respond with the new state.

### 2. The React Frontend Integration
I will update the `App.jsx` and `Board.jsx` to:
- Render the exact grid intersection logic for wall placements (hovering between cells shows a preview wall).
- Make HTTP requests to the Python API when a player moves a pawn or places a wall.
- Animate the pawn moving to the new cell based on the API response.

### 3. The AlphaZero AI (Future Phase)
Once the human vs human gameplay is flawless over the API, I will implement the neural network in PyTorch within the Python engine. The React app will just pass the turn to the AI, which will use Monte Carlo Tree Search (MCTS) to respond.
