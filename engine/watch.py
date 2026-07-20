"""
Live self-play game viewer.

Runs a separate FastAPI server on port 8001 that:
- Plays AI vs AI games using the neural network + MCTS
- Broadcasts each move over WebSocket to connected viewers
- Exposes control endpoints for starting games and adjusting speed
"""

import asyncio
import json
import copy
import sys
import os
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add engine directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import QuoridorGame
from model import QuoridorNet, encode_state, action_to_move, get_legal_action_mask, ACTION_SIZE
from mcts import MCTS


app = FastAPI(title="Barricade Watch Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- State ---

class WatchState:
    def __init__(self):
        self.game: QuoridorGame = None
        self.move_delay: float = 0.3
        self.is_running: bool = False
        self.is_paused: bool = False
        self.move_count: int = 0
        self.games_played: int = 0
        self.winner: int = 0
        self.connected_clients: List[WebSocket] = []
        self.task: asyncio.Task = None
        self.model: QuoridorNet = None
        self.mcts: MCTS = None
        self.red_model: QuoridorNet = None
        self.red_mcts: MCTS = None
        self.blue_model: QuoridorNet = None
        self.blue_mcts: MCTS = None
        self.device: str = 'cpu'

    def load_model(self, model_id='best'):
        """Load a specific model by ID. Returns a (model, mcts) tuple."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = QuoridorNet()
        model.to(self.device)

        if model_id == 'random':
            print(f"Using random (untrained) model")
        else:
            checkpoint_path = self._resolve_model_path(model_id)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                model.load_state_dict(checkpoint)
                print(f"Loaded model: {model_id} from {checkpoint_path}")
            else:
                print(f"Model '{model_id}' not found, using random weights")

        model.eval()
        mcts = MCTS(model, device=self.device, num_simulations=10)
        return model, mcts

    def _resolve_model_path(self, model_id):
        """Find the checkpoint file for a model ID."""
        base = os.path.dirname(__file__)
        if model_id == 'best':
            for path in [
                os.path.join(base, 'checkpoints', 'best_model.pt'),
                os.path.join(base, 'checkpoint.pt'),
            ]:
                if os.path.exists(path):
                    return path
        elif model_id.startswith('iter_'):
            path = os.path.join(base, 'checkpoints', f'model_{model_id}.pt')
            if os.path.exists(path):
                return path
        return None

    def init_model(self):
        """Load default model for backward compatibility."""
        self.model, self.mcts = self.load_model('best')


state = WatchState()


# --- Helpers ---

def game_state_to_dict(game: QuoridorGame, move_count: int, games_played: int) -> dict:
    """Serialize game state for WebSocket broadcast."""
    return {
        "type": "game_state",
        "board_size": game.board_size,
        "p1_pos": {"r": int(game.p1_pos[0]), "c": int(game.p1_pos[1])},
        "p2_pos": {"r": int(game.p2_pos[0]), "c": int(game.p2_pos[1])},
        "p1_walls": int(game.p1_walls),
        "p2_walls": int(game.p2_walls),
        "current_player": int(game.current_player),
        "h_walls": game.h_walls.astype(int).tolist(),
        "v_walls": game.v_walls.astype(int).tolist(),
        "winner": int(game.get_winner()),
        "move_count": move_count,
        "games_played": games_played,
    }


async def broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    data = json.dumps(message)
    disconnected = []
    for ws in state.connected_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in state.connected_clients:
            state.connected_clients.remove(ws)


async def self_play_loop():
    """Main loop: plays games continuously, broadcasting each move."""
    # Use per-side models if set, otherwise fall back to default
    if state.red_mcts is None or state.blue_mcts is None:
        state.init_model()
        state.red_mcts = state.mcts
        state.blue_mcts = state.mcts

    while state.is_running:
        # Start a new game
        state.game = QuoridorGame()
        state.move_count = 0
        state.winner = 0
        state.games_played += 1

        # Broadcast initial state
        await broadcast(game_state_to_dict(state.game, state.move_count, state.games_played))

        # Play moves until game ends
        while state.game.get_winner() == 0 and state.is_running:
            # Wait while paused
            while state.is_paused and state.is_running:
                await asyncio.sleep(0.1)

            if not state.is_running:
                break

            # Use MCTS to select a move (use correct model per player)
            try:
                current_mcts = state.red_mcts if state.game.current_player == 1 else state.blue_mcts
                action_probs = current_mcts.search(
                    state.game,
                    temperature=0.5,
                    add_noise=True
                )
            except Exception as e:
                print(f"MCTS error: {e}")
                # Fallback: pick a random legal move
                legal_moves = state.game.get_legal_moves()
                if not legal_moves:
                    break
                move = legal_moves[np.random.randint(len(legal_moves))]
                state.game.play_move(move)
                state.move_count += 1
                await broadcast(game_state_to_dict(state.game, state.move_count, state.games_played))
                await asyncio.sleep(state.move_delay)
                continue

            # Sample action from distribution
            action = np.random.choice(ACTION_SIZE, p=action_probs)
            move = action_to_move(action)

            # Validate move is legal (safety check)
            legal_moves = state.game.get_legal_moves()
            if move not in legal_moves:
                # Fallback: pick highest-probability legal move
                legal_mask = get_legal_action_mask(state.game)
                masked_probs = action_probs * legal_mask
                if masked_probs.sum() > 0:
                    masked_probs /= masked_probs.sum()
                    action = np.random.choice(ACTION_SIZE, p=masked_probs)
                    move = action_to_move(action)
                else:
                    move = legal_moves[0] if legal_moves else None

            if move is None:
                break

            state.game.play_move(move)
            state.move_count += 1

            # Broadcast updated state
            await broadcast(game_state_to_dict(state.game, state.move_count, state.games_played))

            # Delay between moves
            await asyncio.sleep(state.move_delay)

        # Game over
        state.winner = state.game.get_winner()
        await broadcast({
            "type": "game_over",
            "winner": state.winner,
            "move_count": state.move_count,
            "games_played": state.games_played,
        })

        # Pause before starting next game
        await asyncio.sleep(3.0)

    state.is_running = False


# --- WebSocket Endpoint ---

@app.websocket("/ws/watch")
async def websocket_watch(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.append(websocket)

    # Send current state if a game is in progress
    if state.game is not None:
        await websocket.send_text(json.dumps(
            game_state_to_dict(state.game, state.move_count, state.games_played)
        ))
    else:
        await websocket.send_text(json.dumps({
            "type": "waiting",
            "message": "No game in progress. Start one via POST /watch/start"
        }))

    try:
        while True:
            # Keep connection alive; handle incoming messages
            data = await websocket.receive_text()
            # Clients can send commands via websocket too
            try:
                msg = json.loads(data)
                if msg.get("action") == "pause":
                    state.is_paused = True
                elif msg.get("action") == "resume":
                    state.is_paused = False
                elif msg.get("action") == "speed":
                    delay = msg.get("delay", 1.0)
                    state.move_delay = max(0.01, min(5.0, float(delay)))
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


# --- REST Endpoints ---

class SpeedRequest(BaseModel):
    delay: float

class StartRequest(BaseModel):
    red_model: str = 'best'
    blue_model: str = 'best'


@app.get("/watch/models")
async def list_models():
    """List available model checkpoints."""
    base = os.path.join(os.path.dirname(__file__), 'checkpoints')
    models = []
    if os.path.exists(base):
        for f in sorted(os.listdir(base)):
            if f.startswith('model_iter_') and f.endswith('.pt'):
                name = f.replace('model_', '').replace('.pt', '')
                models.append(name)
    return models


@app.post("/watch/start")
async def start_watch(req: StartRequest = StartRequest()):
    """Start a new self-play game (or restart if one is running)."""
    if state.is_running:
        state.is_running = False
        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except (asyncio.CancelledError, Exception):
                pass

    # Load selected models for each side
    state.red_model, state.red_mcts = state.load_model(req.red_model)
    state.blue_model, state.blue_mcts = state.load_model(req.blue_model)

    state.is_running = True
    state.is_paused = False
    state.task = asyncio.create_task(self_play_loop())

    return {"status": "started", "delay": state.move_delay, "red": req.red_model, "blue": req.blue_model}


@app.post("/watch/stop")
async def stop_watch():
    """Stop the current self-play session."""
    state.is_running = False
    if state.task and not state.task.done():
        state.task.cancel()
        try:
            await state.task
        except (asyncio.CancelledError, Exception):
            pass
    return {"status": "stopped"}


@app.post("/watch/pause")
async def pause_watch():
    """Pause/unpause the current game."""
    state.is_paused = not state.is_paused
    return {"status": "paused" if state.is_paused else "playing"}


@app.post("/watch/speed")
async def set_speed(req: SpeedRequest):
    """Change the delay between moves (seconds)."""
    state.move_delay = max(0.01, min(5.0, req.delay))
    return {"delay": state.move_delay}


@app.get("/watch/status")
async def get_status():
    """Get current watch session status."""
    return {
        "is_running": state.is_running,
        "is_paused": state.is_paused,
        "move_delay": state.move_delay,
        "move_count": state.move_count,
        "games_played": state.games_played,
        "winner": state.winner,
        "connected_viewers": len(state.connected_clients),
        "current_player": state.game.current_player if state.game else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
