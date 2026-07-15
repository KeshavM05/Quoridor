from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from game import QuoridorGame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_instance = QuoridorGame()
move_history = []  # List of (player, move_notation) tuples


def move_to_notation(move):
    """Convert a move tuple to algebraic notation like barricade.gg.

    Columns: a-i (left to right, 0-8)
    Rows: 1-9 (bottom to top, so engine row 8 = notation row 1, row 0 = notation row 9)

    Pawn: "e5" (column letter + row number)
    Wall: "He4" or "Vd3" (orientation + column + row of top-left corner)
    """
    cols = 'abcdefghi'
    if move[0] == 'move':
        r, c = move[1], move[2]
        row_num = 9 - r  # flip: engine row 0 = display row 9 (top)
        return f"{cols[c]}{row_num}"
    elif move[0] == 'wall':
        orient, r, c = move[1], move[2], move[3]
        row_num = 9 - r
        prefix = 'h' if orient == 'h' else 'v'
        return f"{prefix}{cols[c]}{row_num}"
    return "?"


def get_state_response():
    return {
        "board_size": game_instance.board_size,
        "p1_pos": {"r": game_instance.p1_pos[0], "c": game_instance.p1_pos[1]},
        "p2_pos": {"r": game_instance.p2_pos[0], "c": game_instance.p2_pos[1]},
        "p1_walls": game_instance.p1_walls,
        "p2_walls": game_instance.p2_walls,
        "current_player": game_instance.current_player,
        "h_walls": game_instance.h_walls.tolist(),
        "v_walls": game_instance.v_walls.tolist(),
        "winner": game_instance.get_winner(),
        "legal_moves": game_instance.get_legal_moves(),
        "move_history": move_history,
    }


class MoveRequest(BaseModel):
    move_type: str
    r: int
    c: int
    orient: Optional[str] = None


@app.get("/state")
def get_state():
    return get_state_response()


@app.post("/move")
def play_move(req: MoveRequest):
    winner = game_instance.get_winner()
    if winner != 0:
        raise HTTPException(status_code=400, detail="Game already over")

    legal_moves = game_instance.get_legal_moves()

    if req.move_type == 'move':
        move_tuple = ('move', req.r, req.c)
    elif req.move_type == 'wall':
        move_tuple = ('wall', req.orient, req.r, req.c)
    else:
        raise HTTPException(status_code=400, detail="Invalid move type")

    if move_tuple not in legal_moves:
        raise HTTPException(status_code=400, detail="Illegal move")

    player = game_instance.current_player
    notation = move_to_notation(move_tuple)
    move_history.append({"player": player, "notation": notation})

    game_instance.play_move(move_tuple)
    return get_state_response()


@app.post("/reset")
def reset_game():
    global game_instance, move_history
    game_instance = QuoridorGame()
    move_history = []
    return get_state_response()


@app.post("/undo")
def undo_move():
    """Undo is not trivial with the current engine (no state stack).
    For now, return an error. TODO: implement game state stack."""
    raise HTTPException(status_code=501, detail="Undo not yet implemented")


# ============ Replay Endpoints ============
import os
import json
import glob

TRAINING_RUNS_DIR = os.path.join(os.path.dirname(__file__), 'training_runs')


@app.get("/replays")
def list_replays():
    """List all saved game replays from training runs."""
    replays = []
    if not os.path.exists(TRAINING_RUNS_DIR):
        return replays

    for run_dir in sorted(glob.glob(os.path.join(TRAINING_RUNS_DIR, 'run_*'))):
        replays_dir = os.path.join(run_dir, 'replays')
        if not os.path.exists(replays_dir):
            continue
        for f in sorted(os.listdir(replays_dir)):
            if f.endswith('.json'):
                filepath = os.path.join(replays_dir, f)
                try:
                    with open(filepath, 'r') as fp:
                        data = json.load(fp)
                    replays.append({
                        "id": f.replace('.json', ''),
                        "iteration": data.get('iteration', 0),
                        "winner": data.get('winner', 0),
                        "length": data.get('length', len(data.get('moves', []))),
                        "label": data.get('label', ''),
                        "run": os.path.basename(run_dir),
                    })
                except (json.JSONDecodeError, IOError):
                    pass

    return replays


@app.get("/replays/{replay_id}")
def get_replay(replay_id: str):
    """Get a specific replay by ID."""
    if not os.path.exists(TRAINING_RUNS_DIR):
        raise HTTPException(status_code=404, detail="No training runs found")

    for run_dir in glob.glob(os.path.join(TRAINING_RUNS_DIR, 'run_*')):
        filepath = os.path.join(run_dir, 'replays', f'{replay_id}.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

    raise HTTPException(status_code=404, detail="Replay not found")
