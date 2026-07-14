from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from game import QuoridorGame

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game instance for now
game_instance = QuoridorGame()

class MoveRequest(BaseModel):
    move_type: str # 'move' or 'wall'
    r: int
    c: int
    orient: Optional[str] = None # 'h' or 'v' if wall

@app.get("/state")
def get_state():
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
        "legal_moves": game_instance.get_legal_moves()
    }

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
        
    game_instance.play_move(move_tuple)
    return get_state()

@app.post("/reset")
def reset_game():
    global game_instance
    game_instance = QuoridorGame()
    return get_state()
