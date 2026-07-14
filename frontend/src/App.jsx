import React, { useState, useEffect } from 'react';
import './index.css';

const API_URL = 'http://localhost:8000';
const BOARD_SIZE = 9;

export default function App() {
  const [gameState, setGameState] = useState(null);
  const [mode, setMode] = useState('move'); // 'move', 'wall_h', 'wall_v'
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchState();
  }, []);

  const fetchState = async () => {
    try {
      const res = await fetch(`${API_URL}/state`);
      const data = await res.json();
      setGameState(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError('Failed to connect to Python Engine.');
    }
  };

  const playMove = async (move_type, r, c, orient = null) => {
    try {
      const res = await fetch(`${API_URL}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move_type, r, c, orient })
      });
      const data = await res.json();
      if (!res.ok) {
        // Show illegal move feedback briefly (could add toast here)
        console.warn(data.detail);
        return;
      }
      setGameState(data);
    } catch (err) {
      console.error(err);
    }
  };

  const resetGame = async () => {
    try {
      const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
      setGameState(await res.json());
      setMode('move');
    } catch(e) {}
  };

  if (!gameState) return <div style={{color:'white', padding: 20}}>Loading engine...</div>;

  const { p1_pos, p2_pos, p1_walls, p2_walls, current_player, h_walls, v_walls, winner, legal_moves } = gameState;

  const handleCellClick = (r, c) => {
    if (mode === 'move') {
      playMove('move', r, c);
    } else if (mode === 'wall_h') {
      // For H wall, clicking cell (r,c) attempts to place wall between row r and r+1, spanning col c to c+1
      if (r < 8 && c < 8) playMove('wall', r, c, 'h');
    } else if (mode === 'wall_v') {
      if (r < 8 && c < 8) playMove('wall', r, c, 'v');
    }
  };

  const renderPlayerCard = (num, pos, walls) => {
    const isLive = current_player === num;
    const isP1 = num === 1;
    return (
      <div className={`ww-pcard ${isP1 ? 'pA' : 'pB'} ${isLive && winner === 0 ? 'live' : 'dim'}`}>
        {isP1 && <div className="ww-figslot">P1</div>}
        <div className="ww-pinfo" style={{ textAlign: isP1 ? 'left' : 'right' }}>
          <div className="ww-pname">Player {num}</div>
          <div className="ww-pwalls">{walls} walls left</div>
        </div>
        {!isP1 && <div className="ww-figslot">P2</div>}
      </div>
    );
  };

  const renderCells = () => {
    const cells = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        // Highlight legal move cells if in move mode
        let isLegalMove = false;
        if (mode === 'move' && current_player !== winner) {
           isLegalMove = legal_moves.some(m => m[0] === 'move' && m[1] === r && m[2] === c);
        }

        cells.push(
          <div 
            key={`cell-${r}-${c}`} 
            className="cell" 
            onClick={() => handleCellClick(r, c)}
            style={{
              boxShadow: isLegalMove ? 'inset 0 0 15px rgba(251,191,36,0.3)' : undefined
            }}
          >
            {p1_pos.r === r && p1_pos.c === c && <div className="pawn p1"></div>}
            {p2_pos.r === r && p2_pos.c === c && <div className="pawn p2"></div>}
          </div>
        );
      }
    }
    return cells;
  };

  const renderWalls = () => {
    const walls = [];
    // Board is essentially 9 cells + 8 gaps. Cell=1fr, gap=6px.
    // Calculations using CSS variables or exact percentages are tricky without a fixed layout.
    // A simpler way: position absolute using calc.
    // grid size = 100%, each cell is ~11.11%. Gap is 6px.
    // Instead of precise pixel math which is fragile, let's use a simpler mapping.
    // Let's rely on standard % placement.
    
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        if (h_walls[r][c]) {
          walls.push(
            <div key={`hw-${r}-${c}`} className="wall-h" style={{
              top: `calc(${((r + 1) * 100) / 9}% - 3px)`,
              left: `calc(${(c * 100) / 9}%)`,
              width: `calc(${(200) / 9}% + 6px)`,
              height: '6px',
              position: 'absolute',
              background: 'var(--wall-color)',
              borderRadius: '3px',
              zIndex: 20
            }}></div>
          );
        }
        if (v_walls[r][c]) {
          walls.push(
            <div key={`vw-${r}-${c}`} className="wall-v" style={{
              top: `calc(${(r * 100) / 9}%)`,
              left: `calc(${((c + 1) * 100) / 9}% - 3px)`,
              width: '6px',
              height: `calc(${(200) / 9}% + 6px)`,
              position: 'absolute',
              background: 'var(--wall-color)',
              borderRadius: '3px',
              zIndex: 20
            }}></div>
          );
        }
      }
    }
    return walls;
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Barricade</h1>
        {error && <p style={{color: 'red'}}>{error}</p>}
        {winner !== 0 && <p style={{color: 'var(--accent)', fontWeight: 'bold', marginTop: 5}}>Player {winner} wins!</p>}
      </div>

      <div className="ww-players">
        {renderPlayerCard(1, p1_pos, p1_walls)}
        {renderPlayerCard(2, p2_pos, p2_walls)}
      </div>

      <div className="board-container">
        <div className="board-wrapper">
          <div className="board-grid">
            {renderCells()}
          </div>
          {renderWalls()}
        </div>
      </div>

      <div className="ww-bottombar">
        <div 
          className={`ww-wallctl ${mode === 'move' ? 'active' : ''}`}
          onClick={() => setMode('move')}
        >
          MOVE
        </div>
        <div 
          className={`ww-wallctl ${mode === 'wall_h' ? 'active' : ''}`}
          onClick={() => setMode('wall_h')}
        >
          WALL H
        </div>
        <div 
          className={`ww-wallctl ${mode === 'wall_v' ? 'active' : ''}`}
          onClick={() => setMode('wall_v')}
        >
          WALL V
        </div>
      </div>
      
      {winner !== 0 && (
        <div style={{textAlign: 'center', marginTop: 10}}>
          <button onClick={resetGame} style={{padding: '8px 16px', background: 'var(--accent)', border: 'none', borderRadius: 8, cursor: 'pointer', fontWeight: 'bold'}}>Play Again</button>
        </div>
      )}
    </div>
  );
}
