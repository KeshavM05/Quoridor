import { useState, useEffect, useCallback, useRef } from 'react';
import './index.css';

const API_URL = 'http://localhost:8000';
const BOARD_SIZE = 9;

export default function App() {
  const [gameState, setGameState] = useState(null);
  const [mode, setMode] = useState('move'); // 'move', 'wall_h', 'wall_v'
  const [wallPreview, setWallPreview] = useState(null);
  const [error, setError] = useState(null);
  const boardRef = useRef(null);

  useEffect(() => { fetchState(); }, []);

  const fetchState = async () => {
    try {
      const res = await fetch(`${API_URL}/state`);
      setGameState(await res.json());
      setError(null);
    } catch {
      setError('Cannot connect to engine. Start the Python server.');
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
      if (!res.ok) return;
      setGameState(data);
      setWallPreview(null);
    } catch {}
  };

  const resetGame = async () => {
    try {
      const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
      setGameState(await res.json());
      setMode('move');
      setWallPreview(null);
    } catch {}
  };

  const handleCellClick = useCallback((r, c) => {
    if (!gameState || gameState.winner !== 0) return;
    if (mode === 'move') {
      playMove('move', r, c);
    } else if (mode === 'wall_h') {
      if (r < 8 && c < 8) playMove('wall', r, c, 'h');
    } else if (mode === 'wall_v') {
      if (r < 8 && c < 8) playMove('wall', r, c, 'v');
    }
  }, [gameState, mode]);

  const handleCellHover = useCallback((r, c) => {
    if (!gameState || gameState.winner !== 0) return;
    if (mode === 'wall_h' && r < 8 && c < 8) {
      setWallPreview({ orient: 'h', r, c });
    } else if (mode === 'wall_v' && r < 8 && c < 8) {
      setWallPreview({ orient: 'v', r, c });
    } else {
      setWallPreview(null);
    }
  }, [gameState, mode]);

  if (error) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: 32 }}>
        <div>
          <div style={{ fontSize: 32, marginBottom: 12 }}>🎮</div>
          <p style={{ color: 'var(--muted)', fontSize: 14, lineHeight: 1.5 }}>{error}</p>
          <button onClick={fetchState} style={{ marginTop: 16, padding: '8px 20px', borderRadius: 10, background: 'var(--accent)', color: '#000', fontWeight: 800, fontSize: 13 }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!gameState) return null;

  const { p1_pos, p2_pos, p1_walls, p2_walls, current_player, h_walls, v_walls, winner, legal_moves } = gameState;
  const wallsLeft = current_player === 1 ? p1_walls : p2_walls;

  return (
    <>
      {/* Player cards */}
      <PlayerCards
        current_player={current_player}
        p1_walls={p1_walls}
        p2_walls={p2_walls}
        winner={winner}
      />

      {/* Turn indicator */}
      {winner === 0 && (
        <div className={`turn-indicator p${current_player}`}>
          <div className="line" style={{ background: 'currentColor' }} />
          <span className="label">
            {current_player === 1 ? '▼ RED' : '▲ BLUE'}
          </span>
          <div className="line" style={{ background: 'currentColor' }} />
        </div>
      )}

      {/* Board */}
      <div className="board-container">
        <div className="board-frame" ref={boardRef}>
          <div className="board-stud left" />
          <div className="board-stud right" />
          <div className="board-inner">
            <BoardGrid
              p1_pos={p1_pos}
              p2_pos={p2_pos}
              current_player={current_player}
              legal_moves={legal_moves}
              mode={mode}
              winner={winner}
              onCellClick={handleCellClick}
              onCellHover={handleCellHover}
            />
            <Walls h_walls={h_walls} v_walls={v_walls} />
            {wallPreview && <WallPreviewEl preview={wallPreview} />}
            {winner !== 0 && (
              <div className={`winner-banner p${winner}`}>
                <h2>{winner === 1 ? 'Red' : 'Blue'} Wins!</h2>
                <button onClick={resetGame}>Play Again</button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="controls">
        <button
          className={`ctrl-btn ${mode === 'move' ? 'active' : ''}`}
          onClick={() => { setMode('move'); setWallPreview(null); }}
        >
          <span className="icon">👆</span>
          <span>MOVE</span>
        </button>
        <button
          className={`ctrl-btn ${mode === 'wall_h' ? 'active' : ''}`}
          onClick={() => setMode('wall_h')}
        >
          <span className="icon">━</span>
          <span>HORIZONTAL</span>
          <span className="walls-count">{wallsLeft} LEFT</span>
        </button>
        <button
          className={`ctrl-btn ${mode === 'wall_v' ? 'active' : ''}`}
          onClick={() => setMode('wall_v')}
        >
          <span className="icon">┃</span>
          <span>VERTICAL</span>
          <span className="walls-count">{wallsLeft} LEFT</span>
        </button>
      </div>

      {/* Actions */}
      <div className="actions">
        <button className="action-btn" onClick={resetGame}>New Game</button>
      </div>
    </>
  );
}

function PlayerCards({ current_player, p1_walls, p2_walls, winner }) {
  const p1Live = current_player === 1 && winner === 0;
  const p2Live = current_player === 2 && winner === 0;

  return (
    <div className="players">
      <div className={`pcard p1 ${p1Live ? 'live' : 'dim'}`}>
        <div className="sheen" />
        <div className="pcard-fig"><div className="stone" /></div>
        <div className="pcard-info">
          <div className="pcard-name">Red</div>
          <div className="pcard-walls">━ {p1_walls}/10</div>
        </div>
      </div>

      <div className="vs-pill">
        <div className="vs-dots"><i /><i /><i /></div>
        <span className="vs-text">VS</span>
        <div className="vs-dots"><i /><i /><i /></div>
      </div>

      <div className={`pcard p2 ${p2Live ? 'live' : 'dim'}`}>
        <div className="sheen" />
        <div className="pcard-fig"><div className="stone" /></div>
        <div className="pcard-info" style={{ textAlign: 'right' }}>
          <div className="pcard-name">Blue</div>
          <div className="pcard-walls" style={{ justifyContent: 'flex-end' }}>{p2_walls}/10 ━</div>
        </div>
      </div>
    </div>
  );
}

function BoardGrid({ p1_pos, p2_pos, current_player, legal_moves, mode, winner, onCellClick, onCellHover }) {
  const cells = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const isP1 = p1_pos.r === r && p1_pos.c === c;
      const isP2 = p2_pos.r === r && p2_pos.c === c;
      const isLegal = mode === 'move' && winner === 0 &&
        legal_moves.some(m => m[0] === 'move' && m[1] === r && m[2] === c);

      let goalClass = '';
      if (r === 8) goalClass = 'goal-p1'; // p1 races to row 8
      if (r === 0) goalClass = 'goal-p2'; // p2 races to row 0

      cells.push(
        <div
          key={`${r}-${c}`}
          className={`cell ${isLegal ? 'legal-move' : ''} ${goalClass}`}
          onClick={() => onCellClick(r, c)}
          onMouseEnter={() => onCellHover(r, c)}
          onMouseLeave={() => onCellHover(-1, -1)}
        >
          {isP1 && <div className={`pawn p1 ${current_player === 1 && winner === 0 ? 'active' : ''}`} />}
          {isP2 && <div className={`pawn p2 ${current_player === 2 && winner === 0 ? 'active' : ''}`} />}
        </div>
      );
    }
  }

  return <div className="board-grid">{cells}</div>;
}

function Walls({ h_walls, v_walls }) {
  const walls = [];
  const pct = 100 / BOARD_SIZE;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (h_walls[r][c]) {
        walls.push(
          <div
            key={`hw-${r}-${c}`}
            className="wall wall-h"
            style={{
              top: `calc(${(r + 1) * pct}% - 3px)`,
              left: `calc(${c * pct}% + 2px)`,
              width: `calc(${2 * pct}% - 0px)`,
            }}
          />
        );
      }
      if (v_walls[r][c]) {
        walls.push(
          <div
            key={`vw-${r}-${c}`}
            className="wall wall-v"
            style={{
              top: `calc(${r * pct}% + 2px)`,
              left: `calc(${(c + 1) * pct}% - 3px)`,
              height: `calc(${2 * pct}% - 0px)`,
            }}
          />
        );
      }
    }
  }

  return <>{walls}</>;
}

function WallPreviewEl({ preview }) {
  const pct = 100 / BOARD_SIZE;
  const { orient, r, c } = preview;

  if (orient === 'h') {
    return (
      <div
        className="wall wall-preview wall-h"
        style={{
          position: 'absolute',
          top: `calc(${(r + 1) * pct}% - 3px)`,
          left: `calc(${c * pct}% + 2px)`,
          width: `calc(${2 * pct}%)`,
          height: '6px',
        }}
      />
    );
  }

  return (
    <div
      className="wall wall-preview wall-v"
      style={{
        position: 'absolute',
        top: `calc(${r * pct}% + 2px)`,
        left: `calc(${(c + 1) * pct}% - 3px)`,
        width: '6px',
        height: `calc(${2 * pct}%)`,
      }}
    />
  );
}
