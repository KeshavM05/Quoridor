import { useState, useEffect, useCallback, useRef } from 'react';
import { playMove as sfxMove, playWall as sfxWall, playWin as sfxWin, playStart as sfxStart } from './sounds.js';
import WatchGame from './WatchGame.jsx';
import './index.css';

const API_URL = 'http://localhost:8000';
const BOARD_SIZE = 9;

export default function App() {
  const [screen, setScreen] = useState('menu'); // 'menu' | 'game' | 'watch'
  const [gameState, setGameState] = useState(null);
  const [error, setError] = useState(null);

  const fetchState = async () => {
    try {
      const res = await fetch(`${API_URL}/state`);
      setGameState(await res.json());
      setError(null);
      return true;
    } catch {
      setError('Cannot connect to engine.');
      return false;
    }
  };

  const startGame = async () => {
    try {
      await fetch(`${API_URL}/reset`, { method: 'POST' });
      const ok = await fetchState();
      if (ok) {
        setScreen('game');
        sfxStart();
      }
    } catch {
      setError('Cannot connect to engine.');
    }
  };

  if (screen === 'menu') {
    return <MenuScreen onStart={startGame} onWatch={() => setScreen('watch')} error={error} />;
  }

  if (screen === 'watch') {
    return <WatchGame onBack={() => setScreen('menu')} />;
  }

  return (
    <GameScreen
      gameState={gameState}
      setGameState={setGameState}
      onBack={() => setScreen('menu')}
    />
  );
}

// ============ MENU SCREEN ============
function MenuScreen({ onStart, onWatch, error }) {
  return (
    <div className="menu-screen">
      <div className="menu-header">
        <div className="menu-logo">
          <svg width="56" height="56" viewBox="0 0 48 48">
            <rect x="2" y="4" width="44" height="40" rx="7" fill="#7f1d1d"/>
            <rect x="4" y="6" width="18" height="10" rx="2.4" fill="#fb7185"/>
            <rect x="24" y="6" width="20" height="10" rx="2.4" fill="#be123c"/>
            <rect x="4" y="18" width="9" height="10" rx="2.4" fill="#be123c"/>
            <rect x="15" y="18" width="18" height="10" rx="2.4" fill="#fb7185"/>
            <rect x="35" y="18" width="9" height="10" rx="2.4" fill="#be123c"/>
            <rect x="4" y="30" width="20" height="10" rx="2.4" fill="#fb7185"/>
            <rect x="26" y="30" width="18" height="10" rx="2.4" fill="#be123c"/>
          </svg>
        </div>
        <h1 className="menu-title">Barricade</h1>
        <p className="menu-subtitle">STRATEGIC BOARD GAME</p>
      </div>

      <div className="menu-modes">
        <button className="menu-card primary" onClick={onStart}>
          <div className="menu-card-icon">⚔️</div>
          <div className="menu-card-text">
            <span className="menu-card-label">Local Game</span>
            <span className="menu-card-desc">2 players, one device</span>
          </div>
          <span className="menu-card-arrow">›</span>
        </button>

        <button className="menu-card" disabled>
          <div className="menu-card-icon">🤖</div>
          <div className="menu-card-text">
            <span className="menu-card-label">vs Computer</span>
            <span className="menu-card-desc">Coming soon</span>
          </div>
          <span className="menu-card-arrow">›</span>
        </button>

        <button className="menu-card" disabled>
          <div className="menu-card-icon">🌐</div>
          <div className="menu-card-text">
            <span className="menu-card-label">Play Online</span>
            <span className="menu-card-desc">Coming soon</span>
          </div>
          <span className="menu-card-arrow">›</span>
        </button>

        <button className="menu-card" onClick={onWatch}>
          <div className="menu-card-icon">👁️</div>
          <div className="menu-card-text">
            <span className="menu-card-label">Watch AI</span>
            <span className="menu-card-desc">Live self-play viewer</span>
          </div>
          <span className="menu-card-arrow">›</span>
        </button>
      </div>

      <div className="menu-rules">
        <div className="menu-rules-title">How to Play</div>
        <div className="menu-rules-grid">
          <div className="rule-item">
            <div className="rule-icon">🏃</div>
            <span>Race to the other side</span>
          </div>
          <div className="rule-item">
            <div className="rule-icon">🧱</div>
            <span>Place walls to block</span>
          </div>
          <div className="rule-item">
            <div className="rule-icon">⚡</div>
            <span>Jump over opponents</span>
          </div>
          <div className="rule-item">
            <div className="rule-icon">🚫</div>
            <span>Can't fully trap</span>
          </div>
        </div>
      </div>

      {error && <p className="menu-error">{error}</p>}

      <div className="menu-footer">
        <span>Barricade v0.1</span>
      </div>
    </div>
  );
}

// ============ GAME SCREEN ============
function GameScreen({ gameState, setGameState, onBack }) {
  const [dragState, setDragState] = useState(null); // { orient: 'h'|'v', x, y, boardPos: {r,c}|null }
  const boardRef = useRef(null);
  const boardRectRef = useRef(null);

  useEffect(() => {
    if (!gameState) return;
  }, [gameState]);

  const playMove = async (move_type, r, c, orient = null) => {
    try {
      const res = await fetch(`${API_URL}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move_type, r, c, orient })
      });
      const data = await res.json();
      if (!res.ok) return false;
      setGameState(data);
      if (move_type === 'move') sfxMove();
      else if (move_type === 'wall') sfxWall();
      if (data.winner !== 0) setTimeout(sfxWin, 300);
      return true;
    } catch { return false; }
  };

  const resetGame = async () => {
    try {
      const res = await fetch(`${API_URL}/reset`, { method: 'POST' });
      setGameState(await res.json());
    } catch {}
  };

  const handleCellClick = useCallback((r, c) => {
    if (!gameState || gameState.winner !== 0) return;
    if (!dragState) {
      playMove('move', r, c);
    }
  }, [gameState, dragState]);

  // --- Drag & Drop Wall Logic ---
  const getBoardPos = useCallback((clientX, clientY) => {
    const rect = boardRectRef.current;
    if (!rect) return null;
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const cellW = rect.width / BOARD_SIZE;
    const cellH = rect.height / BOARD_SIZE;
    const col = Math.floor(x / cellW);
    const row = Math.floor(y / cellH);
    if (row < 0 || row >= 8 || col < 0 || col >= 8) return null;
    return { r: row, c: col };
  }, []);

  const handleDragStart = useCallback((orient, e) => {
    if (!gameState || gameState.winner !== 0) return;
    const walls = gameState.current_player === 1 ? gameState.p1_walls : gameState.p2_walls;
    if (walls === 0) return;

    e.preventDefault();
    const touch = e.touches ? e.touches[0] : e;
    const boardEl = boardRef.current?.querySelector('.board-inner');
    if (boardEl) {
      boardRectRef.current = boardEl.getBoundingClientRect();
    }
    setDragState({ orient, x: touch.clientX, y: touch.clientY, boardPos: null });
  }, [gameState]);

  const handleDragMove = useCallback((e) => {
    if (!dragState) return;
    e.preventDefault();
    const touch = e.touches ? e.touches[0] : e;
    const pos = getBoardPos(touch.clientX, touch.clientY);
    setDragState(prev => ({ ...prev, x: touch.clientX, y: touch.clientY, boardPos: pos }));
  }, [dragState, getBoardPos]);

  const handleDragEnd = useCallback(async (e) => {
    if (!dragState) return;
    const touch = e.changedTouches ? e.changedTouches[0] : e;
    const pos = getBoardPos(touch.clientX, touch.clientY);
    if (pos) {
      await playMove('wall', pos.r, pos.c, dragState.orient);
    }
    setDragState(null);
  }, [dragState, getBoardPos]);

  useEffect(() => {
    if (!dragState) return;
    const onMove = (e) => handleDragMove(e);
    const onEnd = (e) => handleDragEnd(e);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onEnd);
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('touchend', onEnd);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onEnd);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onEnd);
    };
  }, [dragState, handleDragMove, handleDragEnd]);

  if (!gameState) return null;

  const { p1_pos, p2_pos, p1_walls, p2_walls, current_player, h_walls, v_walls, winner, legal_moves, move_history } = gameState;
  const wallsLeft = current_player === 1 ? p1_walls : p2_walls;

  return (
    <div className="game-screen">
      {/* Header with back + give up */}
      <div className="game-header">
        <button className="back-btn" onClick={onBack}>‹ Menu</button>
        <button className="giveup-btn" onClick={resetGame}>⚑ Give up</button>
      </div>

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
              winner={winner}
              onCellClick={handleCellClick}
            />
            <Walls h_walls={h_walls} v_walls={v_walls} />
            {dragState && dragState.boardPos && (
              <WallPreviewEl orient={dragState.orient} r={dragState.boardPos.r} c={dragState.boardPos.c} />
            )}
            {winner !== 0 && (
              <div className={`winner-banner p${winner}`}>
                <h2>{winner === 1 ? 'Red' : 'Blue'} Wins!</h2>
                <button onClick={resetGame}>Play Again</button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Wall dock - drag source */}
      <div className="wall-dock">
        <span className="wall-dock-label">{wallsLeft} WALLS LEFT</span>
        <div className="wall-sources">
          <div
            className={`wall-source h ${wallsLeft === 0 ? 'empty' : ''}`}
            onMouseDown={(e) => handleDragStart('h', e)}
            onTouchStart={(e) => handleDragStart('h', e)}
          >
            <div className="wall-source-preview wall-h-mini" />
            <span>Horizontal</span>
          </div>
          <div
            className={`wall-source v ${wallsLeft === 0 ? 'empty' : ''}`}
            onMouseDown={(e) => handleDragStart('v', e)}
            onTouchStart={(e) => handleDragStart('v', e)}
          >
            <div className="wall-source-preview wall-v-mini" />
            <span>Vertical</span>
          </div>
        </div>
      </div>

      {/* Move History */}
      {move_history && move_history.length > 0 && (
        <MoveHistory moves={move_history} />
      )}

      {/* Floating drag indicator */}
      {dragState && (
        <div
          className="drag-ghost"
          style={{ left: dragState.x, top: dragState.y }}
        >
          <div className={`drag-wall ${dragState.orient === 'h' ? 'drag-h' : 'drag-v'}`} />
        </div>
      )}

    </div>
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

function BoardGrid({ p1_pos, p2_pos, current_player, legal_moves, winner, onCellClick }) {
  const cells = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const isP1 = p1_pos.r === r && p1_pos.c === c;
      const isP2 = p2_pos.r === r && p2_pos.c === c;
      const isLegal = winner === 0 &&
        legal_moves.some(m => m[0] === 'move' && m[1] === r && m[2] === c);

      let goalClass = '';
      if (r === 8) goalClass = 'goal-p1';
      if (r === 0) goalClass = 'goal-p2';

      cells.push(
        <div
          key={`${r}-${c}`}
          className={`cell ${isLegal ? 'legal-move' : ''} ${goalClass}`}
          onClick={() => onCellClick(r, c)}
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

function MoveHistory({ moves }) {
  const pairs = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({
      num: Math.floor(i / 2) + 1,
      red: moves[i]?.notation || '',
      blue: moves[i + 1]?.notation || '',
    });
  }

  return (
    <div className="move-history">
      <div className="move-history-title">Move History</div>
      <div className="move-history-list">
        {pairs.map(p => (
          <div key={p.num} className="move-row">
            <span className="move-num">{p.num}.</span>
            <span className="move-red">{p.red}</span>
            <span className="move-blue">{p.blue}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function WallPreviewEl({ orient, r, c }) {
  const pct = 100 / BOARD_SIZE;

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
