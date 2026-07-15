import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const WS_URL = 'ws://localhost:8001/ws/watch';
const API_URL = 'http://localhost:8001';
const BOARD_SIZE = 9;

export default function WatchGame({ onBack }) {
  const [gameState, setGameState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [moveDelay, setMoveDelay] = useState(1.0);
  const [status, setStatus] = useState('connecting');
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'game_state') {
          setGameState(data);
          setStatus('playing');
        } else if (data.type === 'game_over') {
          setStatus('game_over');
        } else if (data.type === 'waiting') {
          setStatus('waiting');
        }
      } catch (e) {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setStatus('disconnected');
      // Auto-reconnect
      reconnectRef.current = setTimeout(() => connect(), 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [connect]);

  // Start a game via REST
  const startGame = async () => {
    try {
      await fetch(`${API_URL}/watch/start`, { method: 'POST' });
      setStatus('playing');
      setIsPaused(false);
    } catch {
      setStatus('error');
    }
  };

  // Toggle pause
  const togglePause = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const action = isPaused ? 'resume' : 'pause';
      wsRef.current.send(JSON.stringify({ action }));
      setIsPaused(!isPaused);
    }
  };

  // Change speed
  const changeSpeed = (newDelay) => {
    const clamped = Math.max(0.1, Math.min(5.0, newDelay));
    setMoveDelay(clamped);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'speed', delay: clamped }));
    }
    // Also update via REST for robustness
    fetch(`${API_URL}/watch/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay: clamped })
    }).catch(() => {});
  };

  return (
    <div className="game-screen">
      {/* Header */}
      <div className="game-header">
        <button className="back-btn" onClick={onBack}>&#8249; Menu</button>
        <div className="watch-badge">LIVE AI</div>
      </div>

      {/* Status bar */}
      <div className="watch-status-bar">
        <span className={`watch-dot ${connected ? 'live' : ''}`} />
        <span className="watch-status-text">
          {status === 'connecting' && 'Connecting...'}
          {status === 'connected' && 'Connected - waiting for game'}
          {status === 'waiting' && 'Ready - press Start'}
          {status === 'playing' && `Move ${gameState?.move_count || 0}`}
          {status === 'game_over' && `Game over - ${gameState?.winner === 1 ? 'Red' : 'Blue'} wins!`}
          {status === 'disconnected' && 'Disconnected - reconnecting...'}
          {status === 'error' && 'Error connecting to server'}
        </span>
        {gameState && (
          <span className="watch-game-num">Game #{gameState.games_played}</span>
        )}
      </div>

      {/* Player cards */}
      {gameState && (
        <WatchPlayerCards
          current_player={gameState.current_player}
          p1_walls={gameState.p1_walls}
          p2_walls={gameState.p2_walls}
          winner={gameState.winner}
        />
      )}

      {/* Board */}
      <div className="board-container">
        <div className="board-frame">
          <div className="board-stud left" />
          <div className="board-stud right" />
          <div className="board-inner">
            {gameState ? (
              <>
                <WatchBoardGrid
                  p1_pos={gameState.p1_pos}
                  p2_pos={gameState.p2_pos}
                  current_player={gameState.current_player}
                  winner={gameState.winner}
                />
                <WatchWalls h_walls={gameState.h_walls} v_walls={gameState.v_walls} />
                {gameState.winner !== 0 && (
                  <div className={`winner-banner p${gameState.winner}`}>
                    <h2>{gameState.winner === 1 ? 'Red' : 'Blue'} Wins!</h2>
                    <p className="watch-win-moves">{gameState.move_count} moves</p>
                  </div>
                )}
              </>
            ) : (
              <div className="watch-empty-board">
                <p>Waiting for game...</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="watch-controls">
        <button className="watch-ctrl-btn" onClick={startGame}>
          New Game
        </button>
        <button className="watch-ctrl-btn" onClick={togglePause} disabled={!gameState || gameState.winner !== 0}>
          {isPaused ? 'Resume' : 'Pause'}
        </button>
        <div className="watch-speed-control">
          <button className="watch-speed-btn" onClick={() => changeSpeed(moveDelay + 0.5)}>Slower</button>
          <span className="watch-speed-label">{moveDelay.toFixed(1)}s</span>
          <button className="watch-speed-btn" onClick={() => changeSpeed(moveDelay - 0.5)}>Faster</button>
        </div>
      </div>
    </div>
  );
}


// --- Sub-components (reuse same CSS classes as App.jsx) ---

function WatchPlayerCards({ current_player, p1_walls, p2_walls, winner }) {
  const p1Live = current_player === 1 && winner === 0;
  const p2Live = current_player === 2 && winner === 0;

  return (
    <div className="players">
      <div className={`pcard p1 ${p1Live ? 'live' : 'dim'}`}>
        <div className="sheen" />
        <div className="pcard-fig"><div className="stone" /></div>
        <div className="pcard-info">
          <div className="pcard-name">Red AI</div>
          <div className="pcard-walls">{'━'} {p1_walls}/10</div>
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
          <div className="pcard-name">Blue AI</div>
          <div className="pcard-walls" style={{ justifyContent: 'flex-end' }}>{p2_walls}/10 {'━'}</div>
        </div>
      </div>
    </div>
  );
}


function WatchBoardGrid({ p1_pos, p2_pos, current_player, winner }) {
  const cells = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const isP1 = p1_pos.r === r && p1_pos.c === c;
      const isP2 = p2_pos.r === r && p2_pos.c === c;

      let goalClass = '';
      if (r === 8) goalClass = 'goal-p1';
      if (r === 0) goalClass = 'goal-p2';

      cells.push(
        <div
          key={`${r}-${c}`}
          className={`cell ${goalClass}`}
        >
          {isP1 && <div className={`pawn p1 ${current_player === 1 && winner === 0 ? 'active' : ''}`} />}
          {isP2 && <div className={`pawn p2 ${current_player === 2 && winner === 0 ? 'active' : ''}`} />}
        </div>
      );
    }
  }

  return <div className="board-grid">{cells}</div>;
}


function WatchWalls({ h_walls, v_walls }) {
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
