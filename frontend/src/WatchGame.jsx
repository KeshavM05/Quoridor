import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const WS_URL = 'ws://localhost:8001/ws/watch';
const API_URL = 'http://localhost:8001';
const BOARD_SIZE = 9;

export default function WatchGame({ onBack }) {
  const [gameState, setGameState] = useState(null);
  const [connected, setConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [moveDelay, setMoveDelay] = useState(0.3);
  const [status, setStatus] = useState('connecting');
  const [models, setModels] = useState([]);
  const [redModel, setRedModel] = useState('best');
  const [blueModel, setBlueModel] = useState('best');
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  // Fetch available models
  useEffect(() => {
    fetch(`${API_URL}/watch/models`).then(r => r.json()).then(setModels).catch(() => {});
  }, []);

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
          setGameState(prev => prev ? { ...prev, winner: data.winner, move_count: data.move_count } : prev);
          setStatus('game_over');
        } else if (data.type === 'waiting') {
          setStatus('waiting');
        }
      } catch (e) {}
    };

    ws.onclose = () => {
      setConnected(false);
      setStatus('disconnected');
      reconnectRef.current = setTimeout(() => connect(), 2000);
    };

    ws.onerror = () => { ws.close(); };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
    };
  }, [connect]);

  const startGame = async () => {
    try {
      setGameState(null);
      await fetch(`${API_URL}/watch/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ red_model: redModel, blue_model: blueModel })
      });
      setStatus('playing');
      setIsPaused(false);
    } catch { setStatus('error'); }
  };

  const togglePause = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: isPaused ? 'resume' : 'pause' }));
      setIsPaused(!isPaused);
    }
  };

  const handleSpeedChange = (e) => {
    const val = parseFloat(e.target.value);
    setMoveDelay(val);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'speed', delay: val }));
    }
    fetch(`${API_URL}/watch/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delay: val })
    }).catch(() => {});
  };

  const modelLabel = (id) => {
    if (id === 'best') return 'Best Model';
    if (id === 'random') return 'Random (Untrained)';
    if (id.startsWith('iter_')) return `Iteration ${id.replace('iter_', '')}`;
    return id;
  };

  return (
    <div className="game-screen">
      {/* Header */}
      <div className="game-header">
        <button className="back-btn" onClick={onBack}>&#8249; Menu</button>
        <div className="watch-badge">LIVE AI</div>
      </div>

      {/* Model Selector */}
      <div className="watch-model-select">
        <div className="watch-model-side red">
          <span className="watch-model-label">Red</span>
          <select value={redModel} onChange={e => setRedModel(e.target.value)}>
            <option value="best">Best Model</option>
            <option value="random">Random (Untrained)</option>
            {models.map(m => <option key={m} value={m}>{modelLabel(m)}</option>)}
          </select>
        </div>
        <span className="watch-vs">VS</span>
        <div className="watch-model-side blue">
          <span className="watch-model-label">Blue</span>
          <select value={blueModel} onChange={e => setBlueModel(e.target.value)}>
            <option value="best">Best Model</option>
            <option value="random">Random (Untrained)</option>
            {models.map(m => <option key={m} value={m}>{modelLabel(m)}</option>)}
          </select>
        </div>
      </div>

      {/* Status bar */}
      <div className="watch-status-bar">
        <span className={`watch-dot ${connected ? 'live' : ''}`} />
        <span className="watch-status-text">
          {status === 'connecting' && 'Connecting...'}
          {status === 'connected' && 'Connected — press Start'}
          {status === 'waiting' && 'Ready — press Start'}
          {status === 'playing' && `Move ${gameState?.move_count || 0}`}
          {status === 'game_over' && `Game over — ${gameState?.winner === 1 ? 'Red' : 'Blue'} wins in ${gameState?.move_count} moves!`}
          {status === 'disconnected' && 'Disconnected — reconnecting...'}
          {status === 'error' && 'Error connecting'}
        </span>
        {gameState && <span className="watch-game-num">Game #{gameState.games_played}</span>}
      </div>

      {/* Player cards with model names */}
      {gameState && (
        <div className="players">
          <div className={`pcard p1 ${gameState.current_player === 1 && gameState.winner === 0 ? 'live' : 'dim'}`}>
            <div className="sheen" />
            <div className="pcard-fig"><div className="stone" /></div>
            <div className="pcard-info">
              <div className="pcard-name">{modelLabel(redModel)}</div>
              <div className="pcard-walls">━ {gameState.p1_walls}/10</div>
            </div>
          </div>

          <div className="vs-pill">
            <div className="vs-dots"><i /><i /><i /></div>
            <span className="vs-text">VS</span>
            <div className="vs-dots"><i /><i /><i /></div>
          </div>

          <div className={`pcard p2 ${gameState.current_player === 2 && gameState.winner === 0 ? 'live' : 'dim'}`}>
            <div className="sheen" />
            <div className="pcard-fig"><div className="stone" /></div>
            <div className="pcard-info" style={{ textAlign: 'right' }}>
              <div className="pcard-name">{modelLabel(blueModel)}</div>
              <div className="pcard-walls" style={{ justifyContent: 'flex-end' }}>{gameState.p2_walls}/10 ━</div>
            </div>
          </div>
        </div>
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
                    <p style={{ color: 'var(--muted)', fontSize: 13 }}>{gameState.move_count} moves</p>
                  </div>
                )}
              </>
            ) : (
              <div className="watch-empty-board"><p>Press Start to begin</p></div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="watch-controls">
        <button className="watch-ctrl-btn primary" onClick={startGame}>Start</button>
        <button className="watch-ctrl-btn" onClick={togglePause} disabled={!gameState || gameState.winner !== 0}>
          {isPaused ? '▶ Resume' : '⏸ Pause'}
        </button>
      </div>

      {/* Speed slider */}
      <div className="watch-speed-row">
        <span className="watch-speed-label">⚡ {moveDelay < 0.05 ? 'Instant' : moveDelay < 0.2 ? 'Fast' : moveDelay < 0.8 ? 'Normal' : 'Slow'}</span>
        <input
          type="range"
          className="watch-speed-slider"
          min="0.01"
          max="2"
          step="0.01"
          value={moveDelay}
          onChange={handleSpeedChange}
        />
        <span className="watch-speed-val">{moveDelay.toFixed(2)}s</span>
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
        <div key={`${r}-${c}`} className={`cell ${goalClass}`}>
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
          <div key={`hw-${r}-${c}`} className={`wall wall-h ${r < 4 ? 'p2' : 'p1'}`} style={{
            top: `calc(${(r + 1) * pct}% - 3px)`,
            left: `calc(${c * pct}% + 2px)`,
            width: `calc(${2 * pct}% - 0px)`,
          }} />
        );
      }
      if (v_walls[r][c]) {
        walls.push(
          <div key={`vw-${r}-${c}`} className={`wall wall-v ${r < 4 ? 'p2' : 'p1'}`} style={{
            top: `calc(${r * pct}% + 2px)`,
            left: `calc(${(c + 1) * pct}% - 3px)`,
            height: `calc(${2 * pct}% - 0px)`,
          }} />
        );
      }
    }
  }
  return <>{walls}</>;
}
