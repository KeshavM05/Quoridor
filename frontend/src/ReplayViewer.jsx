import { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

const BOARD_SIZE = 9;

// Reconstruct game state at a given step from the move list
function reconstructState(moves, step) {
  const state = {
    p1_pos: { r: 0, c: 4 },
    p2_pos: { r: 8, c: 4 },
    p1_walls: 10,
    p2_walls: 10,
    h_walls: Array.from({ length: 8 }, () => Array(8).fill(false)),
    v_walls: Array.from({ length: 8 }, () => Array(8).fill(false)),
    current_player: 1,
    winner: 0,
  };

  for (let i = 0; i < step; i++) {
    const { player, move } = moves[i];
    if (move[0] === 'move') {
      const [, r, c] = move;
      if (player === 1) {
        state.p1_pos = { r, c };
      } else {
        state.p2_pos = { r, c };
      }
    } else if (move[0] === 'wall') {
      const [, orient, r, c] = move;
      if (orient === 'h') {
        state.h_walls[r][c] = true;
      } else {
        state.v_walls[r][c] = true;
      }
      if (player === 1) {
        state.p1_walls--;
      } else {
        state.p2_walls--;
      }
    }
    state.current_player = player === 1 ? 2 : 1;
  }

  return state;
}

export default function ReplayViewer({ replay, onBack }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const intervalRef = useRef(null);
  const moveListRef = useRef(null);

  const totalMoves = replay.moves.length;
  const gameState = reconstructState(replay.moves, step);

  // Check if game ended at this step
  const gameOver = step === totalMoves;

  // Auto-advance
  useEffect(() => {
    if (playing && step < totalMoves) {
      intervalRef.current = setTimeout(() => {
        setStep(s => s + 1);
      }, 1000 / speed);
    } else if (step >= totalMoves) {
      setPlaying(false);
    }
    return () => {
      if (intervalRef.current) clearTimeout(intervalRef.current);
    };
  }, [playing, step, speed, totalMoves]);

  // Scroll move list to keep current move visible
  useEffect(() => {
    if (moveListRef.current && step > 0) {
      const activeEl = moveListRef.current.querySelector('.move-row.active');
      if (activeEl) {
        activeEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }, [step]);

  const goToStart = () => { setStep(0); setPlaying(false); };
  const goBack = () => { setStep(s => Math.max(0, s - 1)); setPlaying(false); };
  const goForward = () => { setStep(s => Math.min(totalMoves, s + 1)); };
  const goToEnd = () => { setStep(totalMoves); setPlaying(false); };
  const togglePlay = () => {
    if (step >= totalMoves) {
      setStep(0);
      setPlaying(true);
    } else {
      setPlaying(p => !p);
    }
  };

  const lastMove = step > 0 ? replay.moves[step - 1] : null;

  return (
    <div className="game-screen">
      {/* Header */}
      <div className="game-header">
        <button className="back-btn" onClick={onBack}>&#8249; Back</button>
        <div className="replay-badge">REPLAY</div>
      </div>

      {/* Info bar */}
      <div className="replay-info-bar">
        <span className="replay-info-id">{replay.id}</span>
        <span className="replay-info-detail">
          Move {step}/{totalMoves}
          {lastMove && (
            <> &middot; <span className={`replay-notation p${lastMove.player}`}>{lastMove.notation}</span></>
          )}
        </span>
        {gameOver && (
          <span className={`replay-winner-tag p${replay.winner}`}>
            {replay.winner === 1 ? 'Red' : 'Blue'} wins
          </span>
        )}
      </div>

      {/* Player cards */}
      <div className="players">
        <div className={`pcard p1 ${gameState.current_player === 1 && !gameOver ? 'live' : 'dim'}`}>
          <div className="sheen" />
          <div className="pcard-fig"><div className="stone" /></div>
          <div className="pcard-info">
            <div className="pcard-name">Red</div>
            <div className="pcard-walls">{'━'} {gameState.p1_walls}/10</div>
          </div>
        </div>

        <div className="vs-pill">
          <div className="vs-dots"><i /><i /><i /></div>
          <span className="vs-text">VS</span>
          <div className="vs-dots"><i /><i /><i /></div>
        </div>

        <div className={`pcard p2 ${gameState.current_player === 2 && !gameOver ? 'live' : 'dim'}`}>
          <div className="sheen" />
          <div className="pcard-fig"><div className="stone" /></div>
          <div className="pcard-info" style={{ textAlign: 'right' }}>
            <div className="pcard-name">Blue</div>
            <div className="pcard-walls" style={{ justifyContent: 'flex-end' }}>{gameState.p2_walls}/10 {'━'}</div>
          </div>
        </div>
      </div>

      {/* Board */}
      <div className="board-container">
        <div className="board-frame">
          <div className="board-stud left" />
          <div className="board-stud right" />
          <div className="board-inner">
            <ReplayBoardGrid
              p1_pos={gameState.p1_pos}
              p2_pos={gameState.p2_pos}
              current_player={gameState.current_player}
              gameOver={gameOver}
            />
            <ReplayWalls h_walls={gameState.h_walls} v_walls={gameState.v_walls} />
            {gameOver && (
              <div className={`winner-banner p${replay.winner}`}>
                <h2>{replay.winner === 1 ? 'Red' : 'Blue'} Wins!</h2>
                <p className="watch-win-moves">{totalMoves} moves</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Playback Controls */}
      <div className="replay-controls">
        <div className="replay-btns">
          <button className="replay-ctrl-btn" onClick={goToStart} disabled={step === 0} title="Go to start">&#9198;</button>
          <button className="replay-ctrl-btn" onClick={goBack} disabled={step === 0} title="Back one">&#9664;</button>
          <button className="replay-ctrl-btn play-btn" onClick={togglePlay} title={playing ? 'Pause' : 'Play'}>
            {playing ? '⏸' : '▶'}
          </button>
          <button className="replay-ctrl-btn" onClick={goForward} disabled={step >= totalMoves} title="Forward one">&#9654;</button>
          <button className="replay-ctrl-btn" onClick={goToEnd} disabled={step >= totalMoves} title="Go to end">&#9197;</button>
        </div>
        <div className="replay-speed">
          <span className="replay-speed-label">{speed.toFixed(1)}x</span>
          <input
            type="range"
            className="replay-speed-slider"
            min="0.5"
            max="4"
            step="0.5"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
        </div>
      </div>

      {/* Move History */}
      <div className="replay-history" ref={moveListRef}>
        <div className="replay-history-title">Move History</div>
        <div className="replay-history-list">
          {buildMovePairs(replay.moves, step).map(p => (
            <div
              key={p.num}
              className={`move-row ${p.isActiveRed || p.isActiveBlue ? 'active' : ''}`}
              onClick={() => { setStep(p.isActiveBlue ? p.blueIdx + 1 : p.redIdx + 1); setPlaying(false); }}
            >
              <span className="move-num">{p.num}.</span>
              <span className={`move-red ${p.isActiveRed ? 'highlight' : ''}`}>{p.red}</span>
              <span className={`move-blue ${p.isActiveBlue ? 'highlight' : ''}`}>{p.blue}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function buildMovePairs(moves, step) {
  const pairs = [];
  for (let i = 0; i < moves.length; i += 2) {
    const redIdx = i;
    const blueIdx = i + 1;
    pairs.push({
      num: Math.floor(i / 2) + 1,
      red: moves[i]?.notation || '',
      blue: moves[i + 1]?.notation || '',
      redIdx,
      blueIdx,
      isActiveRed: step === redIdx + 1,
      isActiveBlue: blueIdx < moves.length && step === blueIdx + 1,
    });
  }
  return pairs;
}

function ReplayBoardGrid({ p1_pos, p2_pos, current_player, gameOver }) {
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
          {isP1 && <div className={`pawn p1 ${current_player === 1 && !gameOver ? 'active' : ''}`} />}
          {isP2 && <div className={`pawn p2 ${current_player === 2 && !gameOver ? 'active' : ''}`} />}
        </div>
      );
    }
  }

  return <div className="board-grid">{cells}</div>;
}

function ReplayWalls({ h_walls, v_walls }) {
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
