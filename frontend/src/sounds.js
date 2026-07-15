let ctx = null;

function getCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

function playTone(freq, duration, type = 'sine', volume = 0.15) {
  const c = getCtx();
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  gain.gain.value = volume;
  gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration);
  osc.connect(gain);
  gain.connect(c.destination);
  osc.start();
  osc.stop(c.currentTime + duration);
}

export function playMove() {
  playTone(600, 0.08, 'sine', 0.12);
  setTimeout(() => playTone(800, 0.06, 'sine', 0.08), 40);
}

export function playWall() {
  playTone(200, 0.15, 'square', 0.08);
  setTimeout(() => playTone(150, 0.12, 'square', 0.06), 60);
}

export function playWin() {
  const notes = [523, 659, 784, 1047];
  notes.forEach((f, i) => {
    setTimeout(() => playTone(f, 0.2, 'sine', 0.12), i * 100);
  });
}

export function playStart() {
  playTone(440, 0.1, 'sine', 0.1);
  setTimeout(() => playTone(660, 0.15, 'sine', 0.12), 80);
}

export function playError() {
  playTone(200, 0.15, 'sawtooth', 0.06);
}
