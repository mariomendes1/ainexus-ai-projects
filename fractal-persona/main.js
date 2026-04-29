const fractalCanvas  = document.getElementById('fractal-canvas');
const videoEl        = document.getElementById('video');
const landmarkCanvas = document.getElementById('landmark-canvas');
const welcomeEl      = document.getElementById('welcome');
const wlcBar         = document.getElementById('wlc-bar');
const flashEl        = document.getElementById('flash');
const btnShot        = document.getElementById('btn-shot');
const btnMic         = document.getElementById('btn-mic');
const btnCam         = document.getElementById('btn-cam');
const cameraPanel    = document.getElementById('camera-panel');
const cameraList     = document.getElementById('camera-list');

FractalEngine.start(fractalCanvas);

// ── Welcome screen: auto-dismiss after 6s or on first hand ──
const WELCOME_MS = 6000;
let welcomeDone = false;

function dismissWelcome() {
  if (welcomeDone) return;
  welcomeDone = true;
  welcomeEl.classList.add('hidden');
}

wlcBar.style.transitionDuration = WELCOME_MS + 'ms';
requestAnimationFrame(() => { wlcBar.style.width = '100%'; });
setTimeout(dismissWelcome, WELCOME_MS);

// ── Screenshot ──
btnShot.addEventListener('click', () => {
  FractalEngine.screenshot();
  flashEl.style.opacity = '0.7';
  setTimeout(() => { flashEl.style.opacity = '0'; }, 80);
});

// ── Audio Engine ──
const AudioEngine = (() => {
  let ctx, analyser, stream;
  let smoothAmp = 0;
  let beatCooldown = 0;
  let _onBeat = null;
  let active = false;

  function tick() {
    if (!active) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    // focus on bass + mids (lower 65% of bins)
    const end = Math.floor(data.length * 0.65);
    let sum = 0;
    for (let i = 0; i < end; i++) sum += (data[i] / 255) ** 2;
    const rms = Math.sqrt(sum / end);
    const prev = smoothAmp;
    smoothAmp = smoothAmp * 0.87 + rms * 0.13;

    // beat: sudden amplitude spike
    if (beatCooldown > 0) beatCooldown--;
    if (rms > 0.2 && rms > prev * 1.55 && beatCooldown === 0) {
      beatCooldown = 18;
      if (_onBeat) _onBeat();
    }

    // mic button glow reflects volume
    const g = Math.min(smoothAmp * 5, 1);
    btnMic.style.boxShadow = `0 0 ${g * 22}px rgba(0,220,255,${(g * 0.7).toFixed(2)})`;

    requestAnimationFrame(tick);
  }

  async function start(onBeat) {
    _onBeat = onBeat;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const source = ctx.createMediaStreamSource(stream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    active = true;
    tick();
  }

  function stop() {
    active = false;
    smoothAmp = 0;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ctx) ctx.close();
    btnMic.style.boxShadow = '';
  }

  return {
    start, stop,
    get amplitude() { return smoothAmp; },
    get isActive()  { return active; }
  };
})();

// ── Mic toggle ──
let audioHueOffset = 0;

btnMic.addEventListener('click', async () => {
  if (AudioEngine.isActive) {
    AudioEngine.stop();
    btnMic.classList.remove('active');
    btnMic.title = 'Ativar microfone';
    btnMic.textContent = '🎙️';
  } else {
    try {
      btnMic.textContent = '⏳';
      await AudioEngine.start(() => {
        audioHueOffset += 0.08; // beat → hue jump
      });
      btnMic.classList.add('active');
      btnMic.title = 'Desativar microfone';
      btnMic.textContent = '🔊';
    } catch (e) {
      btnMic.textContent = '🎙️';
      alert('Microfone bloqueado: ' + e.message);
    }
  }
});

// ── Camera panel ──
let activeCamDeviceId = null;
let allCams = [];

async function loadCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  allCams = devices.filter(d => d.kind === 'videoinput');
}

async function populateCameras() {
  await loadCameras();
  cameraList.innerHTML = '';
  if (allCams.length === 0) {
    cameraList.innerHTML = '<p style="color:rgba(255,255,255,0.4);font-size:0.8rem;padding:4px 12px">Nenhuma câmara encontrada</p>';
    return;
  }
  allCams.forEach((cam, i) => {
    const btn = document.createElement('button');
    btn.className = 'cam-option';
    btn.textContent = cam.label || `Câmara ${i + 1}`;
    if (cam.deviceId === activeCamDeviceId || (i === 0 && !activeCamDeviceId)) {
      btn.classList.add('active');
    }
    btn.addEventListener('click', async () => {
      document.querySelectorAll('.cam-option').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeCamDeviceId = cam.deviceId;
      cameraPanel.classList.remove('open');
      syncWelcomeSelect();
      await HandTracker.switchCamera(cam.deviceId);
    });
    cameraList.appendChild(btn);
  });
}

function syncWelcomeSelect() {
  const sel = document.getElementById('wlc-cam-select');
  if (sel && activeCamDeviceId) sel.value = activeCamDeviceId;
}

async function populateWelcomeCameras() {
  await loadCameras();
  if (allCams.length <= 1) return; // só mostra se houver escolha
  const wrap = document.getElementById('wlc-cam-wrap');
  const sel  = document.getElementById('wlc-cam-select');
  sel.innerHTML = '';
  allCams.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Câmara ${i + 1}`;
    sel.appendChild(opt);
  });
  wrap.classList.add('visible');
  sel.addEventListener('change', async () => {
    activeCamDeviceId = sel.value;
    await HandTracker.switchCamera(sel.value);
  });
}

btnCam.addEventListener('click', async (e) => {
  e.stopPropagation();
  if (cameraPanel.classList.contains('open')) {
    cameraPanel.classList.remove('open');
  } else {
    await populateCameras();
    cameraPanel.classList.add('open');
  }
});

document.addEventListener('click', (e) => {
  if (!cameraPanel.contains(e.target) && e.target !== btnCam) {
    cameraPanel.classList.remove('open');
  }
});

// ── Idle: Newton fractal ──
const newtonPaths = [
  { cx:  0.00, cy:  0.00 },
  { cx:  0.28, cy:  0.18 },
  { cx: -0.22, cy:  0.30 },
  { cx:  0.12, cy: -0.25 },
];
let npIdx = 0, npTimer = 0, lastNow = performance.now();
let idleActive = true;

function idleStep() {
  if (!idleActive) return;
  const now = performance.now();
  npTimer += now - lastNow;
  lastNow  = now;
  if (npTimer > 8000) { npTimer = 0; npIdx = (npIdx + 1) % newtonPaths.length; }
  const t = now / 1000;
  const p = newtonPaths[npIdx];
  const audioBoost = AudioEngine.isActive ? AudioEngine.amplitude * 2.0 : 0;
  FractalEngine.update({
    cx:       p.cx + Math.sin(t * 0.09) * 0.10,
    cy:       p.cy + Math.cos(t * 0.07) * 0.10,
    zoom:     1.0  + Math.sin(t * 0.11) * 0.35 + audioBoost,
    rotation: t    * 0.018,
    hue:      t    * 0.010 + audioHueOffset,
    mode:     0.0,
  });
  requestAnimationFrame(idleStep);
}
idleStep();

const T0 = performance.now();

// ── Hand: Julia mode ──
HandTracker.init(videoEl, landmarkCanvas, (params) => {
  dismissWelcome(); // primeira mão detectada fecha o welcome
  idleActive = false;

  const audioBoost = AudioEngine.isActive ? AudioEngine.amplitude * 1.5 : 0;
  FractalEngine.update({
    cx:       params.cx,
    cy:       params.cy,
    zoom:     params.zoom + audioBoost,
    rotation: params.rotation,
    hue:      (performance.now() - T0) / 1000 * 0.010 + audioHueOffset,
    mode:     1.0,
  });

  clearTimeout(window._idleTO);
  window._idleTO = setTimeout(() => {
    idleActive = true;
    lastNow = performance.now();
    idleStep();
  }, 2500);
}, populateWelcomeCameras); // mostra seletor de câmara no welcome assim que a permissão é concedida
