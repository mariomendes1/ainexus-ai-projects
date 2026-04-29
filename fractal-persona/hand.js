const HandTracker = (() => {
  let hands;
  let lmCanvas, lmCtx, vidEl;
  let onHandData = null;
  let running = false;
  let currentStream = null;

  const statusEl = () => document.getElementById('status');

  function dist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  function fingerUp(tip, pip) {
    return tip.y < pip.y - 0.03;
  }

  function onResults(results) {
    lmCtx.clearRect(0, 0, lmCanvas.width, lmCanvas.height);

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      statusEl().textContent = 'mostra a mão';
      return;
    }

    const lm = results.multiHandLandmarks[0];

    // skeleton overlay
    const conn = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17]
    ];
    lmCtx.strokeStyle = 'rgba(0,210,255,0.65)';
    lmCtx.lineWidth = 1.2;
    conn.forEach(([a, b]) => {
      lmCtx.beginPath();
      lmCtx.moveTo(lm[a].x * lmCanvas.width, lm[a].y * lmCanvas.height);
      lmCtx.lineTo(lm[b].x * lmCanvas.width, lm[b].y * lmCanvas.height);
      lmCtx.stroke();
    });
    lmCtx.fillStyle = 'rgba(0,255,190,0.85)';
    lm.forEach(p => {
      lmCtx.beginPath();
      lmCtx.arc(p.x * lmCanvas.width, p.y * lmCanvas.height, 2, 0, Math.PI * 2);
      lmCtx.fill();
    });

    // palm centre → cx / cy  (mirror so moving right → cx positive)
    const palmX = (lm[0].x - 0.5) * -2;
    const palmY = (lm[0].y - 0.5) *  2;

    // pinch → zoom
    const pinch = dist(lm[4], lm[8]);
    const zoom  = 0.5 + (1.0 - Math.min(pinch / 0.3, 1.0)) * 3.5;

    // tilt → rotation
    const tilt = Math.atan2(lm[9].y - lm[0].y, lm[9].x - lm[0].x);

    // fingers up → palette
    const extended = [
      fingerUp(lm[8],  lm[6]),
      fingerUp(lm[12], lm[10]),
      fingerUp(lm[16], lm[14]),
      fingerUp(lm[20], lm[18]),
    ].filter(Boolean).length;

    statusEl().textContent = `${extended} dedos · pinch ${(pinch * 100).toFixed(0)}`;

    if (onHandData) onHandData({
      cx:       palmX * 0.8,
      cy:       palmY * 0.8,
      zoom,
      rotation: tilt,
      palette:  extended >= 3 ? 1.0 : 0.0,
    });
  }

  async function startCamera(deviceId) {
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
    const constraints = { video: { width: 640, height: 480 } };
    if (deviceId) constraints.video.deviceId = { exact: deviceId };
    else constraints.video.facingMode = 'user';
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    vidEl.srcObject = stream;
    await vidEl.play();
  }

  async function init(videoEl, landmarkCanvas, callback, onCameraReady) {
    lmCanvas       = landmarkCanvas;
    lmCtx          = lmCanvas.getContext('2d');
    lmCanvas.width  = 200;
    lmCanvas.height = 150;
    onHandData      = callback;
    vidEl           = videoEl;

    statusEl().textContent = 'a pedir câmara...';

    try {
      await startCamera(null);
      if (onCameraReady) onCameraReady(); // permissão concedida, labels disponíveis
    } catch (e) {
      statusEl().textContent = 'câmara bloqueada: ' + e.message;
      return;
    }

    statusEl().textContent = 'a carregar modelo...';

    hands = new Hands({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    });
    hands.setOptions({
      maxNumHands:            1,
      modelComplexity:        1,
      minDetectionConfidence: 0.65,
      minTrackingConfidence:  0.55,
    });
    hands.onResults(onResults);

    // send frames manually
    running = true;
    async function loop() {
      if (!running) return;
      if (vidEl.readyState >= 2) await hands.send({ image: vidEl });
      requestAnimationFrame(loop);
    }

    statusEl().textContent = 'mostra a mão';
    loop();
  }

  async function switchCamera(deviceId) {
    statusEl().textContent = 'a trocar câmara...';
    try {
      await startCamera(deviceId);
      statusEl().textContent = 'mostra a mão';
    } catch (e) {
      statusEl().textContent = 'erro: ' + e.message;
    }
  }

  return { init, switchCamera };
})();
