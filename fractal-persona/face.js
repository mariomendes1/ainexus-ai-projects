// MediaPipe FaceMesh → fractal parameter mapper
const FaceTracker = (() => {
  let faceMesh, camera;
  let lmCanvas, lmCtx;
  let videoEl;
  let onFaceData = null;

  // landmark indices (MediaPipe 468-point mesh)
  const IDX = {
    noseTip: 1,
    chin: 152,
    leftEye: 33,
    rightEye: 263,
    mouthTop: 13,
    mouthBottom: 14,
    leftEar: 234,
    rightEar: 454,
  };

  function dist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  function onResults(results) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;

    const lm = results.multiFaceLandmarks[0];

    // draw landmarks overlay on small preview
    if (lmCtx && lmCanvas) {
      lmCtx.clearRect(0, 0, lmCanvas.width, lmCanvas.height);
      lmCtx.fillStyle = 'rgba(0,255,180,0.6)';
      for (let i = 0; i < lm.length; i += 4) {
        lmCtx.beginPath();
        lmCtx.arc(lm[i].x * lmCanvas.width, lm[i].y * lmCanvas.height, 1, 0, Math.PI * 2);
        lmCtx.fill();
      }
    }

    // --- extract face metrics ---

    // Head position X/Y (nose tip, normalised 0-1, centred around 0.5)
    const nose = lm[IDX.noseTip];
    const headX = (nose.x - 0.5) * 2;  // -1 to 1
    const headY = (nose.y - 0.5) * 2;  // -1 to 1

    // Head tilt: angle between left eye and right eye
    const le = lm[IDX.leftEye];
    const re = lm[IDX.rightEye];
    const tilt = Math.atan2(re.y - le.y, re.x - le.x); // radians

    // Mouth openness (ratio of mouth gap to eye distance)
    const eyeDist = dist(le, re);
    const mouthGap = dist(lm[IDX.mouthTop], lm[IDX.mouthBottom]);
    const mouthRatio = eyeDist > 0 ? mouthGap / eyeDist : 0;

    // Face scale (proximity) — ratio of face width to canvas
    const faceWidth = dist(lm[IDX.leftEar], lm[IDX.rightEar]);

    // --- map to fractal params ---
    const fractalParams = {
      // Julia cx: head horizontal position → range -0.9 to 0.9
      cx: headX * 0.4,

      // Julia cy: head vertical position → range -0.9 to 0.9
      cy: headY * 0.4,

      // zoom: mouth openness → 0.6 to 3.0
      zoom: 0.8 + mouthRatio * 8,

      // rotation: head tilt → direct radians
      rotation: tilt * 2,

      // hue shift: face scale (proximity) → 0-360 degrees
      hueShift: faceWidth * 1800,
    };

    if (onFaceData) onFaceData(fractalParams, { headX, headY, tilt, mouthRatio, faceWidth });
  }

  function init(videoElement, landmarkCanvas, callback) {
    videoEl = videoElement;
    lmCanvas = landmarkCanvas;
    lmCtx = lmCanvas.getContext('2d');
    lmCanvas.width = 180;
    lmCanvas.height = 135;
    onFaceData = callback;

    faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(onResults);

    camera = new Camera(videoEl, {
      onFrame: async () => { await faceMesh.send({ image: videoEl }); },
      width: 640,
      height: 480,
    });

    camera.start();
  }

  return { init };
})();
