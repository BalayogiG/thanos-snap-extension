import { FilesetResolver, HandLandmarker, ImageSegmenter } from './lib/vision_bundle.mjs';
import * as ort from './lib/ort/ort.wasm.min.mjs';

const $ = (id) => document.getElementById(id);
const video = $('video');
const stage = $('stage');
const baseCanvas = $('base');
const fxCanvas = $('fx');
const bctx = baseCanvas.getContext('2d');

// Tunables
const SWEEP = 2.4;          // seconds for the dust front to cross the body
const MAX_PARTICLES = 45000;
const PINCH = 0.4;          // thumb-middle distance / hand size that counts as pinched
const RELEASE = 0.7;        // ...and the distance that counts as released
const CURL = 1.5;           // middle tip to wrist / hand size when snapped into the palm
const SNAP_WINDOW = 250;    // ms between pinch and release for a snap
const COOLDOWN = 1500;      // ms between triggers
// Label MediaPipe gives the user's real right hand on our raw (unmirrored) frame. Flip to 'Left' if the wrong hand is tracked.
const RIGHT_HAND = 'Right';
const GRID_W = 48, GRID_H = 27;
const COLORS = ['#14161c', '#0f7a4d', '#27386b', '#d8d3c6'];
const PICTURES = ['sunset', 'space', 'forest', 'office', 'beach', 'city', 'snow', 'abstract'];

const S = {
  mode: 'boot',             // boot | live | charging | dusting | gone
  W: 0, H: 0,
  personNZ: null,           // is "person" the non-zero side of the category mask?
  voteVal: null, votes: 0,
  maskReady: false, personFrac: 0,
  wrist: null, handSeen: false,
  t0: 0, endT: 0,
  lastTrigger: 0,
  frame: 0,
  solid: true,              // flat-colour backdrop instead of the real room
  colorIdx: 0,
};

// ---- offscreen canvases ---------------------------------------------------
const mk = (opts) => Object.assign(document.createElement('canvas'), opts);
const maskCanvas = mk();   // raw person alpha, at mask resolution
const dilCanvas = mk();    // blurred/grown person alpha, at video resolution
const plateCanvas = mk();  // remembered background
const tmpCanvas = mk();
const cmpCanvas = mk();    // plate cut to the person's silhouette
const snapCanvas = mk();
const mlCanvas = mk();     // downscaled copy of the frame that the ML models see
const outCanvas = mk();    // composited frame that gets sent to Google Meet
const softCanvas = mk();   // lightly feathered person alpha, for the flat-colour backdrop
const fillCanvas = mk();   // plate with never-seen pixels filled from their surroundings
const mctx = maskCanvas.getContext('2d');
const dctx = dilCanvas.getContext('2d');
const pctx = plateCanvas.getContext('2d');
const tctx = tmpCanvas.getContext('2d');
const cctx = cmpCanvas.getContext('2d');
const sctx = snapCanvas.getContext('2d', { willReadFrequently: true });
const fctx = fillCanvas.getContext('2d');
const softCtx = softCanvas.getContext('2d');
const octx = outCanvas.getContext('2d');
const mlctx = mlCanvas.getContext('2d');
const OFFSCREEN = new URLSearchParams(location.search).get('mode') === 'offscreen';
const ML_WIDTH = 384;
const PW = 64, PH = 36;
const probe = [0, 1, 2].map(() => mk({ width: PW, height: PH }));
const probeCtx = probe.map((c) => c.getContext('2d', { willReadFrequently: true }));
let gain = 1;
let maskImg = null;

const covered = new Uint8Array(GRID_W * GRID_H);
let coveredCount = 0;

// ---- ML -------------------------------------------------------------------
let hands, segmenter;

async function loadModels() {
  const fileset = await FilesetResolver.forVisionTasks('lib/wasm');
  const make = async (delegate) => {
    hands = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: 'models/hand_landmarker.task', delegate },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: 'models/selfie_segmenter.tflite', delegate },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  };
  try { await make('GPU'); S.delegate = 'GPU'; }
  catch (e) { console.warn('GPU delegate failed, using CPU', e); S.delegate = 'CPU (GPU failed: ' + (e && e.message ? e.message : e) + ')'; await make('CPU'); }
}

// ---- camera / mic ---------------------------------------------------------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  await new Promise((r) => (video.videoWidth ? r() : video.addEventListener('loadedmetadata', r, { once: true })));
  const W = video.videoWidth, H = video.videoHeight;
  S.W = W; S.H = H;
  for (const c of [baseCanvas, fxCanvas, dilCanvas, plateCanvas, tmpCanvas, cmpCanvas, snapCanvas, fillCanvas, softCanvas, outCanvas]) { c.width = W; c.height = H; }
  mlCanvas.width = ML_WIDTH;
  mlCanvas.height = Math.round((ML_WIDTH * H) / W);
  stage.style.setProperty('--ar', W / H);
}

const audio = { ctx: null, an: null, buf: null, base: 0.01, on: false };

async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    audio.ctx = new AudioContext();
    const src = audio.ctx.createMediaStreamSource(stream);
    const hp = audio.ctx.createBiquadFilter();   // snaps are bright; voices mostly aren't
    hp.type = 'highpass';
    hp.frequency.value = 1800;
    audio.an = audio.ctx.createAnalyser();
    audio.an.fftSize = 1024;
    audio.buf = new Float32Array(1024);
    src.connect(hp);
    hp.connect(audio.an);
    audio.on = true;
  } catch (e) {
    console.warn('Mic unavailable, falling back to hand gesture only', e);
  }
}

function audioSpike() {
  if (!audio.on) return false;
  audio.an.getFloatTimeDomainData(audio.buf);
  let peak = 0;
  for (let i = 0; i < audio.buf.length; i++) peak = Math.max(peak, Math.abs(audio.buf[i]));
  const spike = peak > Math.max(0.12, audio.base * 5);
  if (!spike) audio.base = audio.base * 0.98 + peak * 0.02;
  return spike;
}

// ---- snap detection -------------------------------------------------------
const snapState = { pinchedFrames: 0, lastPinchT: -1e9, curled: false };
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function detectSnap(allHands, now, spike) {
  let d = Infinity, curl = Infinity;
  for (const lm of allHands) {
    const size = dist(lm[0], lm[9]) || 1e-3;
    d = Math.min(d, dist(lm[4], lm[12]) / size);
    curl = Math.min(curl, dist(lm[12], lm[0]) / size);
  }
  const st = snapState;
  let fired = false;

  if (d < PINCH) {
    st.pinchedFrames++;
    st.lastPinchT = now;
    st.curled = false;
  } else if (st.pinchedFrames >= 2 && now - st.lastPinchT <= SNAP_WINDOW) {
    if (curl < CURL) st.curled = true;
    if (d > RELEASE && st.curled) fired = true;
  } else if (now - st.lastPinchT > SNAP_WINDOW) {
    st.pinchedFrames = 0;
  }

  // A sharp high-frequency click shortly after a pinch is also a snap.
  if (spike && st.pinchedFrames >= 2 && now - st.lastPinchT <= 500) fired = true;

  if (fired) st.pinchedFrames = 0;
  return fired;
}

// ---- mask handling --------------------------------------------------------
function tryCalibrate(d, w, h) {
  let vote = null;
  if (S.wrist) {
    const x = Math.min(w - 1, Math.max(0, (S.wrist.x * w) | 0));
    const y = Math.min(h - 1, Math.max(0, (S.wrist.y * h) | 0));
    vote = d[y * w + x] > 0;
  } else {
    // No hand yet: assume the person is more central than the frame's border.
    let cN = 0, cT = 0, rN = 0, rT = 0;
    for (let y = 0; y < h; y += 8) {
      for (let x = 0; x < w; x += 8) {
        const nz = d[y * w + x] > 0;
        const fx = x / w, fy = y / h;
        if (fx > 0.3 && fx < 0.7 && fy > 0.25 && fy < 0.85) { cT++; if (nz) cN++; }
        else if (fx < 0.1 || fx > 0.9 || fy < 0.1) { rT++; if (nz) rN++; }
      }
    }
    const fc = cN / cT, fr = rN / rT;
    if (Math.abs(fc - fr) > 0.35) vote = fc > fr;
  }
  if (vote === null) return;
  if (vote === S.voteVal) S.votes++; else { S.voteVal = vote; S.votes = 1; }
  if (S.votes >= 8) S.personNZ = vote;
}

function onMask(result) {
  const m = result.categoryMask;
  if (!m) return;
  const w = m.width, h = m.height, d = m.getAsUint8Array();
  if (S.personNZ === null) tryCalibrate(d, w, h);
  if (S.personNZ === null) { S.maskReady = false; return; }

  if (!maskImg || maskImg.width !== w || maskImg.height !== h) {
    maskImg = new ImageData(w, h);
    maskCanvas.width = w; maskCanvas.height = h;
  }
  const nz = S.personNZ, out = maskImg.data;
  let cnt = 0;
  for (let i = 0, j = 3; i < d.length; i++, j += 4) {
    const p = (d[i] > 0) === nz;
    out[j] = p ? 255 : 0;
    if (p) cnt++;
  }
  mctx.putImageData(maskImg, 0, 0);
  S.personFrac = cnt / d.length;
  S.maskReady = true;

  if (S.solid) {
    softCtx.clearRect(0, 0, S.W, S.H);
    softCtx.filter = 'blur(1.5px)';
    softCtx.drawImage(maskCanvas, 0, 0, S.W, S.H);
    softCtx.filter = 'none';
  }

  // Grow the silhouette a little so edge pixels never leak into the plate.
  // Only the real-room mode reads this, so skip the blurs for the flat backdrop.
  if (!S.solid || S.mode !== 'live') {
    dctx.clearRect(0, 0, S.W, S.H);
    dctx.filter = 'blur(9px)';
    for (let i = 0; i < 5; i++) dctx.drawImage(maskCanvas, 0, 0, S.W, S.H);
    dctx.filter = 'none';
  }

  if (S.mode === 'live') {
    for (let gy = 0; gy < GRID_H; gy++) {
      for (let gx = 0; gx < GRID_W; gx++) {
        const idx = gy * GRID_W + gx;
        if (covered[idx]) continue;
        const mx = ((gx + 0.5) / GRID_W * w) | 0, my = ((gy + 0.5) / GRID_H * h) | 0;
        if (out[(my * w + mx) * 4 + 3] === 0) { covered[idx] = 1; coveredCount++; }
      }
    }
  }
}

function updatePlate() {
  tctx.globalCompositeOperation = 'source-over';
  tctx.clearRect(0, 0, S.W, S.H);
  tctx.drawImage(video, 0, 0, S.W, S.H);
  tctx.globalCompositeOperation = 'destination-out';
  tctx.drawImage(dilCanvas, 0, 0);
  tctx.globalCompositeOperation = 'source-over';
  pctx.drawImage(tmpCanvas, 0, 0);
}

// The plate is transparent wherever the camera never saw the background (e.g. behind
// someone who has not moved). Blend in blurred copies of what is known, coarse to fine.
function buildFilledPlate() {
  const { W, H } = S;
  const levels = [];
  let src = plateCanvas;
  for (let d = 2; d <= 64; d *= 2) {
    const c = mk({ width: Math.max(1, Math.ceil(W / d)), height: Math.max(1, Math.ceil(H / d)) });
    const x = c.getContext('2d');
    x.imageSmoothingQuality = 'high';
    x.drawImage(src, 0, 0, c.width, c.height);
    levels.push(c);
    src = c;
  }
  fctx.globalCompositeOperation = 'source-over';
  fctx.imageSmoothingQuality = 'high';
  fctx.clearRect(0, 0, W, H);
  for (let i = levels.length - 1; i >= 0; i--) fctx.drawImage(levels[i], 0, 0, W, H);
  fctx.drawImage(plateCanvas, 0, 0);
  fctx.globalCompositeOperation = 'destination-over';
  fctx.fillStyle = '#2a2a30';
  fctx.fillRect(0, 0, W, H);
  fctx.globalCompositeOperation = 'source-over';
}

// Webcams re-expose when the person leaves the shot, so compare the live background
// with the stored one (only where nobody stands) and scale the patch to match.
function matchExposure() {
  probeCtx[0].drawImage(video, 0, 0, PW, PH);
  probeCtx[1].drawImage(fillCanvas, 0, 0, PW, PH);
  probeCtx[2].clearRect(0, 0, PW, PH);
  probeCtx[2].drawImage(dilCanvas, 0, 0, PW, PH);
  const v = probeCtx[0].getImageData(0, 0, PW, PH).data;
  const f = probeCtx[1].getImageData(0, 0, PW, PH).data;
  const m = probeCtx[2].getImageData(0, 0, PW, PH).data;
  let sv = 0, sf = 0;
  for (let i = 0; i < m.length; i += 4) {
    if (m[i + 3] > 4) continue;
    sv += v[i] + v[i + 1] + v[i + 2];
    sf += f[i] + f[i + 1] + f[i + 2];
  }
  if (sf < 300) return;
  const target = Math.min(1.4, Math.max(0.7, sv / sf));
  gain += (target - gain) * 0.1;
}

// ---- backdrops: flat colours first, then pictures (bundled, or dropped onto the window)
const pictures = PICTURES.map((n) => loadPicture(`backgrounds/${n}.svg`));
function loadPicture(src) {
  const img = new Image();
  img.src = src;
  return img;
}
const BACKDROPS = () => COLORS.length + pictures.length;

function drawBackdrop() {
  const pic = pictures[S.colorIdx - COLORS.length];
  if (!pic) {
    bctx.fillStyle = COLORS[S.colorIdx];
    bctx.fillRect(0, 0, S.W, S.H);
    return;
  }
  if (!pic.naturalWidth) { bctx.fillStyle = COLORS[0]; bctx.fillRect(0, 0, S.W, S.H); return; }
  const k = Math.max(S.W / pic.naturalWidth, S.H / pic.naturalHeight);   // cover-fit
  const w = pic.naturalWidth * k, h = pic.naturalHeight * k;
  bctx.drawImage(pic, (S.W - w) / 2, (S.H - h) / 2, w, h);
}

// Flat-colour mode: the live person, cut out of the video.
function updatePerson() {
  cctx.globalCompositeOperation = 'source-over';
  cctx.clearRect(0, 0, S.W, S.H);
  cctx.drawImage(video, 0, 0, S.W, S.H);
  cctx.globalCompositeOperation = 'destination-in';
  cctx.drawImage(softCanvas, 0, 0);
  cctx.globalCompositeOperation = 'source-over';
}

function updateCutout() {
  matchExposure();
  cctx.globalCompositeOperation = 'source-over';
  cctx.clearRect(0, 0, S.W, S.H);
  cctx.filter = `brightness(${gain.toFixed(3)})`;
  cctx.drawImage(fillCanvas, 0, 0);
  cctx.filter = 'none';
  cctx.globalCompositeOperation = 'destination-in';
  cctx.drawImage(dilCanvas, 0, 0);
  cctx.globalCompositeOperation = 'source-over';
}

// ---- inpainting (MI-GAN via onnxruntime-web) -------------------------------
let inpainter = null;
let holeSnap = null;   // silhouette alpha captured at the moment of the snap

async function loadInpainter() {
  try {
    ort.env.wasm.wasmPaths = new URL('lib/ort/', location.href).href;
    ort.env.wasm.numThreads = 1;   // blob workers are blocked by the extension CSP
    inpainter = await ort.InferenceSession.create('models/migan.onnx', { executionProviders: ['wasm'] });
  } catch (e) {
    console.warn('Inpainting model failed to load, using blurred fill', e);
  }
}

// Removes the silhouette from the frozen frame (snapCanvas) and writes the result to fillCanvas.
async function inpaintFrame() {
  const { W, H } = S;
  const PW = Math.ceil(W / 64) * 64, PH = Math.ceil(H / 64) * 64, area = PW * PH;
  const rgba = sctx.getImageData(0, 0, W, H).data;
  const hole = holeSnap;
  const img = new Uint8Array(3 * area);
  const known = new Uint8Array(area).fill(255);   // MI-GAN: 255 = keep, 0 = fill in
  for (let y = 0; y < PH; y++) {
    const sy = Math.min(y, H - 1);
    for (let x = 0; x < PW; x++) {
      const si = (sy * W + Math.min(x, W - 1)) * 4, di = y * PW + x;
      if (y < H && x < W && hole[si + 3] > 60) { known[di] = 0; continue; }
      img[di] = rgba[si]; img[area + di] = rgba[si + 1]; img[2 * area + di] = rgba[si + 2];
    }
  }
  const out = await inpainter.run({
    image: new ort.Tensor('uint8', img, [1, 3, PH, PW]),
    mask: new ort.Tensor('uint8', known, [1, 1, PH, PW]),
  });
  const o = out[inpainter.outputNames[0]].data;
  const res = new ImageData(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const di = y * PW + x, j = (y * W + x) * 4;
      res.data[j] = o[di]; res.data[j + 1] = o[area + di]; res.data[j + 2] = o[2 * area + di]; res.data[j + 3] = 255;
    }
  }
  fctx.putImageData(res, 0, 0);
}

// ---- particles (WebGL) ----------------------------------------------------
const gl = fxCanvas.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: false });
let prog, buf, loc, particleCount = 0;

const VERT = `
attribute vec2 a_o;
attribute vec3 a_c;
attribute vec4 a_r;           // delay, life, rand1, rand2
uniform vec2 u_res;
uniform float u_t;
uniform float u_size;
varying vec4 v_col;
void main() {
  float age = u_t - a_r.x;
  float t = max(age, 0.0);
  float f = clamp(t / a_r.y, 0.0, 1.0);
  float r1 = a_r.z, r2 = a_r.w;
  vec2 pos = a_o;
  float k = t * (1.0 + 0.6 * t);
  // The canvas is mirrored by CSS, so "blown to the right" is -x here.
  pos += vec2(-1.0, -0.28) * (60.0 + 240.0 * r1) * k;
  pos += vec2(sin(t * 2.7 + r2 * 6.283), cos(t * 2.1 + r1 * 6.283)) * (6.0 + 34.0 * r2) * min(t, 1.5);
  pos.y -= 50.0 * r2 * t;
  float alpha = 1.0 - f * f;
  vec3 col = a_c;
  col += vec3(1.0, 0.55, 0.2) * exp(-t * 5.0) * step(0.0, age) * 0.6;
  col = mix(col, vec3(0.18, 0.16, 0.15), f * 0.7);
  v_col = vec4(col * alpha, alpha);
  gl_Position = vec4(pos.x / u_res.x * 2.0 - 1.0, 1.0 - pos.y / u_res.y * 2.0, 0.0, 1.0);
  gl_PointSize = u_size * (1.0 - 0.55 * f);
}`;
const FRAG = `
precision mediump float;
varying vec4 v_col;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = v_col;
}`;

function initGL() {
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  loc = {
    o: gl.getAttribLocation(prog, 'a_o'), c: gl.getAttribLocation(prog, 'a_c'), r: gl.getAttribLocation(prog, 'a_r'),
    res: gl.getUniformLocation(prog, 'u_res'), t: gl.getUniformLocation(prog, 'u_t'), size: gl.getUniformLocation(prog, 'u_size'),
  };
  buf = gl.createBuffer();
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
}

const hash2 = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };
let pointSize = 4;

function buildParticles() {
  const { W, H } = S;
  sctx.drawImage(video, 0, 0, W, H);
  const px = sctx.getImageData(0, 0, W, H).data;
  const md = maskImg.data, mw = maskImg.width, mh = maskImg.height;
  const isPerson = (x, y) => md[((((y / H) * mh) | 0) * mw + (((x / W) * mw) | 0)) * 4 + 3] > 0;

  const step = Math.min(6, Math.max(2, Math.ceil(Math.sqrt((S.personFrac * W * H) / MAX_PARTICLES))));
  pointSize = step * 1.3;

  // Pass 1: extent along the display x-axis (mirrored), so the front sweeps the body, not the frame.
  let minD = Infinity, maxD = -Infinity, n = 0;
  for (let y = 0; y < H; y += step) {
    for (let x = 0; x < W; x += step) {
      if (!isPerson(x, y)) continue;
      const dx = W - x;
      if (dx < minD) minD = dx;
      if (dx > maxD) maxD = dx;
      n++;
    }
  }
  if (n === 0) return false;
  const span = Math.max(1, maxD - minD);

  const data = new Float32Array(n * 9);
  let i = 0, maxEnd = 0;
  for (let y = 0; y < H; y += step) {
    for (let x = 0; x < W; x += step) {
      if (!isPerson(x, y)) continue;
      const dn = (W - x - minD) / span;
      const life = 1.6 + Math.random() * 1.4;
      const delay = dn * SWEEP + hash2(Math.floor(x / 28), Math.floor(y / 28)) * 0.8 + Math.random() * 0.25;
      maxEnd = Math.max(maxEnd, delay + life);
      const p = (y * W + x) * 4;
      data.set([
        x + (Math.random() - 0.5) * step, y + (Math.random() - 0.5) * step,
        px[p] / 255, px[p + 1] / 255, px[p + 2] / 255,
        delay, life, Math.random(), Math.random(),
      ], i);
      i += 9;
    }
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  particleCount = n;
  S.endT = maxEnd + 0.2;
  return true;
}

function drawParticles(t) {
  gl.viewport(0, 0, S.W, S.H);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!particleCount) return;
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  const stride = 36;
  gl.enableVertexAttribArray(loc.o); gl.vertexAttribPointer(loc.o, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(loc.c); gl.vertexAttribPointer(loc.c, 3, gl.FLOAT, false, stride, 8);
  gl.enableVertexAttribArray(loc.r); gl.vertexAttribPointer(loc.r, 4, gl.FLOAT, false, stride, 20);
  gl.uniform2f(loc.res, S.W, S.H);
  gl.uniform1f(loc.t, t);
  gl.uniform1f(loc.size, pointSize);
  gl.drawArrays(gl.POINTS, 0, particleCount);
}

function clearParticles() {
  particleCount = 0;
  gl.viewport(0, 0, S.W, S.H);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

// ---- state transitions ----------------------------------------------------
function trigger() {
  const now = performance.now();
  if (now - S.lastTrigger < COOLDOWN) return;
  if (S.mode === 'live') {
    if (!S.maskReady || S.personFrac < 0.01) { setHint('Nobody to snap away yet. Stand in view first.', 2500); return; }
    if (!buildParticles()) return;
    S.lastTrigger = now;
    if (S.solid) {
      S.mode = 'dusting';
      S.t0 = now;
      const f = $('flash');
      f.classList.remove('go'); void f.offsetWidth; f.classList.add('go');
      return;
    }
    holeSnap = dctx.getImageData(0, 0, S.W, S.H).data;
    buildFilledPlate();   // blurred fallback, replaced by the inpainted frame when ready
    gain = 1;
    S.mode = 'charging';
    startDust();
  } else if (S.mode === 'gone') {
    S.lastTrigger = now;
    reset();
  }
}

// The figure holds frozen while the background behind it is inpainted from the same
// frame (so lighting and noise match), then the dust is released.
async function startDust() {
  if (inpainter) {
    try {
      await new Promise((r) => setTimeout(r, 50));   // not rAF: hidden pages never get it
      await inpaintFrame();
    } catch (e) {
      console.warn('Inpainting failed, using blurred fill', e);
    }
  }
  if (S.mode !== 'charging') return;   // reset while waiting
  S.mode = 'dusting';
  S.t0 = performance.now();
  const f = $('flash');
  f.classList.remove('go'); void f.offsetWidth; f.classList.add('go');
}

function reset() {
  clearParticles();
  S.mode = 'live';
  setHint('');
}

// ---- main loop ------------------------------------------------------------
let lastVT = -1;

// Hand tracking is the most expensive step, so it runs at ~15 Hz while the mask and the
// output video keep the full frame rate. Mic spikes are sampled every frame regardless.
const HAND_INTERVAL = 66;
let lastHandT = -1e9, lastSpikeT = -1e9;

function process(now) {
  mlctx.drawImage(video, 0, 0, mlCanvas.width, mlCanvas.height);
  if (audioSpike()) lastSpikeT = now;
  if (now - lastHandT >= HAND_INTERVAL) {
    lastHandT = now;
    const th = performance.now();
    const hr = hands.detectForVideo(mlCanvas, now);
    perf.hand += performance.now() - th;
    const right = hr.landmarks.filter((_, i) => hr.handedness[i][0].categoryName === RIGHT_HAND);
    S.handSeen = right.length > 0;
    S.wrist = S.handSeen ? right[0][0] : null;
    if (detectSnap(right, now, now - lastSpikeT < 120)) trigger();
  }

  const ts = performance.now();
  segmenter.segmentForVideo(mlCanvas, now, onMask);
  perf.seg += performance.now() - ts;
  if (S.maskReady) {
    if (S.solid) {
      if (S.mode === 'live') updatePerson();
    } else {
      if (S.mode === 'live' && S.frame % 3 === 0) updatePlate();
      if (S.mode !== 'live') updateCutout();
    }
  }
  S.frame++;
}

function renderScene(now) {
  if (S.solid && S.maskReady) {
    drawBackdrop();
    if (S.mode === 'live') bctx.drawImage(cmpCanvas, 0, 0);
  } else {
    bctx.drawImage(video, 0, 0, S.W, S.H);
    if (S.mode === 'live') return;
    if (S.maskReady) bctx.drawImage(cmpCanvas, 0, 0);
  }
  if (S.mode === 'charging') drawParticles(-1);
  if (S.mode === 'dusting') {
    const t = (now - S.t0) / 1000;
    drawParticles(t);
    if (t > S.endT) {
      clearParticles();
      S.mode = 'gone';
      setHint('Snap again (or press R) to bring yourself back.');
    }
  }
}

function render(now) {
  renderScene(now);
  octx.drawImage(baseCanvas, 0, 0);
  octx.drawImage(fxCanvas, 0, 0);
}

let lastTick = 0;
const perf = { t0: performance.now(), n: 0, proc: 0, procN: 0, rend: 0, maxGap: 0, hand: 0, seg: 0 };
function tick() {
  if (S.mode === 'boot') return;
  const now = performance.now();
  if (now - lastTick < 20) return;   // rAF and the worker timer can both fire
  perf.maxGap = Math.max(perf.maxGap, now - lastTick);
  lastTick = now;
  if (video.readyState >= 2 && video.currentTime !== lastVT) {
    lastVT = video.currentTime;
    const a = performance.now();
    process(now);
    perf.proc += performance.now() - a;
    perf.procN++;
  }
  const b = performance.now();
  render(now);
  perf.rend += performance.now() - b;
  perf.n++;
  if (now - perf.t0 > 5000) {
    const sec = (now - perf.t0) / 1000;
    report(`perf[${S.delegate}]: ${(perf.n / sec).toFixed(1)} fps, hand ${(perf.hand / Math.max(1, perf.procN)).toFixed(0)}ms, seg ${(perf.seg / Math.max(1, perf.procN)).toFixed(0)}ms, process ${(perf.proc / Math.max(1, perf.procN)).toFixed(0)}ms, render ${(perf.rend / perf.n).toFixed(0)}ms, max gap ${perf.maxGap.toFixed(0)}ms, ${document.visibilityState}, sessions ${sessions.size}`);
    Object.assign(perf, { t0: now, n: 0, proc: 0, procN: 0, rend: 0, maxGap: 0, hand: 0, seg: 0 });
  }
}

function loop() {
  requestAnimationFrame(loop);
  tick();
}

// ---- Google Meet source ---------------------------------------------------
// Meet pages ask (via the background worker) for our canvas stream over local WebRTC.
const sessions = new Map();
let outStream = null;

function iceDone(pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const done = () => { pc.removeEventListener('icegatheringstatechange', check); resolve(); };
    const check = () => { if (pc.iceGatheringState === 'complete') done(); };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(done, 2000);
  });
}

function closeSession(key) {
  const pc = sessions.get(key);
  sessions.delete(key);
  if (pc) pc.close();
  // The hidden page should not hold the camera once nobody is using it.
  if (OFFSCREEN && sessions.size === 0) {
    setTimeout(() => {
      if (sessions.size === 0) chrome.runtime.sendMessage({ type: 'release' });
    }, 20000);
  }
}

let sourcePort = null;
function report(text) {
  console.log('[meet]', text);
  try { sourcePort && sourcePort.postMessage({ type: 'log', text }); } catch { /* port gone */ }
}

function connectSource() {
  if (!(globalThis.chrome && chrome.runtime && chrome.runtime.id)) return;
  const port = chrome.runtime.connect({ name: 'source' });
  sourcePort = port;
  // The background worker sleeps and drops this port; reconnect so shortcuts and Meet keep working.
  port.onDisconnect.addListener(() => { if (sourcePort === port) setTimeout(connectSource, 500); });
  report('replaceTrack-v3 ready');
  port.onMessage.addListener(async (m) => {
    if (m.type === 'command') { COMMANDS[m.cmd]?.(); return; }
    report(`got ${m.type} sid=${m.sid} from=${m.from}`);
    const key = `${m.from}:${m.sid}`;
    if (m.type === 'bye') { closeSession(key); return; }
    if (m.type !== 'offer') return;
    port.postMessage({ type: 'ack', to: m.from, sid: m.sid, build: 'replaceTrack-v3' });
    let step = 'start';
    try {
      const pc = new RTCPeerConnection({ iceServers: [] });
      sessions.set(key, pc);
      pc.addEventListener('connectionstatechange', () => {
        if (['failed', 'closed', 'disconnected'].includes(pc.connectionState)) closeSession(key);
      });
      step = 'setRemoteDescription';
      await pc.setRemoteDescription({ type: 'offer', sdp: m.sdp });
      report('remote description set');
      step = 'attach video';
      const tx = pc.getTransceivers().find((t) => t.receiver.track.kind === 'video');
      await tx.sender.replaceTrack(outStream.getVideoTracks()[0]);
      tx.direction = 'sendonly';
      // Keep full resolution (drop frames before pixels) and let the bitrate ramp up.
      try {
        const params = tx.sender.getParameters();
        params.encodings = params.encodings && params.encodings.length ? params.encodings : [{}];
        params.encodings[0].maxBitrate = 4_000_000;
        params.degradationPreference = 'maintain-resolution';
        await tx.sender.setParameters(params);
      } catch (e) { report(`setParameters: ${e.message}`); }
      step = 'createAnswer';
      await pc.setLocalDescription(await pc.createAnswer());
      step = 'ice gathering';
      report('answer created');
      await iceDone(pc);
      port.postMessage({ type: 'answer', to: m.from, sid: m.sid, sdp: pc.localDescription.sdp });
      report('answer sent');
    } catch (e) {
      report(`FAILED at ${step}: ${e && e.message ? e.message : e}`);
      console.warn('Meet session failed at', step, e);
      port.postMessage({ type: 'error', to: m.from, sid: m.sid, message: `${step}: ${e && e.message ? e.message : e}` });
      closeSession(key);
    }
  });
  port.onDisconnect.addListener(() => setTimeout(connectSource, 1000));
}

// ---- HUD ------------------------------------------------------------------
let hintTimer = 0;
function setHint(text, ms) {
  $('hint').textContent = text;
  clearTimeout(hintTimer);
  hintTimer = 0;
  if (ms) hintTimer = setTimeout(() => { hintTimer = 0; $('hint').textContent = S.mode === 'live' ? baseHint() : ''; }, ms);
}

function baseHint() {
  if (S.personNZ === null) return 'Calibrating: stand in view or show your hand.';
  return 'Ready. Snap your fingers.';
}

function updateHud() {
  const set = (id, cls) => { $(id).className = 'dot' + (cls ? ' ' + cls : ''); };
  set('d-cam', S.W ? 'ok' : '');
  set('d-seg', S.personNZ === null ? 'warn' : 'ok');
  $('t-seg').textContent = S.personNZ === null ? 'Calibrating' : 'Person tracked';
  set('d-hand', S.handSeen ? 'ok' : '');
  set('d-meet', sessions.size ? 'ok' : '');
  set('d-mic', audio.on ? (audio.ctx.state === 'running' ? 'ok' : 'warn') : '');
  set('d-inp', inpainter ? 'ok' : 'warn');
  if (S.mode === 'live' && !hintTimer) $('hint').textContent = baseHint();
  else if ((S.mode === 'dusting' || S.mode === 'charging') && !hintTimer) $('hint').textContent = '';
}

// ---- input ----------------------------------------------------------------
function resumeAudio() { if (audio.ctx && audio.ctx.state !== 'running') audio.ctx.resume(); }
addEventListener('pointerdown', resumeAudio);
function toggleRoom() {
  S.solid = !S.solid;
  if (!S.solid && !inpainter) loadInpainter();   // loaded on demand: it is heavy
  reset();
  setHint(S.solid ? 'Flat-colour backdrop' : 'Real room (AI inpainting)', 2000);
}

function nextBackdrop() {
  S.colorIdx = (S.colorIdx + 1) % BACKDROPS();
  if (!S.solid) { S.solid = true; reset(); }
  setHint(S.colorIdx < COLORS.length ? 'Colour backdrop' : 'Picture backdrop', 1500);
}

// Chrome-level shortcuts (manifest "commands"), relayed by the background worker so they
// work while the Meet tab has focus.
const COMMANDS = { reset: () => reset(), 'next-backdrop': nextBackdrop, 'toggle-room': toggleRoom };

addEventListener('keydown', (e) => {
  resumeAudio();
  const k = e.key.toLowerCase();
  if (k === ' ') { e.preventDefault(); trigger(); }
  else if (k === 'r') reset();
  else if (k === 'b') toggleRoom();
  else if (k === 'c') nextBackdrop();
  else if (k === 'h') document.body.classList.toggle('hud-off');
  else if (k === 'f') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (k === 'i' && S.personNZ !== null) {
    S.personNZ = !S.personNZ;
    covered.fill(0); coveredCount = 0;
    pctx.clearRect(0, 0, S.W, S.H);
    reset();
  }
});

// Drop an image onto the window to use it as a backdrop.
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', (e) => {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'));
  if (!f) return;
  pictures.push(loadPicture(URL.createObjectURL(f)));
  S.colorIdx = COLORS.length + pictures.length - 1;
  if (!S.solid) { S.solid = true; reset(); }
  setHint('Picture backdrop', 1500);
});

// ---- boot -----------------------------------------------------------------
function fail(msg) {
  const el = $('error');
  el.textContent = msg;
  el.hidden = false;
}

(async () => {
  try {
    initGL();
    $('hint').textContent = 'Starting camera...';
    await startCamera();
    $('hint').textContent = 'Loading models...';
    await loadModels();
    await startMic();
    outStream = outCanvas.captureStream(30);
    connectSource();
    new Worker('tick-worker.js').onmessage = tick;
    S.mode = 'live';
    $('hint').textContent = '';
    setInterval(updateHud, 250);
    loop();
  } catch (e) {
    console.error(e);
    if (OFFSCREEN && globalThis.chrome && chrome.runtime) {
      chrome.runtime.sendMessage({ type: 'offscreen-failed', error: String(e && e.message ? e.message : e) });
    }
    const denied = e && (e.name === 'NotAllowedError' || e.name === 'NotFoundError');
    fail(denied
      ? 'Camera access was blocked or no camera was found. Allow the camera for this extension (click the camera icon in the address bar) and reload.'
      : 'Something went wrong: ' + (e && e.message ? e.message : e));
  }
})();
