import { FaceLandmarker } from './lib/vision_bundle.mjs';
import { createSpooky } from './spooky.js';

// Face swap: the reference face is cut into triangles over its landmarks, and each frame
// those triangles are stretched onto the live face's landmarks (WebGL, textured mesh).

// Ordered outline of the face and of the inner lips (MediaPipe face mesh indices).
const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const INNER_LIPS = new Set([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]);
// Upper-face points the hat (and the mist) are pinned to.
const HAT_ANCHORS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 234, 127, 162, 21, 54, 103, 67, 109, 33, 263, 168];
const SMOOTH = 0.6;   // share of the new landmark position taken each frame (lower = steadier, laggier)

const VERT = `
attribute vec3 a_p;
attribute vec2 a_uv;
attribute float a_a;
uniform vec2 u_res;
varying vec2 v_uv;
varying float v_a;
void main() {
  v_uv = a_uv;
  v_a = a_a;
  gl_Position = vec4(a_p.x / u_res.x * 2.0 - 1.0, 1.0 - a_p.y / u_res.y * 2.0, clamp(a_p.z * 2.0, -1.0, 1.0), 1.0);
}`;
const FRAG = `
precision mediump float;
uniform sampler2D u_tex;
varying vec2 v_uv;
varying float v_a;
void main() {
  float a = smoothstep(0.0, 1.0, v_a);
  gl_FragColor = vec4(texture2D(u_tex, v_uv).rgb * a, a);
}`;

// Bowyer-Watson; fine for a few hundred points done once.
function delaunay(pts) {
  const n = pts.length;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const d = Math.max(x1 - x0, y1 - y0) * 20, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const P = pts.concat([[mx - d, my - d], [mx + d, my - d], [mx, my + d]]);
  const inCircle = ([a, b, c], [px, py]) => {
    const ax = P[a][0] - px, ay = P[a][1] - py, bx = P[b][0] - px, by = P[b][1] - py, cx = P[c][0] - px, cy = P[c][1] - py;
    const det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (cx * cx + cy * cy) * (ax * by - bx * ay);
    const orient = (P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0]);
    return orient > 0 ? det > 0 : det < 0;
  };
  let tris = [[n, n + 1, n + 2]];
  for (let i = 0; i < n; i++) {
    const keep = [], edges = new Map();
    for (const t of tris) {
      if (!inCircle(t, P[i])) { keep.push(t); continue; }
      for (const [a, b] of [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]) {
        const k = a < b ? `${a},${b}` : `${b},${a}`;
        if (edges.has(k)) edges.delete(k); else edges.set(k, [a, b]);
      }
    }
    for (const [a, b] of edges.values()) keep.push([a, b, i]);
    tris = keep;
  }
  return tris.filter((t) => t.every((v) => v < n));
}

// Least-squares rotation + uniform scale + translation taking points a onto points b.
function fitSimilarity(a, b) {
  const n = a.length;
  let ax = 0, ay = 0, bx = 0, by = 0;
  for (let i = 0; i < n; i++) { ax += a[i][0]; ay += a[i][1]; bx += b[i][0]; by += b[i][1]; }
  ax /= n; ay /= n; bx /= n; by /= n;
  let dot = 0, cross = 0, v = 0;
  for (let i = 0; i < n; i++) {
    const px = a[i][0] - ax, py = a[i][1] - ay, qx = b[i][0] - bx, qy = b[i][1] - by;
    dot += px * qx + py * qy; cross += px * qy - py * qx; v += px * px + py * py;
  }
  const c = dot / v, s = cross / v;
  return [c, s, -s, c, bx - (c * ax - s * ay), by - (s * ax + c * ay)];   // canvas setTransform order
}

// Not Image.decode(): it can wait forever in the hidden offscreen page, which never paints.
const loadImage = async (url) => createImageBitmap(await (await fetch(url)).blob());

function insidePolygon(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// look = { face: image url, hat: { src, x, y } (optional; x/y = its top-left in face-image pixels) }
export async function createFaceSwap(fileset, delegate, look, W, H, log = () => {}) {
  const imageUrl = look.face;
  log(`loading face model (${delegate})`);
  const landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: 'models/face_landmarker.task', delegate },
    runningMode: 'IMAGE',
    numFaces: 1,
  });
  log('face model ready');
  const img = await loadImage(imageUrl);
  log('face image decoded');
  const ref = landmarker.detect(img).faceLandmarks[0];
  if (!ref) throw new Error(`no face found in ${imageUrl}`);
  log('reference face found');
  await landmarker.setOptions({ runningMode: 'VIDEO' });
  log('switched to video mode');
  let hat = null;
  if (look.hat) {
    hat = await loadImage(look.hat.src);
  }

  // Mesh over the reference face: keep triangles inside the face outline, drop the mouth
  // opening (so the real mouth shows when talking), and wind them all the same way so
  // triangles that flip when the head turns away get culled.
  const pts = ref.map((p) => [p.x * img.width, p.y * img.height]);
  const oval = OVAL.map((i) => pts[i]);
  const tris = [];
  for (const t of delaunay(pts)) {
    if (t.every((v) => INNER_LIPS.has(v))) continue;
    const [a, b, c] = t.map((v) => pts[v]);
    if (!insidePolygon(oval, (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3)) continue;
    const cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    tris.push(...(cross < 0 ? t : [t[0], t[2], t[1]]));
  }

  const canvas = Object.assign(document.createElement('canvas'), { width: W, height: H });
  const overlay = Object.assign(document.createElement('canvas'), { width: W, height: H });   // face + hat
  const octx = overlay.getContext('2d', { willReadFrequently: true });
  const hatRef = HAT_ANCHORS.map((i) => pts[i]);
  const gl = canvas.getContext('webgl', { premultipliedAlpha: true, alpha: true, preserveDrawingBuffer: true });
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);

  const n = ref.length;
  const edge = new Set(OVAL);
  const attr = (name, size, data, usage) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    const l = gl.getAttribLocation(prog, name);
    gl.enableVertexAttribArray(l);
    gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0);
    return b;
  };
  attr('a_uv', 2, new Float32Array(ref.flatMap((p) => [p.x, p.y])), gl.STATIC_DRAW);
  attr('a_a', 1, new Float32Array(ref.map((_, i) => (edge.has(i) ? 0 : 1))), gl.STATIC_DRAW);   // feathered outline
  const pos = new Float32Array(n * 3);
  const posBuf = attr('a_p', 3, pos, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(tris), gl.STATIC_DRAW);

  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
    [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  gl.uniform2f(gl.getUniformLocation(prog, 'u_res'), W, H);
  gl.enable(gl.CULL_FACE);
  gl.enable(gl.DEPTH_TEST);
  gl.viewport(0, 0, W, H);

  const spooky = createSpooky({ pts, tris, hat, hatX: look.hat?.x ?? 0, hatY: look.hat?.y ?? 0 });
  let found = false, fresh = true;

  return {
    get found() { return found; },
    // Tracks the face in the current video frame and redraws the mesh onto its own canvas.
    update(source, now) {
      const lm = landmarker.detectForVideo(source, now).faceLandmarks[0];
      found = !!lm;
      if (!found) { fresh = true; return; }
      const k = fresh ? 1 : SMOOTH;
      for (let i = 0; i < n; i++) {
        const j = i * 3;
        pos[j] += (lm[i].x * W - pos[j]) * k;
        pos[j + 1] += (lm[i].y * H - pos[j + 1]) * k;
        pos[j + 2] += (lm[i].z - pos[j + 2]) * k;
      }
      fresh = false;
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.drawElements(gl.TRIANGLES, tris.length, gl.UNSIGNED_SHORT, 0);

      const t = now / 1000;
      const m = fitSimilarity(hatRef, HAT_ANCHORS.map((i) => [pos[i * 3], pos[i * 3 + 1]]));
      const glitch = spooky.glitch(t);
      octx.setTransform(1, 0, 0, 1, 0, 0);
      octx.clearRect(0, 0, W, H);
      if (glitch) {   // face torn into horizontally shifted bands, flickering
        octx.globalAlpha = 1 - 0.6 * glitch * Math.random();
        const bands = 7, h = Math.ceil(H / bands);
        for (let b = 0; b < bands; b++) {
          const dx = (Math.random() - 0.5) * 40 * glitch * Math.hypot(m[0], m[1]);
          octx.drawImage(canvas, 0, b * h, W, h, dx, b * h, W, h);
        }
        octx.globalAlpha = 1;
      } else {
        octx.drawImage(canvas, 0, 0);
      }
      spooky.face(octx, pos, m, t);
      if (hat) {
        octx.setTransform(...m);
        octx.drawImage(hat, look.hat.x, look.hat.y);
        octx.setTransform(1, 0, 0, 1, 0, 0);
      }
      spooky.over(octx, m, t, glitch);
    },
    draw(ctx) { if (found) ctx.drawImage(overlay, 0, 0); },
    // RGBA of the face + hat layer, or null when no face is tracked.
    pixels() { return found ? octx.getImageData(0, 0, W, H).data : null; },
  };
}
