// Halloween effects drawn over the witch: glowing eyes, ink dripping down the cheeks,
// green mist rising off the hat, and a ghostly glitch every few seconds.
// Everything is placed from the live face landmarks, so it moves with the head.

const DRIP_FROM = [145, 153, 144, 374, 380, 373];   // lower eyelid points, both eyes
const EYES = [   // iris centre, two iris edge points, upper lid, lower lid, two corners
  [468, 469, 471, 159, 145, 33, 133],
  [473, 474, 476, 386, 374, 263, 362],
];
const MIST_MAX = 70;

const rand = (a, b) => a + Math.random() * (b - a);
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);

// pts: reference landmarks in face-image pixels; tris: flat triangle index list over them.
export function createSpooky({ pts, tris, hat, hatX, hatY }) {
  // A reference point is pinned to the mesh triangle it falls in, so it follows the live face exactly.
  const pin = ([x, y]) => {
    for (let i = 0; i < tris.length; i += 3) {
      const [a, b, c] = [pts[tris[i]], pts[tris[i + 1]], pts[tris[i + 2]]];
      const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
      const u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d;
      const v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d;
      if (u >= 0 && v >= 0 && u + v <= 1) return [tris[i], tris[i + 1], tris[i + 2], u, v, 1 - u - v];
    }
    return null;
  };

  const faceLen = dist(pts[10], pts[152]);
  const drips = DRIP_FROM.map((i) => {
    const x0 = pts[i][0] + rand(-4, 4), y0 = pts[i][1] + faceLen * 0.03;
    const len = faceLen * rand(0.18, 0.3), wob = rand(-1, 1) * faceLen * 0.015;
    const path = [];
    for (let k = 0; k <= 12; k++) {
      const p = pin([x0 + wob * Math.sin((k / 12) * Math.PI), y0 + (len * k) / 12]);
      if (!p) break;
      path.push(p);
    }
    return { path, period: rand(4.5, 7.5), phase: rand(0, 7) };
  }).filter((d) => d.path.length > 3);

  // Where mist rises from: the hat's lower edge (scanned from its alpha), or the hairline.
  let emitters = [];
  if (hat) {
    const c = Object.assign(document.createElement('canvas'), { width: hat.width, height: hat.height });
    const x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(hat, 0, 0);
    const a = x.getImageData(0, 0, hat.width, hat.height).data;
    for (let cx = 0; cx < hat.width; cx += 12) {
      for (let cy = hat.height - 1; cy >= 0; cy--) {
        if (a[(cy * hat.width + cx) * 4 + 3] > 128) { emitters.push([hatX + cx, hatY + cy]); break; }
      }
    }
  }
  if (!emitters.length) emitters = [10, 338, 297, 332, 109, 67, 103].map((i) => pts[i]);

  const mist = [];
  let lastT = 0, nextGlitch = rand(3, 6), glitchEnd = 0;

  const live = (pos, p) => [
    pos[p[0] * 3] * p[3] + pos[p[1] * 3] * p[4] + pos[p[2] * 3] * p[5],
    pos[p[0] * 3 + 1] * p[3] + pos[p[1] * 3 + 1] * p[4] + pos[p[2] * 3 + 1] * p[5],
  ];
  const at = (pos, i) => [pos[i * 3], pos[i * 3 + 1]];
  const apply = (m, [x, y]) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

  return {
    // 0 normally; during a glitch, how strong it is right now (0..1, jittering).
    glitch(t) {
      if (t >= nextGlitch) { glitchEnd = t + rand(0.25, 0.6); nextGlitch = glitchEnd + rand(4, 9); }
      return t < glitchEnd ? rand(0.4, 1) : 0;
    },

    // Drawn on the face, under the hat. m: reference -> live similarity, for sizes.
    face(ctx, pos, m, t) {
      const s = Math.hypot(m[0], m[1]);
      ctx.save();
      ctx.lineCap = 'round';
      for (const d of drips) {
        const u = ((t + d.phase) % d.period) / d.period;
        const grow = Math.min(1, u / 0.55), fade = u < 0.7 ? 1 : 1 - (u - 0.7) / 0.3;
        const reach = grow * (d.path.length - 1), whole = Math.floor(reach);
        const p = d.path.map((q) => live(pos, q));
        const head = whole + 1 < p.length
          ? [p[whole][0] + (p[whole + 1][0] - p[whole][0]) * (reach - whole), p[whole][1] + (p[whole + 1][1] - p[whole][1]) * (reach - whole)]
          : p[p.length - 1];
        ctx.strokeStyle = ctx.fillStyle = `rgba(6, 10, 8, ${0.85 * fade})`;
        ctx.lineWidth = 2.8 * s;
        ctx.beginPath();
        ctx.moveTo(p[0][0], p[0][1]);
        for (let k = 1; k <= whole; k++) ctx.lineTo(p[k][0], p[k][1]);
        ctx.lineTo(head[0], head[1]);
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(head[0], head[1] + 1.5 * s, 2.6 * s, 3.8 * s, 0, 0, Math.PI * 2);   // the drop at the tip
        ctx.fill();
      }

      ctx.globalCompositeOperation = 'lighter';
      const pulse = 0.6 + 0.4 * Math.sin(t * 2.6);
      for (const [c, e1, e2, up, low, k1, k2] of EYES) {
        const open = dist(at(pos, up), at(pos, low)) / (dist(at(pos, k1), at(pos, k2)) || 1);
        const lit = Math.min(1, Math.max(0, (open - 0.12) / 0.1)) * pulse;   // dark when the eye is shut
        if (lit <= 0) continue;
        const [x, y] = at(pos, c), r = dist(at(pos, e1), at(pos, e2)) / 2 || 3;
        const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2);
        g.addColorStop(0, `rgba(190, 255, 120, ${0.95 * lit})`);
        g.addColorStop(0.3, `rgba(70, 255, 90, ${0.45 * lit})`);
        g.addColorStop(1, 'rgba(0, 255, 60, 0)');
        ctx.fillStyle = g;
        ctx.fillRect(x - r * 3.2, y - r * 3.2, r * 6.4, r * 6.4);
      }
      ctx.restore();
    },

    // Drawn last, over face and hat.
    over(ctx, m, t, glitch) {
      const s = Math.hypot(m[0], m[1]);
      const dt = Math.min(0.1, Math.max(0, t - lastT));
      lastT = t;
      for (let k = Math.round(dt * 30 + Math.random() * 0.5); k > 0 && mist.length < MIST_MAX; k--) {
        const [x, y] = apply(m, emitters[(Math.random() * emitters.length) | 0]);
        mist.push({ x, y, vx: rand(-25, 25) * s, vy: rand(-55, -25) * s, age: 0, life: rand(1.8, 3.2), r: rand(14, 24) * s });
      }
      ctx.save();
      for (let i = mist.length - 1; i >= 0; i--) {
        const p = mist[i];
        p.age += dt;
        if (p.age >= p.life) { mist.splice(i, 1); continue; }
        p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.98;
        const f = p.age / p.life, r = p.r * (1 + 1.8 * f), a = 0.22 * Math.sin(Math.PI * f);
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
        g.addColorStop(0, `rgba(120, 255, 160, ${a})`);
        g.addColorStop(1, 'rgba(60, 200, 120, 0)');
        ctx.fillStyle = g;
        ctx.fillRect(p.x - r, p.y - r, 2 * r, 2 * r);
      }
      if (glitch) {   // ghostly green wash over everything drawn so far
        ctx.globalCompositeOperation = 'source-atop';
        ctx.fillStyle = `rgba(80, 255, 150, ${0.35 * glitch})`;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
      ctx.restore();
    },
  };
}
