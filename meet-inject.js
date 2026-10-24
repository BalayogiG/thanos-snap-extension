// Page-world half: adds a "Thanos Snap" entry to the camera list. When Meet asks for that
// device, it gets the Thanos Snap video, received over a local WebRTC connection. Any other
// camera, and the microphone, are passed through untouched.
(() => {
  const proto = window.MediaDevices && MediaDevices.prototype;
  if (!proto || !proto.getUserMedia) return;
  const origGetUserMedia = proto.getUserMedia;
  const origEnumerateDevices = proto.enumerateDevices;

  const DEVICE_ID = 'thanos-snap-camera';
  const LABEL = 'Thanos Snap';
  const virtualDevice = {
    deviceId: DEVICE_ID, kind: 'videoinput', label: LABEL, groupId: 'thanos-snap',
    toJSON() { return { deviceId: DEVICE_ID, kind: 'videoinput', label: LABEL, groupId: 'thanos-snap' }; },
  };

  if (origEnumerateDevices) {
    proto.enumerateDevices = async function () {
      const list = await origEnumerateDevices.call(this);
      return [...list, virtualDevice];
    };
  }

  const flatten = (x) => (x == null ? [] : Array.isArray(x) ? x.flatMap(flatten)
    : typeof x === 'object' ? flatten(x.exact ?? x.ideal) : [x]);
  const wantsThanos = (video) => !!video && typeof video === 'object' && flatten(video.deviceId)[0] === DEVICE_ID;

  const waiters = new Map();
  const acked = new Set();
  window.addEventListener('message', (e) => {
    if (e.source !== window || !e.data || e.data.__thanos !== 'from-ext') return;
    const m = e.data.msg;
    if (m.type === 'ack') { acked.add(m.sid); console.debug('[Thanos Snap] source acked, build', m.build); window.__thanosBuild = m.build; }
    const w = waiters.get(`${m.sid}:${m.type}`);
    if (w) { waiters.delete(`${m.sid}:${m.type}`); w(m); }
  });
  const send = (msg) => window.postMessage({ __thanos: 'to-ext', msg }, '*');
  const waitFor = (sid, type, ms) => new Promise((resolve, reject) => {
    const t = setTimeout(() => { waiters.delete(`${sid}:${type}`); reject(new Error(`timeout waiting for ${type}`)); }, ms);
    waiters.set(`${sid}:${type}`, (m) => { clearTimeout(t); resolve(m); });
  });

  const iceDone = (pc) => new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') return resolve();
    const done = () => { pc.removeEventListener('icegatheringstatechange', check); resolve(); };
    const check = () => { if (pc.iceGatheringState === 'complete') done(); };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(done, 2000);
  });

  // The connection stays open until the track and every clone of it have been stopped.
  function trackWithLifetime(track, release) {
    let live = 1;
    const wrap = (t) => {
      const stop = t.stop.bind(t);
      t.stop = () => { stop(); if (--live === 0) release(); };
      const clone = t.clone.bind(t);
      t.clone = () => { const c = clone(); live++; wrap(c); return c; };
    };
    wrap(track);
    return track;
  }

  let nextSid = 1;

  // Debug helper: await window.__thanosPing() from the page console.
  window.__thanosPing = () => new Promise((resolve) => {
    const sid = nextSid++;
    const t = setTimeout(() => { waiters.delete(`${sid}:pong`); resolve('no reply from the extension'); }, 3000);
    waiters.set(`${sid}:pong`, (m) => { clearTimeout(t); resolve(m); });
    send({ type: 'ping', sid });
  });
  async function getThanosVideoTrack() {
    const sid = nextSid++;
    const pc = new RTCPeerConnection({ iceServers: [] });
    try {
      const gotTrack = new Promise((resolve) => { pc.ontrack = (ev) => resolve(ev.track); });
      pc.addTransceiver('video', { direction: 'recvonly' });
      await pc.setLocalDescription(await pc.createOffer());
      await iceDone(pc);
      const answered = waitFor(sid, 'answer', 25000);   // covers opening the camera window
      send({ type: 'offer', sid, sdp: pc.localDescription.sdp });
      const failed = waitFor(sid, 'error', 25000).then((m) => { throw new Error(`Thanos Snap window failed at ${m.message}`); });
      answered.catch(() => {});
      failed.catch(() => {});
      let answer;
      try {
        answer = await Promise.race([answered, failed]);
      } catch (e) {
        if (/failed at/.test(e.message)) throw e;
        throw new Error(acked.has(sid)
          ? `the Thanos Snap window (build ${window.__thanosBuild || 'old, no stamp'}) got the request but never answered`
          : 'the Thanos Snap window never received the request (is it open and connected?)');
      }
      await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
      const track = await Promise.race([
        gotTrack,
        new Promise((_, rej) => setTimeout(() => rej(new Error('no video track')), 10000)),
      ]);
      // Present the track as the virtual device so Meet's device bookkeeping stays consistent.
      const getSettings = track.getSettings.bind(track);
      track.getSettings = () => ({ ...getSettings(), deviceId: DEVICE_ID });
      Object.defineProperty(track, 'label', { value: LABEL });
      return trackWithLifetime(track, () => { send({ type: 'bye', sid }); pc.close(); });
    } catch (e) {
      send({ type: 'bye', sid });
      pc.close();
      throw e;
    }
  }

  proto.getUserMedia = async function (constraints) {
    if (!constraints || !wantsThanos(constraints.video)) return origGetUserMedia.call(this, constraints);
    try {
      const tracks = [await getThanosVideoTrack()];
      if (constraints.audio) {
        const mic = await origGetUserMedia.call(this, { audio: constraints.audio });
        tracks.push(...mic.getAudioTracks());
      }
      return new MediaStream(tracks);
    } catch (e) {
      console.warn('[Thanos Snap] falling back to the real camera:', e);
      const { deviceId, ...rest } = constraints.video;
      return origGetUserMedia.call(this, { ...constraints, video: Object.keys(rest).length ? rest : true });
    }
  };
})();
