// Relays WebRTC signaling between Google Meet pages ("meet" ports) and the Thanos Snap
// camera window ("source" port), and opens that window when Meet asks for the camera.
const PAGE = chrome.runtime.getURL('app.html');

const logs = [];
const log = (t) => { logs.push(`${new Date().toISOString().slice(11, 23)} ${t}`); if (logs.length > 60) logs.shift(); };

let source = null;
let nextId = 1;
const meets = new Map();
const queue = [];

const getWindowId = async () => (await chrome.storage.session.get('windowId')).windowId ?? null;
const setWindowId = (windowId) => chrome.storage.session.set({ windowId });

// Calls are chained so concurrent requests can't each create their own popup.
let windowLock = Promise.resolve();
function ensureWindow(focused) {
  const run = windowLock.then(async () => {
    const id = await getWindowId();
    if (id !== null) {
      try {
        await chrome.windows.get(id);
        if (focused) await chrome.windows.update(id, { focused: true });
        return;
      } catch { /* window is gone */ }
    }
    const w = await chrome.windows.create({ url: PAGE, type: 'popup', width: 980, height: 640, focused });
    await setWindowId(w.id);
  });
  windowLock = run.catch(() => {});
  return run;
}

chrome.windows.onRemoved.addListener(async (id) => {
  if (id === (await getWindowId())) await setWindowId(null);
});

// Meet normally gets its video from a hidden offscreen page; the toolbar icon opens the
// visible window instead (closing the hidden one so only one page owns the camera).
let offscreenFailed = false;

let sourceLock = Promise.resolve();
function ensureSource() {
  const run = sourceLock.then(startSource);
  sourceLock = run.catch(() => {});
  return run;
}

async function startSource() {
  const id = await getWindowId();
  if (id !== null) {
    try { await chrome.windows.get(id); return; } catch { /* window is gone */ }
  }
  if (!offscreenFailed) {
    try {
      await chrome.offscreen.createDocument({
        url: 'app.html?mode=offscreen',
        reasons: ['USER_MEDIA'],
        justification: 'Run the camera effect and stream it to Google Meet without a visible window.',
      });
      log('offscreen document created');
      return;
    } catch (e) {
      if (/single/i.test(e.message)) return;   // already running
      log(`offscreen failed: ${e.message}`);
      offscreenFailed = true;
    }
  }
  await ensureWindow(false);
}

chrome.action.onClicked.addListener(async () => {
  try { await chrome.offscreen.closeDocument(); log('closed offscreen document for the visible window'); } catch { /* none open */ }
  await ensureWindow(true);
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'offscreen-failed') {
    log(`offscreen page could not start: ${msg.error}`);
    offscreenFailed = true;
    chrome.offscreen.closeDocument().catch(() => {});
    ensureWindow(false);
  } else if (msg.type === 'release') {
    chrome.offscreen.closeDocument().then(() => log('released offscreen camera')).catch(() => {});
  }
});

// Keyboard shortcuts (chrome://extensions/shortcuts) work while the Meet tab is focused.
chrome.commands.onCommand.addListener((cmd) => {
  if (source) source.postMessage({ type: 'command', cmd });
  else log(`command ${cmd} ignored: no source`);
});

function toSource(msg) {
  if (source) {
    try { source.postMessage(msg); log(`-> source: ${msg.type} sid=${msg.sid} from=${msg.from}`); return; }
    catch (e) { log(`source port dead: ${e.message}`); source = null; }
  }
  log(`queued ${msg.type}, starting source`);
  queue.push(msg);
  ensureSource();
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'source') {
    source = port;
    log(`source connected (tab=${port.sender?.tab?.id}), flushing ${queue.length}`);
    if (port.sender?.tab) setWindowId(port.sender.tab.windowId);
    while (queue.length) port.postMessage(queue.shift());
    port.onMessage.addListener((m) => {
      if (m.type === 'log') { log(`popup: ${m.text}`); return; }
      log(`source -> meet ${m.to}: ${m.type} sid=${m.sid} (meet port ${meets.has(m.to) ? 'alive' : 'GONE'})`);
      meets.get(m.to)?.postMessage(m);
    });
    port.onDisconnect.addListener(() => { log('source disconnected'); if (source === port) source = null; });
  } else if (port.name === 'meet') {
    const id = nextId++;
    meets.set(id, port);
    log(`meet port ${id} connected`);
    port.onMessage.addListener((m) => {
      if (m.type === 'ping') { port.postMessage({ type: 'pong', sid: m.sid, source: !!source, queued: queue.length, logs }); return; }
      toSource({ ...m, from: id });
    });
    port.onDisconnect.addListener(() => { log(`meet port ${id} disconnected`); meets.delete(id); });
  }
});
