// Isolated-world half: forwards messages between the page script and the extension.
let port = null;

function connect() {
  port = chrome.runtime.connect({ name: 'meet' });
  port.onMessage.addListener((msg) => window.postMessage({ __thanos: 'from-ext', msg }, '*'));
  port.onDisconnect.addListener(() => { port = null; });
}

try { connect(); } catch { /* extension reloaded */ }

window.addEventListener('message', (e) => {
  if (e.source !== window || !e.data || e.data.__thanos !== 'to-ext') return;
  try {
    if (!port) connect();
    port.postMessage(e.data.msg);
  } catch { /* extension context invalidated */ }
});
