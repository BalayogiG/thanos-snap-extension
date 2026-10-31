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

// Ctrl+Shift+5 in Meet toggles the witch. Chrome allows an extension only four default
// shortcuts (taken by Ctrl+Shift+1-4), so this one is caught on the page instead.
window.addEventListener('keydown', (e) => {
  if (!(e.ctrlKey && e.shiftKey && !e.altKey && !e.metaKey && e.code === 'Digit5') || e.repeat) return;
  e.preventDefault();
  e.stopPropagation();
  try {
    if (!port) connect();
    port.postMessage({ type: 'command', cmd: 'toggle-face' });
  } catch { /* extension context invalidated */ }
}, true);
