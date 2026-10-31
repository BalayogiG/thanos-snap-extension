// A page that is not visible stops getting requestAnimationFrame; worker timers keep running.
// Only one tick is in flight at a time: if frames take longer than the interval, a backlog
// of ticks would otherwise starve everything else on the page (e.g. shortcut messages).
let waiting = false;
setInterval(() => { if (!waiting) { waiting = true; postMessage(0); } }, 33);
onmessage = () => { waiting = false; };
