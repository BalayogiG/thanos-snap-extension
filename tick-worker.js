// A page that is not visible stops getting requestAnimationFrame; worker timers keep running.
setInterval(() => postMessage(0), 33);
