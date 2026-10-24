# Thanos Snap

Chrome extension (Manifest V3). Click the toolbar icon, allow the camera (and mic),
and snap your fingers: you turn to dust and blow away.

## Install
1. Open `chrome://extensions`, enable **Developer mode**.
2. **Load unpacked** and pick this folder.
3. Click the extension icon. Allow camera + microphone.

## How it works
- `selfie_segmenter` (MediaPipe) cuts the person out of each frame.
- On the snap, the frozen frame is inpainted with MI-GAN (onnxruntime-web, CPU/wasm, ~1.5 s):
  your silhouette is removed and the background is synthesized from the same frame,
  so lighting and noise match. No need to step out of view.
- Snap = thumb and middle finger pinch then slam open (`hand_landmarker`),
  or a sharp high-frequency click on the mic right after a pinch.
  Space triggers it manually.
- Your silhouette becomes ~45k WebGL points that dissolve left-to-right and drift away.

## Keys
Space snap, R reset, B flat-colour backdrop / real room, C change backdrop colour, I invert mask (if the wrong side is treated as "person"),
H hide HUD, F fullscreen.

## Bundled files
`lib/` is `@mediapipe/tasks-vision` plus `onnxruntime-web` (wasm build); `models/` holds the Google-hosted
`hand_landmarker.task` and `selfie_segmenter.tflite`, and MI-GAN `migan.onnx` (MIT, Picsart AI Research). Remote code is not allowed in MV3, so they are local.

## Google Meet
Join a call at meet.google.com with the extension loaded. Meet's camera list gets an extra
**Thanos Snap** entry. Pick it (in the preview or under Settings > Video) and the extension opens
the Thanos Snap window and Meet receives its video. Any other camera works as normal; mic is untouched.

- `meet-inject.js` (page world) adds the virtual device and answers `getUserMedia` for it.
- `meet-bridge.js` forwards messages to the extension; `background.js` relays WebRTC signaling and opens the window.
- `app.js` publishes a canvas stream and answers over a local (no STUN) WebRTC connection.
- If the Thanos Snap camera fails, Meet silently falls back to the real camera.
- Keep the Thanos Snap window open, visible, and not minimised, and snap in front of it.
