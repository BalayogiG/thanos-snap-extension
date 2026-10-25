# Thanos Snap

Chrome extension (Manifest V3). Click the toolbar icon, allow the camera (and mic),
and snap your fingers: you turn to dust and blow away.

## Install
1. Open `chrome://extensions`, enable **Developer mode**.
2. **Load unpacked** and pick this folder.
3. Click the extension icon. Allow camera + microphone.

## How it works
- `selfie_segmenter` (MediaPipe) cuts the person out of each frame.
- Two backdrop modes (toggle with `B`):
  - **Flat colour / picture** (default): you are cut out and drawn over a solid colour or a picture.
  - **Real room**: on the snap, the frozen frame is inpainted with MI-GAN (onnxruntime-web, CPU/wasm, ~1.5 s):
    your silhouette is removed and the background is synthesized from the same frame,
    so lighting and noise match. No need to step out of view. The model loads on first use.
- **Snap = right hand only.** Thumb and middle finger pinch then slam open (`hand_landmarker`),
  or a sharp high-frequency click on the mic right after a pinch. The left hand is ignored.
  Space triggers it manually.
- Your silhouette becomes ~45k WebGL points that dissolve left-to-right and drift away.

## Backdrops
`C` cycles through four flat colours, then the pictures in `backgrounds/`
(sunset, space, forest, office, beach, city, snow, abstract). Pictures are cover-fitted to the frame.

- **Add your own, temporarily:** drag an image onto the Thanos Snap window. It lasts until you reload.
- **Add your own, permanently:** put an SVG in `backgrounds/` and add its name (without `.svg`) to `PICTURES` in `app.js`.
  For PNG/JPG, also change the `backgrounds/${n}.svg` path in `app.js`.
- Avoid text in pictures: the preview is mirrored, but the output sent to Meet is not.

## Keys
Work when the Thanos Snap window is focused.

| Key | Action |
|---|---|
| Space | Snap |
| R | Reset (bring yourself back) |
| B | Switch between flat/picture backdrop and real room |
| C | Next backdrop (colours, then pictures) |
| I | Invert mask (if the wrong side is treated as "person") |
| H | Hide HUD |
| F | Fullscreen |

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

## Shortcuts in Google Meet
Chrome extension commands work while the Meet tab is focused, as long as the Thanos Snap camera is running:

| Default | Action |
|---|---|
| Ctrl+Shift+1 | Reset |
| Ctrl+Shift+2 | Next backdrop |
| Ctrl+Shift+3 | Switch between flat/picture backdrop and real room |

These are the defaults. Chrome leaves a default unassigned if another extension already uses it. If a shortcut
does nothing, open `chrome://extensions/shortcuts`, find **Thanos Snap**, and set your own keys (they need
`Ctrl` or `Alt`). Your choice is stored by Chrome and overrides the defaults.
