/* MirrorFace — 100% static image processing. No user image is uploaded. */
"use strict";

const MEDIAPIPE_VERSION = "0.10.32";
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/+esm`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MAX_IMAGE_SIDE = 3200;
const MAX_DETECTION_SIDE = 1024;
const MAX_PREVIEW_SIDE = 1100;

const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const dropzone = $("dropzone");
const workspace = $("workspace");
const previewWrap = $("previewWrap");
const previewCanvas = $("previewCanvas");
const leftCanvas = $("leftCanvas");
const rightCanvas = $("rightCanvas");
const faceSelect = $("faceSelect");
const axisRange = $("axisRange");
const axisDisplay = $("axisDisplay");
const detectState = $("detectState");

const sourceCanvas = document.createElement("canvas");
const detectionCanvas = document.createElement("canvas");
let axisX = 0;
let faces = [];
let activeFace = -1;
let currentFilename = "mirrorface";
let loadSequence = 0;
let detectorPromise = null;
let drawingQueued = false;
let pointerDown = false;

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function showStatus(message, kind = "success") {
  detectState.textContent = message;
  detectState.className = "detect-state" + (kind === "success" ? "" : ` ${kind}`);
}

function updateAxis(nextValue, manual = false) {
  if (!sourceCanvas.width) return;
  axisX = clamp(Math.round(nextValue), 1, sourceCanvas.width - 1);
  axisRange.value = String(axisX);
  axisDisplay.textContent = `${Math.round(axisX / sourceCanvas.width * 100)}%`;
  if (manual) { activeFace = -1; faceSelect.value = "manual"; }
  requestDraw();
}

function requestDraw() {
  if (drawingQueued) return;
  drawingQueued = true;
  requestAnimationFrame(() => {
    drawingQueued = false;
    if (!sourceCanvas.width) return;
    drawPreview();
    drawMirroredResults();
  });
}

function drawPreview() {
  const s = Math.min(1, MAX_PREVIEW_SIDE / Math.max(sourceCanvas.width, sourceCanvas.height));
  const pw = Math.max(1, Math.round(sourceCanvas.width * s));
  const ph = Math.max(1, Math.round(sourceCanvas.height * s));
  if (previewCanvas.width !== pw || previewCanvas.height !== ph) {
    previewCanvas.width = pw;
    previewCanvas.height = ph;
  }
  const ctx = previewCanvas.getContext("2d");
  ctx.clearRect(0, 0, pw, ph);
  ctx.drawImage(sourceCanvas, 0, 0, pw, ph);
  const sx = pw / sourceCanvas.width;
  const sy = ph / sourceCanvas.height;
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    ctx.save();
    ctx.lineWidth = i === activeFace ? 2.5 : 1.5;
    ctx.strokeStyle = i === activeFace ? "#b3a6ff" : "rgba(255,255,255,.75)";
    ctx.setLineDash(i === activeFace ? [] : [6, 5]);
    ctx.strokeRect(face.x * sx, face.y * sy, face.width * sx, face.height * sy);
    ctx.restore();
  }
  const x = Math.round(axisX * sx) + 0.5;
  ctx.save();
  ctx.lineWidth = Math.max(2, pw / 450);
  ctx.shadowColor = "rgba(20,10,80,.6)";
  ctx.shadowBlur = 7;
  ctx.strokeStyle = "#ffffff";
  ctx.setLineDash([9, 6]);
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ph); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#6859dd";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2.3;
  ctx.beginPath(); ctx.arc(x, ph / 2, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.restore();
}

// The output is a symmetric crop around the axis. Its original pixels are preserved
// up to the application's MAX_IMAGE_SIDE cap, without stretching either half.
function drawMirroredResults() {
  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const half = Math.max(1, Math.min(axisX, srcW - axisX));
  const cropX = axisX - half;
  const outW = half * 2;
  for (const canvas of [leftCanvas, rightCanvas]) {
    if (canvas.width !== outW || canvas.height !== srcH) {
      canvas.width = outW;
      canvas.height = srcH;
    }
  }
  const left = leftCanvas.getContext("2d");
  left.clearRect(0, 0, outW, srcH);
  left.drawImage(sourceCanvas, cropX, 0, half, srcH, 0, 0, half, srcH);
  left.save();
  left.translate(outW, 0);
  left.scale(-1, 1);
  left.drawImage(sourceCanvas, cropX, 0, half, srcH, 0, 0, half, srcH);
  left.restore();

  const right = rightCanvas.getContext("2d");
  right.clearRect(0, 0, outW, srcH);
  right.drawImage(sourceCanvas, axisX, 0, half, srcH, half, 0, half, srcH);
  right.save();
  right.translate(outW, 0);
  right.scale(-1, 1);
  right.drawImage(sourceCanvas, axisX, 0, half, srcH, half, 0, half, srcH);
  right.restore();
}

function setFaceOptions() {
  faceSelect.replaceChildren();
  const manualOption = new Option("中心線を手動で設定", "manual");
  faceSelect.add(manualOption);
  faces.forEach((face, idx) => {
    const suffix = idx === 0 ? "（最大の顔）" : "";
    faceSelect.add(new Option(`顔 ${idx + 1}${suffix}`, String(idx)));
  });
  faceSelect.value = activeFace < 0 ? "manual" : String(activeFace);
}

function applyFace(index) {
  if (!faces[index]) return;
  activeFace = index;
  faceSelect.value = String(index);
  updateAxis(faces[index].centerX);
}

async function createDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const { FaceDetector, FilesetResolver } = await import(MEDIAPIPE_URL);
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      return FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.45,
      });
    })().catch((err) => { detectorPromise = null; throw err; });
  }
  return detectorPromise;
}

async function detectFaces(sequence) {
  showStatus("AIモデルを読み込み、顔を検知しています…", "loading");
  try {
    const detector = await createDetector();
    if (sequence !== loadSequence) return;
    const ratio = Math.min(1, MAX_DETECTION_SIDE / Math.max(sourceCanvas.width, sourceCanvas.height));
    detectionCanvas.width = Math.max(1, Math.round(sourceCanvas.width * ratio));
    detectionCanvas.height = Math.max(1, Math.round(sourceCanvas.height * ratio));
    detectionCanvas.getContext("2d").drawImage(sourceCanvas, 0, 0, detectionCanvas.width, detectionCanvas.height);
    const result = detector.detect(detectionCanvas);
    if (sequence !== loadSequence) return;
    const scaleX = sourceCanvas.width / detectionCanvas.width;
    const scaleY = sourceCanvas.height / detectionCanvas.height;
    faces = (result.detections ?? [])
      .map((item) => {
        const b = item.boundingBox;
        if (!b) return null;
        const x = b.originX * scaleX;
        const y = b.originY * scaleY;
        const width = b.width * scaleX;
        const height = b.height * scaleY;
        return { x, y, width, height, centerX: x + width / 2 };
      })
      .filter(Boolean)
      .sort((a,b) => b.width*b.height - a.width*a.height);
    if (faces.length) {
      activeFace = 0;
      setFaceOptions();
      applyFace(0);
      showStatus(`${faces.length}人の顔を検知しました。顔の中心に合わせて反転軸を設定しました。`);
    } else {
      activeFace = -1;
      setFaceOptions();
      showStatus("顔を自動検知できませんでした。写真の上で中心線を手動調整できます。", "warning");
      requestDraw();
    }
  } catch (err) {
    if (sequence !== loadSequence) return;
    console.warn("Face detection unavailable:", err);
    faces = [];
    activeFace = -1;
    setFaceOptions();
    showStatus("顔検知モデルを読み込めませんでした。ネット接続を確認してください。手動調整と画像保存は使えます。", "warning");
    requestDraw();
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("この画像を開けませんでした。JPEG・PNGなどで試してください。"));
    img.src = url;
  });
}

async function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(jpe?g|png|webp|gif|heic|heif|avif)$/i.test(file.name)) {
    alert("画像ファイルを選択してください。");
    return;
  }
  const sequence = ++loadSequence;
  showStatus("画像を読み込み中…", "loading");
  const objectUrl = URL.createObjectURL(file);
  let image;
  try {
    image = await loadImage(objectUrl);
  } catch (err) {
    if (sequence === loadSequence) alert(err.message);
    return;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
  if (sequence !== loadSequence) return;
  const origW = image.naturalWidth;
  const origH = image.naturalHeight;
  if (origW < 2 || origH < 2) { alert("画像が小さすぎます。"); return; }
  const ratio = Math.min(1, MAX_IMAGE_SIDE / Math.max(origW, origH));
  sourceCanvas.width = Math.max(2, Math.round(origW * ratio));
  sourceCanvas.height = Math.max(2, Math.round(origH * ratio));
  const sourceCtx = sourceCanvas.getContext("2d", { willReadFrequently: false });
  sourceCtx.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
  currentFilename = file.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9\u3040-\u30ff\u3400-\u9fff_-]+/g, "_").slice(0, 75) || "mirrorface";
  $("fileLabel").textContent = `${file.name} · ${sourceCanvas.width} × ${sourceCanvas.height}px`;
  axisRange.max = String(sourceCanvas.width - 1);
  faces = [];
  activeFace = -1;
  setFaceOptions();
  dropzone.classList.add("hidden");
  workspace.classList.remove("hidden");
  $("emptyResults").classList.add("hidden");
  $("results").classList.remove("hidden");
  updateAxis(Math.round(sourceCanvas.width / 2));
  await detectFaces(sequence);
}

function setAxisFromPointer(event) {
  if (!sourceCanvas.width) return;
  const box = previewCanvas.getBoundingClientRect();
  if (!box.width) return;
  const x = (event.clientX - box.left) / box.width * sourceCanvas.width;
  updateAxis(x, true);
}

async function saveCanvas(canvas, filename) {
  if (!canvas.width || !canvas.height) return;
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) { alert("画像の保存に失敗しました。"); return; }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.add("is-dragging");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault(); dropzone.classList.remove("is-dragging");
}));
dropzone.addEventListener("drop", (event) => handleFile(event.dataTransfer?.files?.[0]));
fileInput.addEventListener("change", () => {
  handleFile(fileInput.files?.[0]);
  fileInput.value = "";
});
$("changeImage").addEventListener("click", () => fileInput.click());
axisRange.addEventListener("input", () => updateAxis(Number(axisRange.value), true));
faceSelect.addEventListener("change", () => {
  if (faceSelect.value === "manual") { activeFace = -1; requestDraw(); return; }
  applyFace(Number(faceSelect.value));
});
$("resetAxis").addEventListener("click", () => {
  if (faces.length) applyFace(activeFace >= 0 ? activeFace : 0);
  else updateAxis(sourceCanvas.width / 2, true);
});
previewWrap.addEventListener("pointerdown", (event) => {
  if (!sourceCanvas.width) return;
  pointerDown = true;
  previewWrap.setPointerCapture(event.pointerId);
  setAxisFromPointer(event);
});
previewWrap.addEventListener("pointermove", (event) => {
  if (pointerDown) setAxisFromPointer(event);
});
previewWrap.addEventListener("pointerup", () => { pointerDown = false; });
previewWrap.addEventListener("pointercancel", () => { pointerDown = false; });
$("saveLeft").addEventListener("click", () => saveCanvas(leftCanvas, `${currentFilename}_left_to_right.png`));
$("saveRight").addEventListener("click", () => saveCanvas(rightCanvas, `${currentFilename}_right_to_left.png`));
