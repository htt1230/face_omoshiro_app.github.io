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
const resultGrid = $("resultGrid");
const faceSelect = $("faceSelect");
const axisRange = $("axisRange");
const axisDisplay = $("axisDisplay");
const angleRange = $("angleRange");
const angleDisplay = $("angleDisplay");
const detectState = $("detectState");

const sourceCanvas = document.createElement("canvas");
const detectionCanvas = document.createElement("canvas");
const halfCanvas = document.createElement("canvas");
let axisX = 0;
let axisY = 0;
let axisAngle = 0;
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
    ctx.strokeStyle = i === activeFace ? "#777" : "rgba(255,255,255,.75)";
    ctx.setLineDash(i === activeFace ? [] : [6, 5]);
    ctx.strokeRect(face.x * sx, face.y * sy, face.width * sx, face.height * sy);
    ctx.restore();
  }
  const cx = axisX * sx;
  const cy = axisY * sy;
  const radians = axisAngle * Math.PI / 180;
  const dx = Math.sin(radians) * Math.max(pw, ph);
  const dy = Math.cos(radians) * Math.max(pw, ph);
  ctx.save();
  ctx.lineWidth = Math.max(2, pw / 450);
  ctx.shadowColor = "rgba(20,10,80,.6)";
  ctx.shadowBlur = 7;
  ctx.strokeStyle = "#ffffff";
  ctx.setLineDash([9, 6]);
  ctx.beginPath(); ctx.moveTo(cx - dx, cy - dy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#555";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2.3;
  ctx.beginPath(); ctx.arc(cx, cy, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.restore();
}

// Clip the rotated source rectangle at u = 0 (the reflection axis).
// Each side keeps its own complete extent instead of sharing the shorter width.
function getSideBounds(corners, side) {
  const inside = ([u]) => side === "left" ? u <= 0 : u >= 0;
  const polygon = [];
  for (let i = 0; i < corners.length; i++) {
    const start = corners[i];
    const end = corners[(i + 1) % corners.length];
    if (inside(start)) polygon.push(start);
    if (inside(start) !== inside(end)) {
      const t = -start[0] / (end[0] - start[0]);
      polygon.push([0, start[1] + t * (end[1] - start[1])]);
    }
  }
  // Ignore floating point noise at integer boundaries, without cropping a pixel.
  const roundExtent = (value) => Math.max(1, Math.ceil(value - 1e-9));
  const halfWidth = roundExtent(Math.max(...polygon.map(([u]) => Math.abs(u))));
  const top = Math.min(...polygon.map(([, v]) => v));
  const bottom = Math.max(...polygon.map(([, v]) => v));
  return { halfWidth, top, height: roundExtent(bottom - top) };
}

function drawMirroredResults() {
  const srcW = sourceCanvas.width, srcH = sourceCanvas.height;
  const theta = axisAngle * Math.PI / 180;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  const cx = axisX, cy = axisY;
  // u is perpendicular to the preview axis; v runs along it.
  const corners = [[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]].map(([x, y]) => [
    cos * (x - cx) - sin * (y - cy),
    sin * (x - cx) + cos * (y - cy),
  ]);
  const drawSide = (canvas, side) => {
    const { halfWidth, top, height } = getSideBounds(corners, side);
    const outW = halfWidth * 2;
    if (canvas.width !== outW || canvas.height !== height) {
      canvas.width = outW;
      canvas.height = height;
    }
    halfCanvas.width = halfWidth;
    halfCanvas.height = height;
    const halfCtx = halfCanvas.getContext("2d");
    const offset = side === "left" ? halfWidth : 0;
    halfCtx.setTransform(cos, sin, -sin, cos,
      offset - cos * cx + sin * cy, -top - sin * cx - cos * cy);
    halfCtx.drawImage(sourceCanvas, 0, 0);

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, outW, height);
    if (side === "left") {
      // The left half's axis edge is on its right.
      ctx.drawImage(halfCanvas, 0, 0);
      ctx.save(); ctx.translate(outW, 0); ctx.scale(-1, 1);
      ctx.drawImage(halfCanvas, 0, 0); ctx.restore();
    } else {
      // The right half's axis edge is on its left. Keep it at the seam.
      ctx.drawImage(halfCanvas, halfWidth, 0);
      ctx.save(); ctx.translate(halfWidth, 0); ctx.scale(-1, 1);
      ctx.drawImage(halfCanvas, 0, 0); ctx.restore();
    }
  };
  drawSide(leftCanvas, "left");
  drawSide(rightCanvas, "right");
  layoutResults();
}

function layoutResults() {
  if (!resultGrid.clientWidth || !sourceCanvas.width) return;
  const style = getComputedStyle(resultGrid);
  const available = Math.max(0, resultGrid.clientWidth - (parseFloat(style.columnGap) || 0));
  const totalWidth = leftCanvas.width + rightCanvas.width;
  const maxHeight = parseFloat(style.getPropertyValue("--result-max-height"));
  const scale = Math.min(available / totalWidth,
    maxHeight / Math.max(leftCanvas.height, rightCanvas.height));

  // Use the entire container. The divider follows the actual output width ratio,
  // with no minimum card width, padding allowance, or horizontal scrolling.
  resultGrid.style.gridTemplateColumns =
    `minmax(0, ${leftCanvas.width}fr) minmax(0, ${rightCanvas.width}fr)`;
  for (const canvas of [leftCanvas, rightCanvas]) {
    canvas.style.setProperty("--display-width", `${canvas.width * scale}px`);
    canvas.style.setProperty("--display-height", `${canvas.height * scale}px`);
  }
}

new ResizeObserver(layoutResults).observe(resultGrid);

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
  axisY = faces[index].centerY;
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
        return { x, y, width, height, centerX: x + width / 2, centerY: y + height / 2 };
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
  axisY = sourceCanvas.height / 2;
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
  $("adjustPanel").classList.remove("hidden");
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
  const y = (event.clientY - box.top) / box.height * sourceCanvas.height;
  const theta = axisAngle * Math.PI / 180;
  const pivotOffset = Math.tan(theta) * (y - axisY);
  updateAxis(x - pivotOffset, true);
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
angleRange.addEventListener("input", () => {
  axisAngle = Number(angleRange.value);
  angleDisplay.value = `${axisAngle}°`;
  angleDisplay.textContent = `${axisAngle}°`;
  requestDraw();
});
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
