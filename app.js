/* MirrorFace — 100% static image processing. No user image is uploaded. */
"use strict";

const MEDIAPIPE_VERSION = "0.10.32";
const MEDIAPIPE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/+esm`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const MAX_IMAGE_SIDE = 3200;
const MAX_DETECTION_SIDE = 1024;

const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const dropzone = $("dropzone");
const leftCanvas = $("leftCanvas");
const rightCanvas = $("rightCanvas");
const resultGrid = $("resultGrid");
const resultDivider = $("resultDivider");
const angleRange = $("angleRange");
const angleDisplay = $("angleDisplay");
const detectState = $("detectState");

const sourceCanvas = document.createElement("canvas");
const detectionCanvas = document.createElement("canvas");
const halfCanvas = document.createElement("canvas");
let axisX = 0;
let axisY = 0;
let axisAngle = 0;
let initialAxis = { x: 0, y: 0 };
let manualRevision = 0;
let currentFilename = "mirrorface";
let loadSequence = 0;
let detectorPromise = null;
let drawingQueued = false;
let dividerPointerId = null;
let dividerGrabOffset = 0;

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function showStatus(message, kind = "success") {
  detectState.textContent = kind === "success" ? "" : message;
  detectState.className = "detect-state" + (kind === "success" ? " hidden" : ` ${kind}`);
}

function updateAxis(nextValue, manual = false) {
  if (!sourceCanvas.width) return;
  axisX = clamp(Math.round(nextValue), 1, sourceCanvas.width - 1);
  if (manual) manualRevision++;
  requestDraw();
}

function requestDraw() {
  if (drawingQueued) return;
  drawingQueued = true;
  requestAnimationFrame(() => {
    drawingQueued = false;
    if (!sourceCanvas.width) return;
    drawMirroredResults();
  });
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
  // u is perpendicular to the reflection axis; v runs along it.
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
  resultDivider.style.left = `${available * leftCanvas.width / totalWidth + (parseFloat(style.columnGap) || 0) / 2}px`;
  resultDivider.setAttribute("aria-valuenow", String(Math.round(axisX / sourceCanvas.width * 100)));
  for (const canvas of [leftCanvas, rightCanvas]) {
    canvas.style.setProperty("--display-width", `${canvas.width * scale}px`);
    canvas.style.setProperty("--display-height", `${canvas.height * scale}px`);
  }
}

new ResizeObserver(layoutResults).observe(resultGrid);

function setAxisFromDivider(clientX) {
  const rect = resultGrid.getBoundingClientRect();
  const gap = parseFloat(getComputedStyle(resultGrid).columnGap) || 0;
  const width = rect.width - gap;
  if (!sourceCanvas.width || width <= 0) return;
  const fraction = clamp((clientX - dividerGrabOffset - rect.left - gap / 2) / width, 0, 1);
  const theta = axisAngle * Math.PI / 180;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  // Invert the projected width ratio, including the slanted axis's contribution.
  const projectedWidth = cos * sourceCanvas.width + Math.abs(sin) * sourceCanvas.height;
  const leftAtZero = sin >= 0 ? sin * (sourceCanvas.height - axisY) : -sin * axisY;
  updateAxis((fraction * projectedWidth - leftAtZero) / cos, true);
}

resultDivider.addEventListener("pointerdown", (event) => {
  if (dividerPointerId !== null || (event.pointerType === "mouse" && event.button !== 0)) return;
  event.preventDefault();
  const rect = resultDivider.getBoundingClientRect();
  dividerGrabOffset = event.clientX - (rect.left + rect.width / 2);
  dividerPointerId = event.pointerId;
  resultDivider.focus({ preventScroll: true });
  resultDivider.setPointerCapture(event.pointerId);
  resultDivider.classList.add("is-dragging");
});
resultDivider.addEventListener("pointermove", (event) => {
  if (event.pointerId === dividerPointerId) setAxisFromDivider(event.clientX);
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
  resultDivider.addEventListener(name, (event) => {
    if (event.pointerId !== dividerPointerId) return;
    dividerPointerId = null;
    resultDivider.classList.remove("is-dragging");
    if (resultDivider.hasPointerCapture(event.pointerId)) resultDivider.releasePointerCapture(event.pointerId);
  });
}
resultDivider.addEventListener("keydown", (event) => {
  const step = Math.max(1, Math.round(sourceCanvas.width / (event.shiftKey ? 20 : 200)));
  const values = { ArrowLeft: axisX - step, ArrowRight: axisX + step, Home: 1, End: sourceCanvas.width - 1 };
  if (!(event.key in values)) return;
  event.preventDefault();
  updateAxis(values[event.key], true);
});

function updateAngle(value, manual = false) {
  axisAngle = value;
  angleRange.value = String(value);
  angleDisplay.value = `${value}°`;
  angleDisplay.textContent = `${value}°`;
  if (manual) manualRevision++;
  requestDraw();
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
  const revision = manualRevision;
  showStatus("顔を検知中…", "loading");
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
    const faces = (result.detections ?? [])
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
      initialAxis = { x: faces[0].centerX, y: faces[0].centerY };
      // Detection initializes the image once; never overwrite an edit made while loading.
      if (manualRevision === revision) {
        axisY = initialAxis.y;
        updateAxis(initialAxis.x);
      }
      showStatus("");
    } else {
      showStatus("顔を検知できませんでした。中央の↔で調整できます。", "warning");
      requestDraw();
    }
  } catch (err) {
    if (sequence !== loadSequence) return;
    console.warn("Face detection unavailable:", err);
    showStatus("顔検知を読み込めませんでした。中央の↔で調整できます。", "warning");
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
  initialAxis = { x: sourceCanvas.width / 2, y: sourceCanvas.height / 2 };
  manualRevision = 0;
  updateAngle(0);
  dropzone.classList.add("hidden");
  document.querySelector(".workspace-grid").classList.add("has-image");
  $("adjustPanel").classList.remove("hidden");
  $("emptyResults").classList.add("hidden");
  $("results").classList.remove("hidden");
  updateAxis(Math.round(sourceCanvas.width / 2));
  resultDivider.focus({ preventScroll: true });
  await detectFaces(sequence);
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
angleRange.addEventListener("input", () => updateAngle(Number(angleRange.value), true));
$("resetAxis").addEventListener("click", () => {
  axisY = initialAxis.y;
  updateAngle(0);
  updateAxis(initialAxis.x, true);
});
$("saveLeft").addEventListener("click", () => saveCanvas(leftCanvas, `${currentFilename}_left_to_right.png`));
$("saveRight").addEventListener("click", () => saveCanvas(rightCanvas, `${currentFilename}_right_to_left.png`));
