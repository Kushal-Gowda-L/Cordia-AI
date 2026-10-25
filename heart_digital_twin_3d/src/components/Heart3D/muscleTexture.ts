import * as THREE from "three";

/**
 * Procedural cardiac-muscle textures, generated entirely on a <canvas> --
 * no external image assets anywhere in this app. Short overlapping
 * brush-stroke "fibers" at a shared rough orientation plus fine speckle
 * noise approximate the striated, slightly mottled look of real
 * myocardium far better than a flat material color, while staying fully
 * procedural per this project's stated scope (no licensed/segmented
 * anatomy mesh was available to bundle -- see README.md).
 */

function makeCanvas(size = 512) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  return { canvas, ctx };
}

function drawFibers(ctx: CanvasRenderingContext2D, size: number, strokeA: string, strokeB: string, count: number) {
  for (let i = 0; i < count; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const len = size * (0.03 + Math.random() * 0.06);
    const angle = Math.PI / 3 + (Math.random() - 0.5) * 0.9; // mostly-aligned fiber direction, slight variance
    ctx.strokeStyle = Math.random() > 0.5 ? strokeA : strokeB;
    ctx.globalAlpha = 0.1 + Math.random() * 0.22;
    ctx.lineWidth = 1 + Math.random() * 1.6;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export function createMuscleColorTexture(base = "#9c3a38", fiberA = "#b85450", fiberB = "#752523"): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(512);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 512, 512);
  drawFibers(ctx, 512, fiberA, fiberB, 1000);

  const img = ctx.getImageData(0, 0, 512, 512);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 16;
    img.data[i] = Math.min(255, Math.max(0, img.data[i] + n));
    img.data[i + 1] = Math.min(255, Math.max(0, img.data[i + 1] + n * 0.6));
    img.data[i + 2] = Math.min(255, Math.max(0, img.data[i + 2] + n * 0.6));
  }
  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  tex.needsUpdate = true;
  return tex;
}

// Neutral grayscale fiber texture used as a bumpMap on the color-coded
// chamber materials (red/blue), so it adds fibrous surface relief without
// overriding the chamber's own tint.
export function createMuscleBumpTexture(): THREE.CanvasTexture {
  const { canvas, ctx } = makeCanvas(512);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 1200; i++) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const len = 512 * (0.02 + Math.random() * 0.06);
    const angle = Math.PI / 3 + (Math.random() - 0.5) * 0.9;
    const v = Math.random() > 0.5 ? 42 : -42;
    ctx.strokeStyle = `rgb(${128 + v},${128 + v},${128 + v})`;
    ctx.globalAlpha = 0.18 + Math.random() * 0.25;
    ctx.lineWidth = 1 + Math.random() * 1.4;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  tex.needsUpdate = true;
  return tex;
}
