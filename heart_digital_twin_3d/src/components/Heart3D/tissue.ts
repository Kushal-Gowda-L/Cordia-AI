import * as THREE from "three";

/**
 * Colouring for the realistic heart, painted per vertex from 3D noise so it
 * has no texture seams or pinched poles and looks right from every side of
 * the 360-degree turn: rich red muscle with darker and brighter blotches,
 * pale peach atria and auricles, and red vessel walls.
 */

function hash(x: number, y: number, z: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return h - Math.floor(h);
}

function smooth(t: number) {
  return t * t * (3 - 2 * t);
}

/** Value noise in [0, 1]. */
export function noise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const zf = smooth(z - zi);
  const lerp = THREE.MathUtils.lerp;
  const c = (dx: number, dy: number, dz: number) => hash(xi + dx, yi + dy, zi + dz);
  return lerp(
    lerp(lerp(c(0, 0, 0), c(1, 0, 0), xf), lerp(c(0, 1, 0), c(1, 1, 0), xf), yf),
    lerp(lerp(c(0, 0, 1), c(1, 0, 1), xf), lerp(c(0, 1, 1), c(1, 1, 1), xf), yf),
    zf
  );
}

/** Fractal noise in roughly [0, 1]. */
export function fbm(p: THREE.Vector3, scale: number, octaves = 4, seed = 0): number {
  let sum = 0;
  let amp = 0.5;
  let f = scale;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise3(p.x * f + seed, p.y * f + seed * 1.7, p.z * f - seed);
    norm += amp;
    amp *= 0.5;
    f *= 2.03;
  }
  return sum / norm;
}

export interface TissuePalette {
  dark: string;
  base: string;
  bright: string;
  fat: string;
}

export const MUSCLE: TissuePalette = { dark: "#850d16", base: "#a8141e", bright: "#cc2a33", fat: "#d9806f" };
export const ATRIUM: TissuePalette = { dark: "#a24b48", base: "#c97568", bright: "#e29d8c", fat: "#eebfae" };
export const AURICLE: TissuePalette = { dark: "#c27a68", base: "#e2a28d", bright: "#f3c3ad", fat: "#f7d6c6" };
export const ARTERY: TissuePalette = { dark: "#6c0a12", base: "#a3161f", bright: "#c9303a", fat: "#d77b72" };
export const VEIN: TissuePalette = { dark: "#5a0c18", base: "#8a1b2b", bright: "#a8394a", fat: "#c98088" };

const ca = new THREE.Color();
const cb = new THREE.Color();

/**
 * Writes a vertex `color` attribute. `fatAt(i, p)` gives 0-1 extra fat at a
 * vertex (grooves, vessel edges), `vesselAt(i)` 0-1 for the raised vessels,
 * `shadeAt(p)` a 0-1 multiplier for broad darker regions.
 */
export function paintTissue(
  geo: THREE.BufferGeometry,
  pal: TissuePalette,
  seed: number,
  fatAt: (i: number, p: THREE.Vector3) => number = () => 0,
  vesselAt: (i: number) => number = () => 0,
  shadeAt: (p: THREE.Vector3) => number = () => 1
) {
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const p = new THREE.Vector3();
  const dark = new THREE.Color(pal.dark);
  const base = new THREE.Color(pal.base);
  const bright = new THREE.Color(pal.bright);
  const fat = new THREE.Color(pal.fat);
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    const large = fbm(p, 2.3, 4, seed);
    const small = fbm(p, 9, 3, seed + 11);
    // Muscle: dark-to-red blotches, with brighter wet patches.
    ca.copy(dark).lerp(base, THREE.MathUtils.smoothstep(large, 0.3, 0.62));
    ca.lerp(bright, 0.55 * THREE.MathUtils.smoothstep(small, 0.58, 0.8));
    // Pale fat, patchy rather than a clean band.
    const patch = THREE.MathUtils.smoothstep(fbm(p, 5, 3, seed + 23), 0.38, 0.68);
    const f = THREE.MathUtils.clamp(fatAt(i, p) * (0.35 + 0.9 * patch), 0, 0.92);
    ca.lerp(fat, f);
    // Vessels: a redder line through the fat.
    const v = vesselAt(i);
    if (v > 0) ca.lerp(cb.copy(bright).multiplyScalar(0.85), v * 0.8);
    ca.multiplyScalar(shadeAt(p));
    colors.set([ca.r, ca.g, ca.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

/** Small lumps over a surface, along its normals: the uneven, fatty texture of a real heart. */
export function roughenSurface(geo: THREE.BufferGeometry, amount: number, seed: number) {
  const pos = geo.attributes.position;
  const nrm = geo.attributes.normal;
  const p = new THREE.Vector3();
  const n = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    n.fromBufferAttribute(nrm, i);
    const d = (fbm(p, 6, 3, seed) - 0.5) * 2 * amount;
    p.addScaledVector(n, d);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  geo.computeVertexNormals();
}
