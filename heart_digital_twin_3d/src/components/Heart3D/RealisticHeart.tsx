import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { createMuscleBumpTexture } from "./muscleTexture";
import { useHeartStore } from "../../state/store";
import { ModelValves } from "./ModelValves";
import { ModelChambers } from "./ModelChambers";
import { SHELL_CUT } from "./shellCut";
import { ARTERY, ATRIUM, AURICLE, MUSCLE, VEIN, fbm, paintTissue, roughenSurface } from "./tissue";

/**
 * The "Realistic heart" view: the closed heart as an anatomical model,
 * styled on a sculpted anatomy render: rich red ventricles with the
 * muscle bundles showing as fine grooves running toward the apex, pale
 * peach atria with flap-like, scalloped auricles, a branching tree of
 * coronary arteries (dark red) and cardiac veins (pale grey) laid over the
 * surface, and the great vessels with open, hollow ends: the aortic arch
 * with its three branches, the pulmonary trunk and arteries, the venae
 * cavae and pulmonary veins.
 *
 * It is still the live twin: the ventricles squeeze with the simulated
 * ventricular volume and the atria with the atrial volume, so heart rate,
 * pumping strength and every data change show here too.
 *
 * Orientation follows the anterior view: the heart's left side (LV, aortic
 * arch, left pulmonary veins) is on the viewer's right.
 */

const LUMEN = "#3a0509";
export const VEINS = new Set(["Superior vena cava", "Inferior vena cava", "Left upper pulmonary vein", "Left lower pulmonary vein", "Right pulmonary veins"]);
const R = 0.95;

export type V3 = [number, number, number];

export const dirOf = (polar: number, azimuth: number) =>
  new THREE.Vector3(Math.sin(polar) * Math.cos(azimuth), Math.cos(polar), Math.sin(polar) * Math.sin(azimuth));

// Tilt so the apex points down and to the viewer's right, as in the chest.
export const BODY_TILT = new THREE.Euler(0, -0.25, 0.35);
export const BODY_SHIFT = new THREE.Vector3(0.02, -0.2, 0);

/** 0-1: how deep in a muscle groove a direction lies (0 on the base and at the very tip). */
function grooveAt(d: THREE.Vector3): number {
  const az = Math.atan2(d.z, d.x);
  // Grooves wind slightly as they descend, like the spiralling muscle layers.
  const wave = 0.5 + 0.5 * Math.cos(az * 24 + d.y * 2.2 + 1.3 * Math.sin(az * 3 + d.y * 4));
  const mask = THREE.MathUtils.smoothstep(-d.y, -0.15, 0.25) * (1 - THREE.MathUtils.smoothstep(-d.y, 0.9, 0.99));
  return Math.pow(wave, 3) * mask;
}

/** A unit direction -> a point on the ventricular surface (broad base, tapered apex, LV side fuller). */
export function bodyPoint(d: THREE.Vector3): THREE.Vector3 {
  const y = d.y;
  const taper = y < 0 ? 1 - 0.6 * Math.pow(-y, 1.25) : 1 - 0.08 * y * y;
  let x = d.x * R * taper * (d.x > 0 ? 1.08 : 0.96);
  const z = d.z * R * 0.78 * taper * (d.z > 0 ? 1.05 : 1);
  // Short, rounded base above the equator, long tapering apex below, blended smoothly (no crease).
  const k = 0.5 * (1 + Math.tanh(4 * y));
  const yy = y * R * (1.5 * (1 - k) + 0.72 * k);
  // Gentle organic unevenness so it doesn't read as a smooth egg, and the
  // muscle bundles as shallow grooves running from the base toward the apex.
  const n = 0.018 * Math.sin(3.1 * d.x + 1.7 * y) * Math.cos(2.3 * d.z + 0.5) + 0.008 * Math.sin(7 * d.x - 5 * y + 3 * d.z) - 0.04 * grooveAt(d);
  x *= 1 + n;
  return new THREE.Vector3(x, yy * (1 + n), z * (1 + n)).applyEuler(BODY_TILT).add(BODY_SHIFT);
}

// Main coronary vessels in [polar, azimuth] over the body (azimuth PI/2 faces the viewer, -PI/2 is the back).
interface SurfaceVessel {
  path: [number, number][];
  width: number;
  vein: boolean;
  branches: number;
}
const MAIN_VESSELS: SurfaceVessel[] = [
  { path: [[1.1, 1.45], [1.5, 1.5], [1.9, 1.45], [2.3, 1.3], [2.75, 1.0], [3.02, 0.55]], width: 0.022, vein: false, branches: 7 }, // LAD
  { path: [[1.28, 1.95], [1.4, 2.4], [1.46, 2.9], [1.52, 3.4], [1.56, 3.9]], width: 0.02, vein: false, branches: 5 }, // right coronary
  { path: [[1.56, 3.9], [1.95, 4.0], [2.4, 4.15], [2.8, 4.3]], width: 0.016, vein: false, branches: 4 }, // posterior descending
  { path: [[1.16, 1.2], [1.27, 0.7], [1.33, 0.2], [1.38, -0.3], [1.42, -0.8]], width: 0.019, vein: false, branches: 5 }, // circumflex
  { path: [[1.3, 0.5], [1.8, 0.38], [2.25, 0.32], [2.6, 0.45]], width: 0.015, vein: false, branches: 4 }, // obtuse marginal
  { path: [[1.43, 2.6], [1.9, 2.55], [2.35, 2.35], [2.65, 2.1]], width: 0.014, vein: false, branches: 3 }, // acute marginal
  { path: [[1.2, 1.7], [1.65, 1.75], [2.15, 1.68], [2.6, 1.5]], width: 0.017, vein: true, branches: 5 }, // great cardiac vein
  { path: [[1.45, -1.45], [1.9, -1.4], [2.35, -1.3], [2.8, -1.1]], width: 0.018, vein: true, branches: 5 }, // middle cardiac vein
  { path: [[1.4, -0.55], [1.85, -0.65], [2.3, -0.7]], width: 0.014, vein: true, branches: 3 }, // posterior LV vein
  { path: [[1.38, 2.95], [1.8, 3.05], [2.2, 3.0]], width: 0.012, vein: true, branches: 2 }, // anterior cardiac vein
];

/** Deterministic random numbers, so the vessel tree is the same on every load. */
function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) / 2147483647);
}

/**
 * Grows the coronary tree: each main vessel sprouts side branches that head
 * toward the apex, wander a little and fork again, thinning as they go.
 */
export function growTree(): { points: [number, number][]; width: number; vein: boolean }[] {
  const out: { points: [number, number][]; width: number; vein: boolean }[] = [];
  const rand = rng(97);
  const grow = (start: [number, number], heading: number, length: number, width: number, vein: boolean, depth: number) => {
    const pts: [number, number][] = [start];
    let [pol, az] = start;
    let h = heading;
    const steps = Math.max(3, Math.round(length / 0.07));
    for (let k = 0; k < steps; k++) {
      h += (rand() - 0.5) * 0.5;
      // Pulled toward the apex (heading 0 = straight down).
      h *= 0.9;
      pol = Math.min(3.05, pol + 0.07 * Math.cos(h));
      az += (0.07 * Math.sin(h)) / Math.max(0.35, Math.sin(pol));
      pts.push([pol, az]);
      if (depth > 0 && k > 1 && rand() < 0.35) {
        grow([pol, az], h + (rand() < 0.5 ? -1 : 1) * (0.6 + 0.4 * rand()), length * 0.5, width * 0.65, vein, depth - 1);
      }
    }
    out.push({ points: pts, width, vein });
  };
  for (const v of MAIN_VESSELS) {
    out.push({ points: v.path, width: v.width, vein: v.vein });
    const curve = new THREE.SplineCurve(v.path.map(([p, a]) => new THREE.Vector2(p, a)));
    for (let b = 0; b < v.branches; b++) {
      const t = (b + 0.6) / (v.branches + 0.4);
      const at = curve.getPointAt(t);
      const side = b % 2 ? 1 : -1;
      grow([at.x, at.y], side * (0.8 + 0.5 * rand()), 0.35 + 0.35 * rand() * (1 - t * 0.5), v.width * 0.62, v.vein, 2);
    }
  }
  return out;
}

/**
 * A surface path in [polar, azimuth] as 3D points riding just proud of the
 * ventricles. Resampled densely in surface coordinates first, so anything
 * drawn along it hugs the curved surface instead of cutting straight chords
 * across it.
 */
export function surfacePolyline(points: [number, number][], lift: number, spacing = 0.03): THREE.Vector3[] {
  const flat = new THREE.SplineCurve(points.map(([a, b]) => new THREE.Vector2(a, b)));
  const dense = flat.getSpacedPoints(Math.max(4, Math.ceil(flat.getLength() / spacing)));
  return dense.map(({ x: pol, y: az }) => {
    const p = bodyPoint(dirOf(pol, az));
    return p.add(p.clone().sub(BODY_SHIFT).normalize().multiplyScalar(lift));
  });
}

/** The coronary tree as two merged meshes laid over the ventricles: arteries and veins. */
export function makeCoronaries(): { arteries: THREE.BufferGeometry; veins: THREE.BufferGeometry } {
  const arteries: THREE.BufferGeometry[] = [];
  const veins: THREE.BufferGeometry[] = [];
  for (const v of growTree()) {
    if (v.points.length < 2) continue;
    // Ride on the surface, half proud of it.
    const pts = surfacePolyline(v.points, v.width * 0.45);
    const curve = new THREE.CatmullRomCurve3(pts);
    const tubular = Math.max(8, pts.length * 2);
    const radial = 8;
    const geo = new THREE.TubeGeometry(curve, tubular, v.width, radial, false);
    // Taper each vessel toward its tip.
    const pos = geo.attributes.position;
    const p = new THREE.Vector3();
    for (let i = 0; i <= tubular; i++) {
      const c = curve.getPointAt(i / tubular);
      const s = 1 - 0.65 * (i / tubular);
      for (let j = 0; j <= radial; j++) {
        const k = i * (radial + 1) + j;
        p.fromBufferAttribute(pos, k).sub(c).multiplyScalar(s).add(c);
        pos.setXYZ(k, p.x, p.y, p.z);
      }
    }
    geo.computeVertexNormals();
    (v.vein ? veins : arteries).push(geo);
  }
  return { arteries: mergeGeometries(arteries)!, veins: mergeGeometries(veins)! };
}

export function makeBody(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 200, 150);
  const pos = geo.attributes.position;
  const d = new THREE.Vector3();
  const dirY = new Float32Array(pos.count);
  const groove = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize();
    dirY[i] = d.y;
    groove[i] = grooveAt(d);
    const p = bodyPoint(d);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  geo.computeVertexNormals();
  roughenSurface(geo, 0.008, 3);

  // A little pale fat in the groove between atria and ventricles; the
  // muscle grooves and the apex are the darkest red.
  paintTissue(
    geo,
    MUSCLE,
    7,
    (i) => 0.55 * Math.exp(-Math.pow((dirY[i] - 0.32) / 0.12, 2)),
    () => 0,
    (q) => 1.12 - 0.35 * THREE.MathUtils.smoothstep(-q.y, 0.7, 1.6)
  );
  const col = geo.attributes.color;
  for (let i = 0; i < pos.count; i++) {
    const k = 1 - 0.5 * groove[i];
    col.setXYZ(i, col.getX(i) * k, col.getY(i) * k, col.getZ(i) * k);
  }
  return geo;
}

/** An auricle: a thin, flap-like pouch with a scalloped, crinkled edge. */
export function makeFlap(scale: V3, seed: number): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 64, 40);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const ang = Math.atan2(v.z, v.x);
    const edge = 1 + 0.13 * Math.sin(5 * ang + seed) + 0.07 * Math.sin(11 * ang + 2 * seed);
    const crinkle = 0.25 * Math.sin(9 * ang + seed) * (1 - Math.abs(v.y));
    pos.setXYZ(i, v.x * scale[0] * edge, (v.y + crinkle * 0.6) * scale[1], v.z * scale[2] * edge);
  }
  geo.computeVertexNormals();
  roughenSurface(geo, 0.008, seed);
  paintTissue(geo, AURICLE, seed * 17, (_i, p) => 0.3 * fbm(p, 4, 2, seed));
  return geo;
}

/** A soft lobe (atrium or auricle): a squashed, slightly lumpy sphere. */
export function makeLobe(scale: V3, lumps = 0.06, seed = 1): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 48, 36);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = 1 + lumps * Math.sin(5 * v.x + seed) * Math.sin(4 * v.y + 2 * seed) * Math.cos(6 * v.z);
    pos.setXYZ(i, v.x * scale[0] * n, v.y * scale[1] * n, v.z * scale[2] * n);
  }
  geo.computeVertexNormals();
  roughenSurface(geo, 0.01, seed);
  paintTissue(geo, ATRIUM, seed * 13, (_i, p) => 0.3 * fbm(p, 3, 2, seed));
  return geo;
}

export interface GreatVessel {
  name: string;
  points: V3[];
  radius: number;
  /** End radius as a fraction of the start, for tapering branches. */
  taper?: number;
  open: boolean; // last point is an open, cut end
}

export const GREAT_VESSELS: GreatVessel[] = [
  // The aorta rises behind the pulmonary trunk and arches back and to the heart's left.
  { name: "Aorta", points: [[0.02, 0.2, 0.0], [-0.05, 0.6, 0.06], [-0.04, 0.98, 0.0], [0.12, 1.16, -0.2], [0.38, 1.08, -0.42], [0.5, 0.75, -0.55], [0.52, 0.2, -0.6], [0.5, -0.45, -0.6]], radius: 0.19, open: true },
  { name: "Brachiocephalic trunk", points: [[-0.04, 1.08, -0.08], [-0.1, 1.35, -0.08], [-0.16, 1.58, -0.08]], radius: 0.075, taper: 0.85, open: true },
  { name: "Left common carotid", points: [[0.14, 1.17, -0.22], [0.15, 1.4, -0.22], [0.14, 1.62, -0.2]], radius: 0.058, taper: 0.9, open: true },
  { name: "Left subclavian", points: [[0.3, 1.12, -0.33], [0.36, 1.35, -0.34], [0.44, 1.55, -0.36]], radius: 0.06, taper: 0.9, open: true },
  // Pulmonary trunk leaves the front of the right ventricle, crossing in front of the aorta.
  { name: "Pulmonary trunk", points: [[-0.12, 0.25, 0.36], [-0.05, 0.55, 0.36], [0.08, 0.8, 0.22], [0.16, 0.9, 0.06]], radius: 0.17, open: false },
  { name: "Left pulmonary artery", points: [[0.16, 0.9, 0.06], [0.42, 0.98, 0.0], [0.72, 0.95, -0.06]], radius: 0.1, taper: 0.9, open: true },
  { name: "Right pulmonary artery", points: [[0.16, 0.9, 0.06], [-0.15, 0.93, -0.24], [-0.48, 0.88, -0.3], [-0.72, 0.84, -0.3]], radius: 0.1, taper: 0.9, open: true },
  { name: "Superior vena cava", points: [[-0.48, 0.42, -0.04], [-0.5, 0.85, -0.06], [-0.5, 1.3, -0.06]], radius: 0.13, open: true },
  { name: "Inferior vena cava", points: [[-0.42, 0.05, -0.3], [-0.45, -0.45, -0.38], [-0.46, -0.95, -0.4]], radius: 0.13, open: true },
  { name: "Left upper pulmonary vein", points: [[0.45, 0.52, -0.38], [0.72, 0.6, -0.38], [0.98, 0.64, -0.36]], radius: 0.07, open: true },
  { name: "Left lower pulmonary vein", points: [[0.45, 0.34, -0.42], [0.72, 0.3, -0.44], [0.96, 0.26, -0.44]], radius: 0.068, open: true },
  { name: "Right pulmonary veins", points: [[0.05, 0.42, -0.5], [-0.35, 0.44, -0.58], [-0.62, 0.48, -0.6]], radius: 0.066, open: true },
];

export function makeVessel(v: GreatVessel) {
  const curve = new THREE.CatmullRomCurve3(v.points.map((p) => new THREE.Vector3(...p)));
  const tubular = 64;
  const radial = 28;
  const geo = new THREE.TubeGeometry(curve, tubular, v.radius, radial, false);
  if (v.taper && v.taper !== 1) {
    // Narrow the tube along its length, each ring toward its own centre.
    const pos = geo.attributes.position;
    const p = new THREE.Vector3();
    for (let i = 0; i <= tubular; i++) {
      const c = curve.getPointAt(i / tubular);
      const s = 1 + (v.taper - 1) * (i / tubular);
      for (let j = 0; j <= radial; j++) {
        const k = i * (radial + 1) + j;
        p.fromBufferAttribute(pos, k).sub(c).multiplyScalar(s).add(c);
        pos.setXYZ(k, p.x, p.y, p.z);
      }
    }
    geo.computeVertexNormals();
  }
  roughenSurface(geo, v.radius * 0.06, v.radius * 100);
  paintTissue(geo, VEINS.has(v.name) ? VEIN : ARTERY, v.points.length * 5 + v.radius * 50, () => 0.45);
  // A rim at the cut end gives the vessel wall visible thickness.
  const end = curve.getPointAt(1);
  const tangent = curve.getTangentAt(1);
  const rimRadius = v.radius * (v.taper ?? 1);
  const rim = { position: end, quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent), radius: rimRadius };
  return { name: v.name, geo, rim: v.open ? rim : null };
}

export const LOBES: { position: V3; rotation: V3; scale: V3; seed: number }[] = [
  { position: [-0.46, 0.3, 0.0], rotation: [0, 0, 0.2], scale: [0.3, 0.29, 0.31], seed: 1 }, // right atrium
  { position: [0.28, 0.42, -0.34], rotation: [0, 0, 0], scale: [0.42, 0.26, 0.3], seed: 2 }, // left atrium
];

// The auricles hang over the base like ear-shaped flaps either side of the great arteries.
export const AURICLES: { position: V3; rotation: V3; scale: V3; seed: number }[] = [
  { position: [-0.4, 0.4, 0.3], rotation: [0.6, -0.4, 0.7], scale: [0.34, 0.08, 0.22], seed: 3 }, // right auricle
  { position: [0.52, 0.34, 0.22], rotation: [0.4, 0.55, -0.9], scale: [0.34, 0.08, 0.2], seed: 4 }, // left auricle
];

export function chamberScale(volume: number, reference: number): number {
  return THREE.MathUtils.clamp(Math.pow(Math.max(volume, 0) / reference, 0.6), 0.5, 1.3);
}

export function RealisticHeart({ shadow = true }: { shadow?: boolean }) {
  const body = useMemo(() => makeBody(), []);
  const vessels = useMemo(() => GREAT_VESSELS.map(makeVessel), []);
  const lobes = useMemo(() => LOBES.map((l) => makeLobe(l.scale, 0.05, l.seed)), []);
  const flaps = useMemo(() => AURICLES.map((l) => makeFlap(l.scale, l.seed)), []);
  const coronaries = useMemo(() => makeCoronaries(), []);
  const bump = useMemo(() => createMuscleBumpTexture(), []);

  const ventricles = useRef<THREE.Group>(null);
  const atria = useRef<THREE.Group>(null);
  const roots = useRef<THREE.Group>(null);

  // Look inside: fade the wall and/or cut it open; the valves then show within.
  const wall = useHeartStore((s) => s.xray.wallOpacity);
  const cut = useHeartStore((s) => s.xray.cutDepth > 0);
  const lookingInside = wall < 0.97 || cut;
  const showValves = useHeartStore((s) => s.toggles.valves) && lookingInside;
  const showChambers = useHeartStore((s) => s.toggles.chambers) && lookingInside;
  const seeThrough = wall < 0.98;

  useFrame(() => {
    const out = liveOutputsRef.current;
    if (!out) return;
    const v = chamberScale(out.vLV + out.vRV, 350);
    const a = chamberScale(out.vLA + out.vRA, 105);
    if (ventricles.current) {
      // The ventricles shorten toward the base and twist slightly as they squeeze.
      const s = 1 + (v - 1) * 0.42;
      ventricles.current.scale.set(s, 1 + (v - 1) * 0.5, s);
      ventricles.current.rotation.y = (1 - v) * 0.12;
    }
    if (atria.current) atria.current.scale.setScalar(1 + (a - 1) * 0.3);
    // The great vessels swell a little as the ejected blood stretches them.
    if (roots.current) roots.current.scale.setScalar(1 + THREE.MathUtils.clamp((out.pAo - 80) / 40, 0, 1) * 0.015);
  });

  // Satin tissue: colour from the painted vertices, a soft clear coat on top.
  const skin = (
    <meshPhysicalMaterial
      vertexColors
      color="#ffffff"
      roughness={0.42}
      metalness={0}
      clearcoat={0.6}
      clearcoatRoughness={0.3}
      bumpMap={bump}
      bumpScale={0.004}
      transparent={seeThrough}
      opacity={wall}
      depthWrite={!seeThrough}
      clippingPlanes={SHELL_CUT}
    />
  );
  const rimSkin = (
    <meshPhysicalMaterial
      color="#e3998a"
      roughness={0.45}
      clearcoat={0.5}
      clearcoatRoughness={0.3}
      transparent={seeThrough}
      opacity={wall}
      clippingPlanes={SHELL_CUT}
    />
  );
  const vesselSkin = (color: string) => (
    <meshPhysicalMaterial
      color={color}
      roughness={0.38}
      clearcoat={0.7}
      clearcoatRoughness={0.25}
      transparent={seeThrough}
      opacity={wall}
      depthWrite={!seeThrough}
      clippingPlanes={SHELL_CUT}
    />
  );
  // The inner face of the wall, seen where the heart is cut open: a dark cavity.
  const cavity = cut && (
    <meshStandardMaterial color="#4a0b12" roughness={0.6} side={THREE.BackSide} transparent={seeThrough} opacity={wall} clippingPlanes={SHELL_CUT} />
  );

  return (
    <group position={[0, -0.12, 0]} scale={0.88}>
      {/* Pivot near the base, so contraction pulls the apex up toward it. */}
      <group position={[0, 0.35, 0]}>
        <group ref={ventricles}>
          <group position={[0, -0.35, 0]}>
            <mesh geometry={body} castShadow>
              {skin}
            </mesh>
            {cavity && <mesh geometry={body}>{cavity}</mesh>}
            <mesh geometry={coronaries.arteries}>{vesselSkin("#7d0711")}</mesh>
            <mesh geometry={coronaries.veins}>{vesselSkin("#c4bcc4")}</mesh>
          </group>
        </group>
      </group>
      <group ref={atria} position={[0, 0.4, -0.1]}>
        {LOBES.map((l, i) => (
          <mesh
            key={i}
            geometry={lobes[i]}
            position={[l.position[0], l.position[1] - 0.4, l.position[2] + 0.1]}
            rotation={l.rotation}
          >
            {skin}
          </mesh>
        ))}
        {cavity &&
          LOBES.map((l, i) => (
            <mesh key={`cavity-${i}`} geometry={lobes[i]} position={[l.position[0], l.position[1] - 0.4, l.position[2] + 0.1]} rotation={l.rotation}>
              {cavity}
            </mesh>
          ))}
        {AURICLES.map((l, i) => (
          <mesh
            key={`auricle-${i}`}
            geometry={flaps[i]}
            position={[l.position[0], l.position[1] - 0.4, l.position[2] + 0.1]}
            rotation={l.rotation}
          >
            {skin}
          </mesh>
        ))}
      </group>
      <group ref={roots}>
        {vessels.map((v) => (
          <group key={v.name}>
            <mesh geometry={v.geo}>{skin}</mesh>
            {/* Inside of the vessel, seen through its open end */}
            <mesh geometry={v.geo}>
              <meshStandardMaterial
                color={LUMEN}
                roughness={0.7}
                side={THREE.BackSide}
                transparent={seeThrough}
                opacity={wall}
                clippingPlanes={SHELL_CUT}
              />
            </mesh>
            {v.rim && (
              <mesh position={v.rim.position} quaternion={v.rim.quaternion}>
                <torusGeometry args={[v.rim.radius * 0.93, v.rim.radius * 0.1, 10, 36]} />
                {rimSkin}
              </mesh>
            )}
          </group>
        ))}
      </group>
      {showChambers && <ModelChambers style="tissue" />}
      {showValves && <ModelValves style="tissue" />}
      {shadow && <ContactShadows position={[0, -1.7, 0]} opacity={0.5} blur={2.6} scale={5} far={3.2} color="#000000" />}
    </group>
  );
}

/** Soft studio lighting for the realistic view, on a dark grey backdrop. */
export function StudioLights() {
  return (
    <>
      <hemisphereLight args={["#ffffff", "#5a5456", 1.0]} />
      <directionalLight position={[2.5, 3.5, 3]} intensity={2.4} color="#ffffff" />
      <directionalLight position={[-3, 1.5, 2]} intensity={1.0} color="#f4f0ee" />
      <directionalLight position={[0, 2, -3.5]} intensity={1.2} color="#ffffff" />
      <pointLight position={[0.6, 0.6, 2.6]} intensity={6} distance={8} decay={2} color="#ffffff" />
    </>
  );
}
