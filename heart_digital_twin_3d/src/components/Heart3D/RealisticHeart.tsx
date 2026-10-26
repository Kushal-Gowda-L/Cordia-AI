import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { createMuscleBumpTexture } from "./muscleTexture";

/**
 * The "Realistic heart" view: the closed heart as an anatomical sculpture
 * (the look of a studio render of an anatomical model), in one soft coral
 * tone -- a tapered ventricular body with the coronary arteries and veins
 * raised on its surface, both atria and their auricles, and the great
 * vessels with open, hollow ends: the aortic arch with its three branches,
 * the pulmonary trunk and arteries, the venae cavae and pulmonary veins.
 *
 * It is still the live twin: the ventricles squeeze with the simulated
 * ventricular volume and the atria with the atrial volume, so heart rate,
 * pumping strength and every data change show here too.
 *
 * Orientation follows the anterior view: the heart's left side (LV, aortic
 * arch, left pulmonary veins) is on the viewer's right.
 */

const SKIN = "#f2a596";
const LUMEN = "#8f3a38";
const R = 0.95;

type V3 = [number, number, number];

const dirOf = (polar: number, azimuth: number) =>
  new THREE.Vector3(Math.sin(polar) * Math.cos(azimuth), Math.cos(polar), Math.sin(polar) * Math.sin(azimuth));

// Tilt so the apex points down and to the viewer's right, as in the chest.
const BODY_TILT = new THREE.Euler(0, -0.25, 0.35);
const BODY_SHIFT = new THREE.Vector3(0.02, -0.2, 0);

/** A unit direction -> a point on the ventricular surface (broad base, tapered apex, LV side fuller). */
function bodyPoint(d: THREE.Vector3): THREE.Vector3 {
  const y = d.y;
  const taper = y < 0 ? 1 - 0.6 * Math.pow(-y, 1.25) : 1 - 0.08 * y * y;
  let x = d.x * R * taper * (d.x > 0 ? 1.08 : 0.96);
  const z = d.z * R * 0.78 * taper * (d.z > 0 ? 1.05 : 1);
  // Short, rounded base above the equator, long tapering apex below, blended smoothly (no crease).
  const k = 0.5 * (1 + Math.tanh(4 * y));
  const yy = y * R * (1.5 * (1 - k) + 0.72 * k);
  // Gentle organic unevenness so it doesn't read as a smooth egg.
  const n = 0.018 * Math.sin(3.1 * d.x + 1.7 * y) * Math.cos(2.3 * d.z + 0.5) + 0.008 * Math.sin(7 * d.x - 5 * y + 3 * d.z);
  x *= 1 + n;
  return new THREE.Vector3(x, yy * (1 + n), z * (1 + n)).applyEuler(BODY_TILT).add(BODY_SHIFT);
}

// Surface vessels in [polar, azimuth] over the body (azimuth PI/2 faces the viewer).
const SURFACE_VESSELS: { path: [number, number][]; width: number }[] = [
  { path: [[1.1, 1.45], [1.5, 1.5], [1.9, 1.45], [2.3, 1.3], [2.75, 1.0], [3.02, 0.55]], width: 0.026 }, // LAD
  { path: [[1.5, 1.47], [1.8, 1.17], [2.1, 0.97], [2.45, 0.8]], width: 0.018 }, // first diagonal
  { path: [[1.95, 1.43], [2.3, 1.18], [2.62, 0.98]], width: 0.015 }, // second diagonal
  { path: [[1.22, 1.68], [1.65, 1.72], [2.15, 1.66], [2.55, 1.5]], width: 0.016 }, // great cardiac vein
  { path: [[1.28, 1.95], [1.4, 2.4], [1.46, 2.9], [1.52, 3.4]], width: 0.024 }, // right coronary
  { path: [[1.43, 2.6], [1.9, 2.55], [2.35, 2.35], [2.6, 2.1]], width: 0.016 }, // acute marginal
  { path: [[1.16, 1.2], [1.27, 0.7], [1.33, 0.2], [1.38, -0.3]], width: 0.022 }, // circumflex
  { path: [[1.3, 0.5], [1.8, 0.38], [2.25, 0.32], [2.55, 0.4]], width: 0.016 }, // obtuse marginal
];

function makeBody(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 160, 120);
  const pos = geo.attributes.position;
  const d = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    d.fromBufferAttribute(pos, i).normalize();
    const p = bodyPoint(d);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  geo.computeVertexNormals();

  // Raise the coronary vessels as ridges, each in a shallow groove, like the
  // vessels and their fat-filled sulci on a real heart's surface.
  const samples: { p: THREE.Vector3; w: number }[] = [];
  for (const v of SURFACE_VESSELS) {
    const curve = new THREE.CatmullRomCurve3(v.path.map(([pol, az]) => dirOf(pol, az)));
    for (const q of curve.getPoints(90)) samples.push({ p: bodyPoint(q.normalize()), w: v.width });
  }
  const normal = geo.attributes.normal;
  const p = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    let lift = 0;
    for (const s of samples) {
      const dist2 = p.distanceToSquared(s.p);
      if (dist2 > 0.04) continue;
      const k = dist2 / (s.w * s.w);
      lift = Math.max(lift, 0.9 * s.w * Math.exp(-k) - 0.25 * s.w * Math.exp(-k / 9));
    }
    if (lift !== 0) {
      nrm.fromBufferAttribute(normal, i);
      p.addScaledVector(nrm, lift);
      pos.setXYZ(i, p.x, p.y, p.z);
    }
  }
  geo.computeVertexNormals();
  return geo;
}

/** A soft lobe (atrium or auricle): a squashed, slightly lumpy sphere. */
function makeLobe(scale: V3, lumps = 0.06, seed = 1): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, 48, 36);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = 1 + lumps * Math.sin(5 * v.x + seed) * Math.sin(4 * v.y + 2 * seed) * Math.cos(6 * v.z);
    pos.setXYZ(i, v.x * scale[0] * n, v.y * scale[1] * n, v.z * scale[2] * n);
  }
  geo.computeVertexNormals();
  return geo;
}

interface GreatVessel {
  name: string;
  points: V3[];
  radius: number;
  /** End radius as a fraction of the start, for tapering branches. */
  taper?: number;
  open: boolean; // last point is an open, cut end
}

const GREAT_VESSELS: GreatVessel[] = [
  // The aorta rises behind the pulmonary trunk and arches back and to the heart's left.
  { name: "Aorta", points: [[0.02, 0.2, 0.0], [-0.05, 0.6, 0.06], [-0.04, 0.98, 0.0], [0.12, 1.16, -0.2], [0.38, 1.08, -0.42], [0.5, 0.75, -0.55], [0.52, 0.2, -0.6], [0.5, -0.45, -0.6]], radius: 0.17, open: true },
  { name: "Brachiocephalic trunk", points: [[-0.04, 1.08, -0.08], [-0.1, 1.35, -0.08], [-0.16, 1.58, -0.08]], radius: 0.075, taper: 0.85, open: true },
  { name: "Left common carotid", points: [[0.14, 1.17, -0.22], [0.15, 1.4, -0.22], [0.14, 1.62, -0.2]], radius: 0.058, taper: 0.9, open: true },
  { name: "Left subclavian", points: [[0.3, 1.12, -0.33], [0.36, 1.35, -0.34], [0.44, 1.55, -0.36]], radius: 0.06, taper: 0.9, open: true },
  // Pulmonary trunk leaves the front of the right ventricle, crossing in front of the aorta.
  { name: "Pulmonary trunk", points: [[-0.12, 0.25, 0.36], [-0.05, 0.55, 0.36], [0.08, 0.8, 0.22], [0.16, 0.9, 0.06]], radius: 0.15, open: false },
  { name: "Left pulmonary artery", points: [[0.16, 0.9, 0.06], [0.42, 0.98, 0.0], [0.72, 0.95, -0.06]], radius: 0.1, taper: 0.9, open: true },
  { name: "Right pulmonary artery", points: [[0.16, 0.9, 0.06], [-0.15, 0.93, -0.24], [-0.48, 0.88, -0.3], [-0.72, 0.84, -0.3]], radius: 0.1, taper: 0.9, open: true },
  { name: "Superior vena cava", points: [[-0.48, 0.42, -0.04], [-0.5, 0.85, -0.06], [-0.5, 1.3, -0.06]], radius: 0.13, open: true },
  { name: "Inferior vena cava", points: [[-0.42, 0.05, -0.3], [-0.45, -0.45, -0.38], [-0.46, -0.95, -0.4]], radius: 0.13, open: true },
  { name: "Left upper pulmonary vein", points: [[0.45, 0.52, -0.38], [0.72, 0.6, -0.38], [0.98, 0.64, -0.36]], radius: 0.07, open: true },
  { name: "Left lower pulmonary vein", points: [[0.45, 0.34, -0.42], [0.72, 0.3, -0.44], [0.96, 0.26, -0.44]], radius: 0.068, open: true },
  { name: "Right pulmonary veins", points: [[0.05, 0.42, -0.5], [-0.35, 0.44, -0.58], [-0.62, 0.48, -0.6]], radius: 0.066, open: true },
];

function makeVessel(v: GreatVessel) {
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
  // A rim at the cut end gives the vessel wall visible thickness.
  const end = curve.getPointAt(1);
  const tangent = curve.getTangentAt(1);
  const rimRadius = v.radius * (v.taper ?? 1);
  const rim = { position: end, quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent), radius: rimRadius };
  return { name: v.name, geo, rim: v.open ? rim : null };
}

const LOBES: { position: V3; rotation: V3; scale: V3; atrium: boolean; seed: number }[] = [
  { position: [-0.44, 0.3, 0.0], rotation: [0, 0, 0.2], scale: [0.3, 0.28, 0.3], atrium: true, seed: 1 }, // right atrium
  { position: [0.28, 0.42, -0.34], rotation: [0, 0, 0], scale: [0.42, 0.26, 0.3], atrium: true, seed: 2 }, // left atrium
  { position: [-0.3, 0.44, 0.24], rotation: [0.3, -0.4, 0.55], scale: [0.18, 0.08, 0.11], atrium: true, seed: 3 }, // right auricle
  { position: [0.42, 0.42, 0.2], rotation: [0.2, 0.4, -0.7], scale: [0.17, 0.08, 0.1], atrium: true, seed: 4 }, // left auricle
];

function chamberScale(volume: number, reference: number): number {
  return THREE.MathUtils.clamp(Math.pow(Math.max(volume, 0) / reference, 0.6), 0.5, 1.3);
}

export function RealisticHeart() {
  const body = useMemo(() => makeBody(), []);
  const vessels = useMemo(() => GREAT_VESSELS.map(makeVessel), []);
  const lobes = useMemo(() => LOBES.map((l) => makeLobe(l.scale, 0.05, l.seed)), []);
  const bump = useMemo(() => createMuscleBumpTexture(), []);

  const ventricles = useRef<THREE.Group>(null);
  const atria = useRef<THREE.Group>(null);
  const roots = useRef<THREE.Group>(null);

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

  const skin = (
    <meshPhysicalMaterial
      color={SKIN}
      roughness={0.42}
      metalness={0}
      clearcoat={0.55}
      clearcoatRoughness={0.32}
      sheen={0.35}
      sheenColor="#ffd6cf"
      bumpMap={bump}
      bumpScale={0.004}
    />
  );

  return (
    <group position={[0, -0.12, 0]} scale={0.88}>
      {/* Pivot near the base, so contraction pulls the apex up toward it. */}
      <group position={[0, 0.35, 0]}>
        <group ref={ventricles}>
          <mesh geometry={body} position={[0, -0.35, 0]} castShadow>
            {skin}
          </mesh>
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
      </group>
      <group ref={roots}>
        {vessels.map((v) => (
          <group key={v.name}>
            <mesh geometry={v.geo}>{skin}</mesh>
            {/* Inside of the vessel, seen through its open end */}
            <mesh geometry={v.geo}>
              <meshStandardMaterial color={LUMEN} roughness={0.7} side={THREE.BackSide} />
            </mesh>
            {v.rim && (
              <mesh position={v.rim.position} quaternion={v.rim.quaternion}>
                <torusGeometry args={[v.rim.radius * 0.93, v.rim.radius * 0.1, 10, 36]} />
                {skin}
              </mesh>
            )}
          </group>
        ))}
      </group>
      <ContactShadows position={[0, -1.7, 0]} opacity={0.32} blur={2.6} scale={5} far={3.2} color="#6e3d44" />
    </group>
  );
}

/** Soft studio lighting for the realistic view. */
export function StudioLights() {
  return (
    <>
      <hemisphereLight args={["#fff2ef", "#c49297", 1.15]} />
      <directionalLight position={[2.5, 3.5, 3]} intensity={1.7} color="#fff4ee" />
      <directionalLight position={[-3, 1.2, 1.5]} intensity={0.55} color="#ffd9d6" />
      <directionalLight position={[0, 2, -3.5]} intensity={0.9} color="#ffe6e2" />
      <pointLight position={[0, -0.4, 2.8]} intensity={4} distance={8} decay={2} color="#ffd8d0" />
    </>
  );
}
