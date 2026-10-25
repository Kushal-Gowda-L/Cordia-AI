import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { Vec3 } from "./heartGeometry";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

// The full circulation as one closed loop, following the vessel and
// chamber layout in heartGeometry.ts. Indices are referenced by ZONES below.
const LOOP_POINTS: Vec3[] = [
  [-0.6, -0.95, -0.3], // 0  inferior vena cava
  [-0.64, -0.35, -0.36], // 1
  [-0.58, 0.22, -0.16], // 2  entering the right atrium
  [-0.52, 0.46, 0.0], // 3  right atrium
  [-0.45, 0.14, -0.02], // 4  tricuspid valve
  [-0.36, -0.25, 0.0], // 5  right ventricle
  [-0.27, 0.2, 0.1], // 6  pulmonary valve
  [-0.2, 0.5, 0.2], // 7  pulmonary trunk, crossing in front of the aorta
  [0.0, 0.75, 0.16], // 8
  [0.18, 0.86, 0.04], // 9  bifurcation
  [0.45, 0.9, -0.04], // 10 left pulmonary artery
  [0.8, 0.86, -0.14], // 11 -> lungs
  [1.12, 0.72, -0.22], // 12 (implied) lungs: gas exchange
  [0.95, 0.66, -0.26], // 13 pulmonary veins
  [0.72, 0.6, -0.18], // 14
  [0.48, 0.5, -0.02], // 15 left atrium
  [0.41, 0.15, -0.05], // 16 mitral valve
  [0.34, -0.36, -0.02], // 17 left ventricle
  [0.19, 0.2, -0.06], // 18 aortic valve
  [0.1, 0.55, -0.02], // 19 ascending aorta
  [0.04, 0.95, -0.06], // 20
  [0.12, 1.14, -0.2], // 21 aortic arch
  [0.34, 1.1, -0.36], // 22
  [0.46, 0.8, -0.54], // 23 descending aorta
  [0.46, 0.2, -0.56], // 24
  [0.42, -0.6, -0.54], // 25 -> body
  [0.0, -1.15, -0.45], // 26 (implied) systemic tissues: oxygen extracted
];

type FlowKey = "tricuspid" | "pulmonic" | "mitral" | "aortic" | "constant";

// Which live valve flow drives particle speed on each stretch of the loop
// (from waypoint index `from` up to the next zone's `from`).
const ZONES: { from: number; flow: FlowKey }[] = [
  { from: 0, flow: "tricuspid" }, // venous return through RA and tricuspid
  { from: 6, flow: "pulmonic" }, // RV ejection into the pulmonary arteries
  { from: 11, flow: "constant" }, // lungs
  { from: 13, flow: "mitral" }, // pulmonary veins, LA, mitral filling
  { from: 18, flow: "aortic" }, // LV ejection into the aorta
  { from: 23, flow: "constant" }, // systemic circulation
];

// Blood turns red crossing the lungs and blue crossing the body -- the only
// two places real blood changes oxygenation.
const LUNGS = [11, 13];
const BODY = [25, LOOP_POINTS.length];

// Deoxygenated vs oxygenated. Kept deliberately far apart in hue and
// matched to the dashboard legend, so which blood is which is readable at
// a glance anywhere in the circuit.
const BLUE = new THREE.Color("#2e9bff");
const RED = new THREE.Color("#ff2e3c");
const PARTICLE_COUNT = 150;

export function BloodFlowParticles() {
  const toggles = useHeartStore((s) => s.toggles);
  const viewMode = useHeartStore((s) => s.viewMode);
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const { curve, zoneStarts, lungs, body } = useMemo(() => {
    const c = new THREE.CatmullRomCurve3(
      LOOP_POINTS.map((p) => new THREE.Vector3(...p)),
      true
    );
    // Convert waypoint indices to arc-length fractions, since particles
    // advance along the curve by arc length (getPointAt), not by index.
    const divisions = 2000;
    const lengths = c.getLengths(divisions);
    const total = lengths[divisions];
    const u = (i: number) =>
      i >= LOOP_POINTS.length ? 1 : lengths[Math.round((i / LOOP_POINTS.length) * divisions)] / total;
    return {
      curve: c,
      zoneStarts: ZONES.map((z) => ({ start: u(z.from), flow: z.flow })),
      lungs: [u(LUNGS[0]), u(LUNGS[1])],
      body: [u(BODY[0]), u(BODY[1])],
    };
  }, []);

  const particleT = useRef<Float32Array>(
    (() => {
      const arr = new Float32Array(PARTICLE_COUNT);
      for (let i = 0; i < PARTICLE_COUNT; i++) arr[i] = i / PARTICLE_COUNT;
      return arr;
    })()
  );

  const dummy = useMemo(() => new THREE.Object3D(), []);

  // three.js only multiplies the per-instance colour into the fragment
  // when USE_COLOR is defined (i.e. material.vertexColors), but that same
  // flag makes the shader read a geometry `color` attribute -- absent on a
  // plain sphere, WebGL then supplies black and zeroes the instance colour.
  // Supplying an all-white color attribute satisfies both requirements.
  const particleGeometry = useMemo(() => {
    const g = new THREE.SphereGeometry(1, 10, 10);
    const white = new Float32Array(g.attributes.position.count * 3).fill(1);
    g.setAttribute("color", new THREE.BufferAttribute(white, 3));
    return g;
  }, []);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const out = liveOutputsRef.current;
    if (!out) return;

    const flowScale: Record<FlowKey, number> = {
      tricuspid: THREE.MathUtils.clamp(out.flowSystemicIn / 40, 0.15, 2.2),
      pulmonic: THREE.MathUtils.clamp(out.flowPulmonaryOut / 40, 0.15, 2.2),
      mitral: THREE.MathUtils.clamp(out.flowPulmonaryIn / 40, 0.15, 2.2),
      aortic: THREE.MathUtils.clamp(out.flowSystemicOut / 40, 0.15, 2.2),
      constant: 0.7,
    };

    const baseSpeed = 0.06 * delta * 60;
    const emphasis = viewMode === "bloodFlow" ? 1.3 : 1;
    const size = viewMode === "bloodFlow" ? 0.045 : 0.032;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      let t = particleT.current[i];
      let flow: FlowKey = zoneStarts[0].flow;
      for (const z of zoneStarts) if (t >= z.start) flow = z.flow;
      t += flowScale[flow] * baseSpeed * emphasis * 0.02;
      if (t > 1) t -= 1;
      particleT.current[i] = t;

      dummy.position.copy(curve.getPointAt(t));
      dummy.scale.setScalar(size);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      let oxygenation: number;
      if (t < lungs[0]) oxygenation = 0;
      else if (t < lungs[1]) oxygenation = (t - lungs[0]) / (lungs[1] - lungs[0]);
      else if (t < body[0]) oxygenation = 1;
      else oxygenation = 1 - (t - body[0]) / (body[1] - body[0]);

      // setColorAt must be called unconditionally: three.js lazily creates
      // mesh.instanceColor on the first call, so gating it on that
      // attribute already existing means it never gets created.
      mesh.setColorAt(i, BLUE.clone().lerp(RED, oxygenation));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  if (!toggles.bloodFlow) return null;

  return (
    <instancedMesh ref={meshRef} args={[particleGeometry, undefined, PARTICLE_COUNT]}>
      {/* Unlit so each particle renders its assigned oxygenation colour at
          full saturation against the scene's bright fill lighting. */}
      <meshBasicMaterial vertexColors toneMapped={false} />
    </instancedMesh>
  );
}
