import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { makeLeafletGeometry, VALVES } from "./heartGeometry";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

const MAX_OPEN_ANGLE = 1.25; // radians, ~72 degrees
const LEAFLET = new THREE.Color("#f2e6c9");
const LEAFLET_OPEN = new THREE.Color("#fff7e0");
const CALCIFIED = new THREE.Color("#a89466");

export function ValveIndicators() {
  const toggles = useHeartStore((s) => s.toggles);
  const viewMode = useHeartStore((s) => s.viewMode);
  // leafletRefs[valveIndex][leafletIndex]
  const leafletRefs = useRef<(THREE.Mesh | null)[][]>(VALVES.map((v) => Array(v.leaflets).fill(null)));
  const matRefs = useRef<(THREE.MeshStandardMaterial | null)[][]>(VALVES.map((v) => Array(v.leaflets).fill(null)));

  const leafletGeo = useMemo(() => makeLeafletGeometry(), []);
  const shown = useRef<number[]>(VALVES.map(() => 0));

  useFrame((_, delta) => {
    const out = liveOutputsRef.current;
    if (!out) return;
    const { aorticStenosis, mitralRegurgitation } = useHeartStore.getState().valves;
    VALVES.forEach((v, vi) => {
      // Leaflets swing open and shut over a few tens of milliseconds, as real
      // ones do, instead of snapping between states each frame.
      shown.current[vi] += (out.valves[v.name] - shown.current[vi]) * Math.min(1, delta * 22);
      let openness = shown.current[vi];
      // A leaky mitral valve never seals: its leaflets stay slightly apart.
      if (v.name === "mitral") openness = Math.max(openness, mitralRegurgitation * 0.22);
      const calcified = v.name === "aortic" ? aorticStenosis : 0;
      const angle = -openness * MAX_OPEN_ANGLE;
      for (let li = 0; li < v.leaflets; li++) {
        const mesh = leafletRefs.current[vi][li];
        if (!mesh) continue;
        mesh.rotation.x = angle;
        const mat = matRefs.current[vi][li];
        if (mat) {
          mat.emissiveIntensity = openness > 0.05 ? 0.6 + openness * 1.3 : 0.15;
          // A narrowed (stenotic) aortic valve is stiff and calcified: duller and yellow-grey.
          mat.color.copy(openness > 0.05 ? LEAFLET_OPEN : LEAFLET).lerp(CALCIFIED, calcified * 0.85);
          mat.emissiveIntensity *= 1 - calcified * 0.6;
        }
      }
    });
  });

  if (!toggles.valves) return null;
  const emphasized = viewMode === "valves";

  return (
    <group>
      {VALVES.map((v, vi) => (
        <group key={v.name} position={v.position} rotation={v.rotation} scale={v.scale * (emphasized ? 2.1 : 1)}>
          {/* Fixed fibrous annulus the leaflets are mounted in */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.074, 0.009, 8, 20]} />
            <meshStandardMaterial color="#4a3a3a" roughness={0.8} />
          </mesh>
          {Array.from({ length: v.leaflets }).map((_, li) => (
            <group key={li} rotation={[0, 0, (li * (2 * Math.PI)) / v.leaflets]}>
              <mesh
                geometry={leafletGeo}
                ref={(m) => {
                  leafletRefs.current[vi][li] = m;
                }}
              >
                <meshStandardMaterial
                  ref={(m) => {
                    matRefs.current[vi][li] = m;
                  }}
                  color="#f2e6c9"
                  emissive="#8fe3ff"
                  emissiveIntensity={0.15}
                  roughness={0.4}
                  metalness={0.05}
                  side={THREE.DoubleSide}
                  transparent
                  opacity={0.92}
                  toneMapped={false}
                />
              </mesh>
            </group>
          ))}
        </group>
      ))}
    </group>
  );
}

const JET_COUNT = 36;
const JET_FROM = new THREE.Vector3(0.39, -0.05, -0.04); // inside the LV, just under the mitral valve
const JET_TO = new THREE.Vector3(0.5, 0.55, -0.05); // into the left atrium

/**
 * Mitral regurgitation made visible: while the left pump squeezes, the
 * simulated backward leak (regurgFlow, mL/s) is drawn as a jet squirting
 * through the mitral valve into the left atrium. More leak = denser, faster jet.
 */
export function RegurgitationJet() {
  const toggles = useHeartStore((s) => s.toggles);
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const t = useRef(Float32Array.from({ length: JET_COUNT }, (_, i) => i / JET_COUNT));
  const offsets = useMemo(
    () => Array.from({ length: JET_COUNT }, (_, i) => new THREE.Vector3(Math.sin(i * 2.4) * 0.035, 0, Math.cos(i * 2.4) * 0.035)),
    []
  );
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const strength = useRef(0);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    const out = liveOutputsRef.current;
    if (!mesh || !out) return;
    const target = THREE.MathUtils.clamp(out.regurgFlow / 60, 0, 1);
    strength.current += (target - strength.current) * Math.min(1, delta * 12);
    for (let i = 0; i < JET_COUNT; i++) {
      t.current[i] = (t.current[i] + delta * (0.6 + 2.2 * strength.current)) % 1;
      const k = t.current[i];
      dummy.position.lerpVectors(JET_FROM, JET_TO, k).addScaledVector(offsets[i], k * 1.6);
      // Each particle shows only while the leak is flowing; the jet fans out as it enters the atrium.
      dummy.scale.setScalar(strength.current > 0.03 ? 0.022 + 0.02 * k * strength.current : 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  if (!toggles.valves) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, JET_COUNT]} renderOrder={3}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color="#ff8a3d" toneMapped={false} transparent opacity={0.9} depthTest={false} />
    </instancedMesh>
  );
}
