import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { makeLeafletGeometry, VALVES } from "./heartGeometry";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

const MAX_OPEN_ANGLE = 1.25; // radians, ~72 degrees

export function ValveIndicators() {
  const toggles = useHeartStore((s) => s.toggles);
  const viewMode = useHeartStore((s) => s.viewMode);
  // leafletRefs[valveIndex][leafletIndex]
  const leafletRefs = useRef<(THREE.Mesh | null)[][]>(VALVES.map((v) => Array(v.leaflets).fill(null)));
  const matRefs = useRef<(THREE.MeshStandardMaterial | null)[][]>(VALVES.map((v) => Array(v.leaflets).fill(null)));

  const leafletGeo = useMemo(() => makeLeafletGeometry(), []);

  useFrame(() => {
    const out = liveOutputsRef.current;
    if (!out) return;
    VALVES.forEach((v, vi) => {
      const openness = out.valves[v.name];
      const angle = -openness * MAX_OPEN_ANGLE;
      for (let li = 0; li < v.leaflets; li++) {
        const mesh = leafletRefs.current[vi][li];
        if (!mesh) continue;
        mesh.rotation.x = angle;
        const mat = matRefs.current[vi][li];
        if (mat) {
          mat.emissiveIntensity = openness > 0.05 ? 0.6 + openness * 1.3 : 0.15;
          mat.color.set(openness > 0.05 ? "#fff7e0" : "#f2e6c9");
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
