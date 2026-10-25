import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { CONDUCTION_PATH } from "./heartGeometry";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

const ACTIVATION_WINDOW = 0.06; // how long each node stays lit after activation, as a fraction of the cycle

export function ConductionSystem() {
  const toggles = useHeartStore((s) => s.toggles);
  const viewMode = useHeartStore((s) => s.viewMode);
  const nodeRefs = useRef<(THREE.Mesh | null)[]>([]);

  useFrame(() => {
    const out = liveOutputsRef.current;
    if (!out) return;
    for (let i = 0; i < CONDUCTION_PATH.length; i++) {
      const mesh = nodeRefs.current[i];
      if (!mesh) continue;
      const node = CONDUCTION_PATH[i];
      const delta = Math.min(Math.abs(out.tFrac - node.tAt), 1 - Math.abs(out.tFrac - node.tAt));
      const active = delta < ACTIVATION_WINDOW;
      const glow = active ? 1 - delta / ACTIVATION_WINDOW : 0;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.3 + glow * 3.2;
      mesh.scale.setScalar(1 + glow * 0.9);
    }
  });

  if (!(toggles.conduction || viewMode === "electrical")) return null;

  return (
    <group>
      {CONDUCTION_PATH.map((node, i) => (
        <mesh
          key={node.name}
          position={node.position}
          ref={(m) => {
            nodeRefs.current[i] = m;
          }}
        >
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color="#ffe066" emissive="#ffcc33" emissiveIntensity={0.3} toneMapped={false} />
        </mesh>
      ))}
      {CONDUCTION_PATH.slice(1).map((node, i) => (
        <line key={`link-${node.name}`}>
          <bufferGeometry
            attach="geometry"
            onUpdate={(g) => {
              const prev = CONDUCTION_PATH[i];
              g.setFromPoints([new THREE.Vector3(...prev.position), new THREE.Vector3(...node.position)]);
            }}
          />
          <lineBasicMaterial color="#8a7a2a" transparent opacity={0.5} />
        </line>
      ))}
    </group>
  );
}
