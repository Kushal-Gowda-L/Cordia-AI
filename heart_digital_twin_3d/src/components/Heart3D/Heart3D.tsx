import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { SimulationDriver } from "./SimulationDriver";
import { HeartChambers } from "./HeartChambers";
import { Vessels } from "./Vessels";
import { BloodFlowParticles } from "./BloodFlowParticles";
import { ConductionSystem } from "./ConductionSystem";
import { ValveIndicators } from "./ValveIndicators";
import { CHAMBERS } from "./heartGeometry";
import { showsInterior, useHeartStore } from "../../state/store";
import { CutPlane } from "./CutPlane";
import { RiskLegend } from "./RiskLegend";

function ChamberLabels() {
  const toggles = useHeartStore((s) => s.toggles);
  const interior = useHeartStore((s) => showsInterior(s));
  if (!toggles.labels || !interior) return null;
  return (
    <>
      {CHAMBERS.map((c) => (
        <Html key={c.name} position={c.center} center distanceFactor={6} occlude={false}>
          <div className="chamber-label">{c.name}</div>
        </Html>
      ))}
    </>
  );
}

export function Heart3D() {
  return (
    <div className="heart3d-canvas">
      <Canvas
        camera={{ position: [0, 0.3, 4.2], fov: 42 }}
        gl={{ antialias: true, localClippingEnabled: true }}
        dpr={[1, 1.75]}
      >
        <color attach="background" args={["#050a14"]} />
        <ambientLight intensity={1.0} />
        <directionalLight position={[3, 4, 4]} intensity={1.5} />
        <directionalLight position={[-3, 1, 2]} intensity={0.7} />
        <directionalLight position={[-3, -2, -3]} intensity={0.4} color="#6699ff" />
        {/* Front fill so the cutaway interior (cavities, valves, chordae) is lit */}
        <pointLight position={[0, 0.2, 2.6]} intensity={14} distance={9} decay={2} color="#ffe8e0" />
        <Suspense fallback={null}>
          <SimulationDriver />
          <CutPlane />
          <group rotation={[0, 0, 0]}>
            <HeartChambers />
            <Vessels />
            <BloodFlowParticles />
            <ConductionSystem />
            <ValveIndicators />
            <ChamberLabels />
          </group>
        </Suspense>
        <OrbitControls enablePan minDistance={1.8} maxDistance={9} target={[0, 0, 0]} makeDefault />
      </Canvas>
      <RiskLegend />
    </div>
  );
}

export function disposeMaterial(mat: THREE.Material | THREE.Material[]) {
  if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
  else mat.dispose();
}
