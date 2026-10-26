import { Suspense, useRef, type ReactNode } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { SimulationDriver } from "./SimulationDriver";
import { HeartChambers } from "./HeartChambers";
import { Vessels } from "./Vessels";
import { BloodFlowParticles } from "./BloodFlowParticles";
import { ConductionSystem } from "./ConductionSystem";
import { RegurgitationJet, ValveIndicators } from "./ValveIndicators";
import { CHAMBERS } from "./heartGeometry";
import { showsInterior, useHeartStore } from "../../state/store";
import { CutPlane } from "./CutPlane";
import { RiskLegend } from "./RiskLegend";
import { RealisticHeart, StudioLights } from "./RealisticHeart";
import { DigitalHeart, DigitalLights } from "./DigitalHeart";
import { HoloHud } from "./HoloHud";
import { DIGITAL } from "../../variant";

/** Turns its contents round on the spot (for the side-by-side view, where each heart spins in place). */
function Spinner({ on, children }: { on: boolean; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (on && ref.current) ref.current.rotation.y += delta * 0.5;
  });
  return <group ref={ref}>{children}</group>;
}

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
  const viewMode = useHeartStore((s) => s.viewMode);
  const realistic = viewMode === "realistic";
  const digital = viewMode === "hologram" || viewMode === "lowpoly" || viewMode === "dual";
  const autoRotate = useHeartStore((s) => s.autoRotate);
  const setAutoRotate = useHeartStore((s) => s.setAutoRotate);
  const backdrop = digital || (DIGITAL && realistic) ? "holo" : realistic ? "studio" : "";
  return (
    <div className={`heart3d-canvas ${backdrop}`}>
      <Canvas
        camera={{ position: [0, 0.3, 4.2], fov: 42 }}
        gl={{ antialias: true, localClippingEnabled: true, alpha: true }}
        dpr={[1, 1.75]}
      >
        {digital ? (
          viewMode === "dual" ? <StudioLights /> : <DigitalLights />
        ) : realistic ? (
          <StudioLights />
        ) : (
          <>
            <color attach="background" args={["#050a14"]} />
            <ambientLight intensity={1.0} />
            <directionalLight position={[3, 4, 4]} intensity={1.5} />
            <directionalLight position={[-3, 1, 2]} intensity={0.7} />
            <directionalLight position={[-3, -2, -3]} intensity={0.4} color="#6699ff" />
            {/* Front fill so the cutaway interior (cavities, valves, chordae) is lit */}
            <pointLight position={[0, 0.2, 2.6]} intensity={14} distance={9} decay={2} color="#ffe8e0" />
          </>
        )}
        <Suspense fallback={null}>
          <SimulationDriver />
          {viewMode === "dual" ? (
            // The red heart and its blue digital twin side by side, each turning in place.
            <>
              <group position={[-1.1, 0.12, 0]} scale={0.66}>
                <Spinner on={autoRotate}>
                  <RealisticHeart shadow={false} />
                </Spinner>
              </group>
              <group position={[1.1, 0.12, 0]} scale={0.66}>
                <Spinner on={autoRotate}>
                  <DigitalHeart look="hologram" />
                </Spinner>
              </group>
            </>
          ) : digital ? (
            <DigitalHeart look={viewMode === "lowpoly" ? "lowpoly" : "hologram"} />
          ) : realistic ? (
            <RealisticHeart />
          ) : (
            <>
              <CutPlane />
              <group rotation={[0, 0, 0]}>
                <HeartChambers />
                <Vessels />
                <BloodFlowParticles />
                <ConductionSystem />
                <ValveIndicators />
                <RegurgitationJet />
                <ChamberLabels />
              </group>
            </>
          )}
        </Suspense>
        {/* Full 360-degree orbit by dragging, plus an optional turntable */}
        <OrbitControls
          enablePan
          enableDamping
          minDistance={1.8}
          maxDistance={9}
          target={[0, 0, 0]}
          autoRotate={autoRotate && viewMode !== "dual"}
          autoRotateSpeed={2.4}
          makeDefault
        />
      </Canvas>
      <button
        type="button"
        className={`rotate-toggle ${autoRotate ? "on" : ""}`}
        aria-pressed={autoRotate}
        title="Turn the heart round continuously; drag to turn it yourself"
        onClick={() => setAutoRotate(!autoRotate)}
      >
        <span aria-hidden="true">&#10227;</span> {autoRotate ? "Rotating 360°" : "Rotate 360°"}
      </button>
      <RiskLegend />
      {DIGITAL && <HoloHud />}
    </div>
  );
}

export function disposeMaterial(mat: THREE.Material | THREE.Material[]) {
  if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
  else mat.dispose();
}
