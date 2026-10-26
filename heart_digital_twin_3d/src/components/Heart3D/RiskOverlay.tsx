import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { CORONARIES, ladTerritoryWeight, myocardiumSurfacePoint } from "./heartGeometry";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

const ARTERY_COLOR = new THREE.Color("#c2412f");
const LOW = new THREE.Color("#2e9e5b");
const MID = new THREE.Color("#e0a020");
const HIGH = new THREE.Color("#e2302a");

/** Green -> amber -> red across the app's 35% / 65% display zones (src/risk_reading.py). */
export function riskColor(diseasePercent: number, target = new THREE.Color()): THREE.Color {
  const p = THREE.MathUtils.clamp(diseasePercent, 0, 100);
  if (p <= 50) return target.copy(LOW).lerp(MID, THREE.MathUtils.smoothstep(p, 20, 50));
  return target.copy(MID).lerp(HIGH, THREE.MathUtils.smoothstep(p, 50, 80));
}

function diseasePercent(): number {
  return useHeartStore.getState().prediction?.reading.disease_percent ?? 0;
}

/**
 * Coronary arteries on the outer heart surface. In the risk view they take
 * the risk colour and pulse more strongly the higher the predicted risk.
 */
export function CoronaryArteries() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const matRefs = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const tubes = useMemo(
    () =>
      CORONARIES.map((c) => {
        const curve = new THREE.CatmullRomCurve3(c.path.map(([pol, az]) => myocardiumSurfacePoint(pol, az)));
        return { ...c, geo: new THREE.TubeGeometry(curve, 48, c.radius, 8, false) };
      }),
    []
  );
  const color = useRef(new THREE.Color());

  useFrame(() => {
    const risk = viewMode === "risk";
    const pct = diseasePercent();
    const out = liveOutputsRef.current;
    // Brightens with each ventricular contraction, more so at higher risk.
    const beat = out ? Math.max(0, Math.sin(out.tFrac * Math.PI * 2)) : 0;
    const target = risk ? riskColor(pct, color.current) : color.current.copy(ARTERY_COLOR);
    for (const mat of matRefs.current) {
      if (!mat) continue;
      mat.color.copy(target);
      mat.emissive.copy(target);
      mat.emissiveIntensity = risk ? 0.25 + (0.25 + pct / 100) * beat * 0.8 : 0.08;
    }
  });

  return (
    <group>
      {tubes.map((t, i) => (
        <mesh key={t.name} geometry={t.geo}>
          <meshStandardMaterial
            ref={(m) => {
              matRefs.current[i] = m;
            }}
            color={ARTERY_COLOR}
            roughness={0.45}
            metalness={0.05}
          />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Risk shading laid over the outer heart: a skin just above the muscle whose
 * per-vertex alpha is the LAD territory weight and whose overall opacity and
 * colour follow the predicted chance of heart disease. It marks where
 * coronary disease most often shows -- it is not a measured perfusion map.
 */
export function RiskTerritory({ geometry }: { geometry: THREE.BufferGeometry }) {
  const matRef = useRef<THREE.MeshBasicMaterial>(null);
  const shown = useRef(0);
  const skin = useMemo(() => {
    const geo = geometry.clone();
    const pos = geo.attributes.position;
    const rgba = new Float32Array(pos.count * 4);
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      // Every ventricle vertex gets some tint; the LAD territory the most.
      const w = 0.25 + 0.75 * ladTerritoryWeight(v);
      const ventricle = THREE.MathUtils.smoothstep(-v.y, -0.45, 0.05);
      rgba.set([1, 1, 1, w * ventricle], i * 4);
    }
    geo.setAttribute("color", new THREE.BufferAttribute(rgba, 4));
    return geo;
  }, [geometry]);

  useFrame((_, delta) => {
    const mat = matRef.current;
    if (!mat) return;
    const pct = diseasePercent();
    riskColor(pct, mat.color);
    // Ease toward the new strength so a what-if change fades in.
    shown.current += (0.1 + 0.5 * (pct / 100) - shown.current) * Math.min(1, delta * 3);
    mat.opacity = shown.current;
  });

  return (
    <mesh geometry={skin} scale={1.004} renderOrder={2}>
      <meshBasicMaterial ref={matRef} vertexColors transparent depthWrite={false} opacity={0} toneMapped={false} />
    </mesh>
  );
}
