import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  AURICLES,
  CHAMBERS,
  makeAuricleGeometry,
  makeMyocardiumGeometry,
  makeThickShell,
  PAPILLARY,
} from "./heartGeometry";
import { createMuscleBumpTexture, createMuscleColorTexture } from "./muscleTexture";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { showsInterior, useHeartStore } from "../../state/store";
import { CoronaryArteries, RiskTerritory } from "./RiskOverlay";

const REFERENCE_VOLUME: Record<string, number> = { LV: 170, RV: 180, LA: 55, RA: 50 };
const STRAIN_COLOR = new THREE.Color("#ff2a18");

/**
 * Drawn size for a chamber volume. A cube root (true geometric scaling) moves
 * the chamber by only a few percent between a strong and a weak beat, which
 * is invisible on screen; the steeper 0.6 power makes a shallower beat or a
 * fuller heart visibly different while staying monotonic in volume.
 */
function chamberScale(volume: number, reference: number): number {
  return THREE.MathUtils.clamp(Math.pow(Math.max(volume, 0) / reference, 0.6), 0.5, 1.3);
}

/** Red glow strength for how much harder the heart works than in the record (heart rate x blood pressure). */
function strainIntensity(): number {
  const { currentVitals, recordVitals, scenario } = useHeartStore.getState();
  if (!currentVitals || !recordVitals) return 0;
  const extra = currentVitals.workload / recordVitals[scenario].workload - 1;
  return Math.min(0.9, 2.2 * Math.max(0, extra));
}

function pressureColor(p: number, maxP: number): THREE.Color {
  const t = THREE.MathUtils.clamp(p / maxP, 0, 1);
  const c = new THREE.Color();
  c.setHSL(THREE.MathUtils.lerp(0.55, 0.0, t), 0.85, THREE.MathUtils.lerp(0.35, 0.55, t));
  return c;
}

export function HeartChambers() {
  const viewMode = useHeartStore((s) => s.viewMode);
  const toggles = useHeartStore((s) => s.toggles);
  const wallOpacity = useHeartStore((s) => s.xray.wallOpacity);
  const interior = useHeartStore((s) => showsInterior(s));

  const poolRefs = useRef<Record<string, THREE.Mesh | null>>({});
  const wallRefs = useRef<Record<string, THREE.Group | null>>({});
  const poolMatRefs = useRef<Record<string, THREE.MeshStandardMaterial | null>>({});
  const muscleMatRefs = useRef<Record<string, THREE.MeshStandardMaterial | null>>({});
  const myoRef = useRef<THREE.Group>(null);
  const glowRef = useRef(0);
  const viewRef = useRef(viewMode);
  viewRef.current = viewMode;

  const wallGeometries = useMemo(() => {
    const map: Record<string, THREE.BufferGeometry> = {};
    for (const c of CHAMBERS) {
      map[c.name] = makeThickShell({
        radii: c.radii,
        thickness: c.thickness,
        gap: c.gap,
        phiStart: c.phiStart,
        apexTaper: c.apexTaper,
      });
    }
    return map;
  }, []);
  const myoGeometry = useMemo(() => makeMyocardiumGeometry(), []);
  const auricleGeometry = useMemo(() => makeAuricleGeometry(), []);
  const muscleColorMap = useMemo(() => createMuscleColorTexture(), []);
  const muscleBumpMap = useMemo(() => createMuscleBumpTexture(), []);
  const endocardiumMap = useMemo(() => createMuscleColorTexture("#5e1414", "#742020", "#400d0d"), []);

  // Chordae tendineae: fine tendon strands from each papillary muscle tip
  // up to the rim of the AV valve it anchors.
  const chordaeGeometry = useMemo(() => {
    const pts: number[] = [];
    for (const pap of PAPILLARY) {
      const tip = new THREE.Vector3(pap.position[0], pap.position[1] + 0.07, pap.position[2]);
      for (let i = 0; i < pap.chords; i++) {
        const spread = (i / (pap.chords - 1) - 0.5) * 0.11;
        pts.push(tip.x, tip.y, tip.z, pap.anchor[0] + spread, pap.anchor[1] - 0.03, pap.anchor[2] + spread * 0.6);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }, []);

  useFrame((_, delta) => {
    // Ease the strain glow so a slider change fades in rather than flashing.
    glowRef.current += (strainIntensity() - glowRef.current) * Math.min(1, delta * 3);
    for (const mat of Object.values(muscleMatRefs.current)) {
      if (!mat) continue;
      mat.emissive.copy(STRAIN_COLOR);
      mat.emissiveIntensity = glowRef.current;
    }

    const out = liveOutputsRef.current;
    if (!out) return;
    if (myoRef.current) {
      myoRef.current.scale.setScalar(1 + (chamberScale(out.vLV + out.vRV, 350) - 1) * 0.4);
    }
    const volumes: Record<string, number> = { LV: out.vLV, RV: out.vRV, LA: out.vLA, RA: out.vRA };
    const pressures: Record<string, number> = { LV: out.pLV, RV: out.pRV, LA: out.pLA, RA: out.pRA };

    for (const c of CHAMBERS) {
      const scale = chamberScale(volumes[c.name], REFERENCE_VOLUME[c.name]);

      // The blood pool tracks chamber volume directly; the muscular wall
      // follows most of that movement (real myocardium thickens as the
      // cavity empties rather than collapsing with it). 0.96 keeps the pool
      // just inside the endocardial surface instead of z-fighting against it.
      const pool = poolRefs.current[c.name];
      if (pool) {
        pool.scale.set(
          (c.radii[0] - c.thickness) * scale * 0.96,
          (c.radii[1] - c.thickness) * scale * 0.96,
          (c.radii[2] - c.thickness) * scale * 0.96
        );
      }
      const wall = wallRefs.current[c.name];
      if (wall) wall.scale.setScalar(1 + (scale - 1) * 0.65);

      const mat = poolMatRefs.current[c.name];
      if (!mat) continue;
      if (viewRef.current === "pressure") {
        const maxP = c.name === "LV" ? 150 : c.name === "RV" ? 35 : 20;
        mat.color.copy(pressureColor(pressures[c.name], maxP));
        mat.emissive.copy(mat.color).multiplyScalar(0.3);
      } else {
        const base = c.colorOxygenated ? new THREE.Color("#c8323c") : new THREE.Color("#3f6fc0");
        mat.color.copy(base);
        mat.emissive.copy(base).multiplyScalar(0.12 + 0.18 * (pressures[c.name] / (c.name.includes("V") ? 150 : 20)));
      }
    }
  });

  // The closed heart (Outside, Risk) hides the chambers unless the wall is
  // made see-through or sliced away, which reveals the same interior the
  // opened views show.
  const showIntactMyocardium = viewMode === "anatomy" || viewMode === "risk";
  const cutawayVisible = toggles.chambers && interior;
  const seeThrough = wallOpacity < 0.98;
  // The blood pool shows how full each chamber is, but it must not hide the
  // cutaway interior -- papillary muscles, chordae and the endocardial
  // surface all sit inside it -- so it stays translucent.
  const poolOpacity = viewMode === "electrical" || viewMode === "valves" ? 0.22 : 0.45;

  return (
    <group>
      {showIntactMyocardium && (
        <group ref={myoRef}>
        <mesh geometry={myoGeometry}>
          <meshStandardMaterial
            ref={(m) => {
              muscleMatRefs.current.myocardium = m;
            }}
            color="#ffffff"
            map={muscleColorMap}
            bumpMap={muscleBumpMap}
            bumpScale={0.012}
            roughness={0.6}
            metalness={0.05}
            side={THREE.DoubleSide}
            transparent={seeThrough}
            opacity={wallOpacity}
            depthWrite={!seeThrough}
          />
        </mesh>
        {toggles.vessels && <CoronaryArteries />}
        {viewMode === "risk" && <RiskTerritory geometry={myoGeometry} />}
        </group>
      )}

      {cutawayVisible && (
        <>
          {CHAMBERS.map((c) => (
            <group key={c.name}>
              {/* Myocardial wall with real thickness -- the exposed cut rim
                  shows the LV wall is ~2.5x the RV's */}
              <group
                position={c.center}
                ref={(g) => {
                  wallRefs.current[c.name] = g;
                }}
              >
                <mesh geometry={wallGeometries[c.name]}>
                  <meshStandardMaterial
                    ref={(m) => {
                      muscleMatRefs.current[c.name] = m;
                    }}
                    color="#ffffff"
                    map={muscleColorMap}
                    bumpMap={muscleBumpMap}
                    bumpScale={0.014}
                    roughness={0.65}
                    metalness={0.04}
                    side={THREE.DoubleSide}
                  />
                </mesh>
              </group>

              {/* Blood inside the cavity, sized by the live simulated volume */}
              <mesh
                position={[c.center[0], c.center[1] + c.poolOffset, c.center[2]]}
                scale={[c.radii[0] - c.thickness, c.radii[1] - c.thickness, c.radii[2] - c.thickness]}
                ref={(m) => {
                  poolRefs.current[c.name] = m;
                }}
              >
                <sphereGeometry args={[1, 28, 20]} />
                <meshStandardMaterial
                  ref={(m) => {
                    poolMatRefs.current[c.name] = m;
                  }}
                  color={c.colorOxygenated ? "#c8323c" : "#3f6fc0"}
                  emissive={c.colorOxygenated ? "#5e1010" : "#12325e"}
                  emissiveIntensity={0.5}
                  bumpMap={endocardiumMap}
                  bumpScale={0.008}
                  roughness={0.45}
                  metalness={0.08}
                  transparent
                  opacity={poolOpacity}
                  depthWrite={false}
                />
              </mesh>
            </group>
          ))}

          {/* Interventricular septum -- kept behind the opening so it reads
              as the wall between the ventricles */}
          <mesh position={[-0.01, -0.34, -0.09]} rotation={[0, 0, 0.04]}>
            <boxGeometry args={[0.1, 0.95, 0.42]} />
            <meshStandardMaterial
              color="#ffffff"
              map={endocardiumMap}
              bumpMap={muscleBumpMap}
              bumpScale={0.012}
              roughness={0.75}
            />
          </mesh>

          {PAPILLARY.map((p, i) => (
            <mesh key={i} position={p.position}>
              <coneGeometry args={[0.045, 0.16, 12]} />
              <meshStandardMaterial
                color="#ffffff"
                map={endocardiumMap}
                bumpMap={muscleBumpMap}
                bumpScale={0.01}
                roughness={0.7}
              />
            </mesh>
          ))}

          <lineSegments geometry={chordaeGeometry}>
            <lineBasicMaterial color="#e8e0cf" transparent opacity={0.75} />
          </lineSegments>
        </>
      )}

      {/* Atrial auricles -- present in both the intact and cutaway views */}
      {(showIntactMyocardium || cutawayVisible) &&
        AURICLES.map((a, i) => (
          <mesh key={i} position={a.position} rotation={a.rotation} geometry={auricleGeometry}>
            <meshStandardMaterial
              color={a.oxygenated ? "#b0434c" : "#5b78ad"}
              bumpMap={muscleBumpMap}
              bumpScale={0.012}
              roughness={0.7}
            />
          </mesh>
        ))}
    </group>
  );
}
