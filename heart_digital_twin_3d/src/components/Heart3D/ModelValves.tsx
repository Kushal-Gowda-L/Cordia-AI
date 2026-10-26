import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";

/**
 * The four heart valves inside the model hearts (realistic, hologram and
 * low-poly), seen by looking inside: cupped cusps that open and shut with the
 * simulation, a ring at their base, and for the two valves between atria and
 * ventricles (mitral, tricuspid) the chordae tendineae tying the cusps to
 * papillary muscles below, which tighten as the valve shuts.
 *
 * A narrowed aortic valve opens only part way and turns amber; a leaky mitral
 * valve never quite shuts and turns red. Each valve's ring briefly glows as
 * it closes -- the moment of a heart sound. The valves are never cut away
 * with the shell, so cutting the heart open shows them whole.
 */

type V3 = [number, number, number];

interface ValveSpec {
  name: "aortic" | "pulmonic" | "mitral" | "tricuspid";
  label: string;
  position: V3;
  /** Direction blood flows through it. */
  axis: V3;
  radius: number;
  cusps: number;
  /** Mitral and tricuspid: tethered by chordae to papillary muscles. */
  chordae: boolean;
  /** Label position relative to the valve, so the four labels don't overlap. */
  labelAt: V3;
}

// In the model hearts' own coordinates: at the roots of the great arteries and between atria and ventricles.
const VALVES: ValveSpec[] = [
  { name: "aortic", label: "Aortic valve", position: [0.01, 0.26, 0.02], axis: [-0.15, 1, 0.1], radius: 0.15, cusps: 3, chordae: false, labelAt: [0.42, 0.18, 0.1] },
  { name: "pulmonic", label: "Pulmonary valve", position: [-0.12, 0.3, 0.36], axis: [0.15, 1, 0], radius: 0.135, cusps: 3, chordae: false, labelAt: [-0.42, 0.22, 0.1] },
  { name: "mitral", label: "Mitral valve", position: [0.3, 0.1, -0.18], axis: [-0.25, -1, 0.3], radius: 0.17, cusps: 2, chordae: true, labelAt: [0.5, -0.12, 0.2] },
  { name: "tricuspid", label: "Tricuspid valve", position: [-0.36, 0.06, 0.04], axis: [0.25, -1, 0.25], radius: 0.18, cusps: 3, chordae: true, labelAt: [-0.5, -0.14, 0.2] },
];

const MAX_OPEN = 1.3; // radians a cusp swings open

/** Cusp outline at (u across -1..1, v from ring 0 to centre 1), in the hinge frame (ring edge on y = 0). */
function cuspPoint(radius: number, cusps: number, u: number, v: number, target = new THREE.Vector3()) {
  const half = radius * Math.sin(Math.PI / cusps) * 0.98;
  const L = radius * 0.97;
  // Narrows to a point at the centre, and bellies back against the flow like a pocket.
  const width = half * (1 - v) ** 0.85;
  const belly = -radius * 0.32 * (1 - u * u) * Math.sin(Math.PI * Math.min(1, v * 1.1));
  return target.set(u * width, -v * L, belly);
}

function makeCusp(radius: number, cusps: number): THREE.BufferGeometry {
  const nu = 12;
  const nv = 10;
  const pos: number[] = [];
  const idx: number[] = [];
  const p = new THREE.Vector3();
  for (let j = 0; j <= nv; j++) {
    for (let i = 0; i <= nu; i++) {
      cuspPoint(radius, cusps, -1 + (2 * i) / nu, j / nv, p);
      pos.push(p.x, p.y, p.z);
    }
  }
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const a = j * (nu + 1) + i;
      idx.push(a, a + 1, a + nu + 1, a + 1, a + nu + 2, a + nu + 1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

/** The two free edges of a cusp, where it meets its neighbours when shut. */
function makeCuspEdge(radius: number, cusps: number): THREE.BufferGeometry {
  const pts: THREE.Vector3[] = [];
  for (let j = 0; j <= 10; j++) pts.push(cuspPoint(radius, cusps, -1, j / 10));
  for (let j = 10; j >= 0; j--) pts.push(cuspPoint(radius, cusps, 1, j / 10));
  return new THREE.BufferGeometry().setFromPoints(pts);
}

const X = new THREE.Vector3(1, 0, 0);
const Z = new THREE.Vector3(0, 0, 1);
const CHORDS_PER_CUSP = 3;

/** Where a point of a cusp is, in the valve's frame, with the cusp swung open by `open` radians. */
function cuspPointInValve(spec: ValveSpec, cusp: number, u: number, v: number, open: number, target: THREE.Vector3) {
  cuspPoint(spec.radius, spec.cusps, u, v, target);
  target.applyAxisAngle(X, -open);
  target.y += spec.radius;
  return target.applyAxisAngle(Z, (cusp * 2 * Math.PI) / spec.cusps);
}

function papillaryTip(spec: ValveSpec, cusp: number, target = new THREE.Vector3()) {
  return target.set(0, spec.radius * 0.45, spec.radius * 2.3).applyAxisAngle(Z, (cusp * 2 * Math.PI) / spec.cusps);
}

export type ValveStyle = "glow" | "tissue";

const STYLE = {
  glow: { closed: "#3aa8ff", open: "#c8f4ff", edge: "#e6fbff", ring: "#7fe0ff", flash: "#ffffff", chord: "#9fe6ff", papillary: "#2f8dff" },
  tissue: { closed: "#e9dcc0", open: "#fff4dc", edge: "#fff8ea", ring: "#c47a6c", flash: "#ffd9c9", chord: "#f2e8d2", papillary: "#a8222c" },
};
const CALCIFIED = new THREE.Color("#ffb03a");
const LEAKY = new THREE.Color("#ff5a6e");

export function ModelValves({ style }: { style: ValveStyle }) {
  const showLabels = useHeartStore((s) => s.toggles.labels);
  const s = STYLE[style];
  const glow = style === "glow";

  const geos = useMemo(() => VALVES.map((v) => ({ cusp: makeCusp(v.radius, v.cusps), edge: makeCuspEdge(v.radius, v.cusps) })), []);
  const orient = useMemo(
    () => VALVES.map((v) => new THREE.Quaternion().setFromUnitVectors(Z, new THREE.Vector3(...v.axis).normalize())),
    []
  );
  // One material per valve, so all its cusps change colour together.
  const mats = useMemo(() => {
    const common = { side: THREE.DoubleSide, transparent: true };
    return VALVES.map(() => ({
      cusp: glow
        ? new THREE.MeshBasicMaterial({ ...common, color: s.closed, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })
        : new THREE.MeshStandardMaterial({ ...common, color: s.closed, roughness: 0.45, opacity: 0.95 }),
      ring: new THREE.MeshBasicMaterial({
        color: s.ring,
        transparent: true,
        opacity: glow ? 0.9 : 1,
        blending: glow ? THREE.AdditiveBlending : THREE.NormalBlending,
        depthWrite: !glow,
        toneMapped: false,
      }),
    }));
  }, [glow, s]);
  const edgeMat = useMemo(
    () => new THREE.LineBasicMaterial({ color: s.edge, transparent: true, opacity: glow ? 0.85 : 0.7, toneMapped: false }),
    [glow, s]
  );
  const chordMat = useMemo(
    () => new THREE.LineBasicMaterial({ color: s.chord, transparent: true, opacity: glow ? 0.55 : 0.8, toneMapped: false }),
    [glow, s]
  );
  const chordGeos = useMemo(
    () =>
      VALVES.map((v) => {
        if (!v.chordae) return null;
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(v.cusps * CHORDS_PER_CUSP * 2 * 3), 3));
        return g;
      }),
    []
  );

  const hinges = useRef<(THREE.Group | null)[][]>(VALVES.map((v) => Array(v.cusps).fill(null)));
  const rings = useRef<(THREE.Mesh | null)[]>([]);
  const shown = useRef(VALVES.map(() => 0));
  const flash = useRef(VALVES.map(() => 0));
  const tmp = useMemo(() => new THREE.Vector3(), []);
  const tip = useMemo(() => new THREE.Vector3(), []);
  const ringBase = useMemo(() => new THREE.Color(s.ring), [s]);
  const ringFlash = useMemo(() => new THREE.Color(s.flash), [s]);
  const closedColor = useMemo(() => new THREE.Color(s.closed), [s]);
  const openColor = useMemo(() => new THREE.Color(s.open), [s]);

  useFrame((_, delta) => {
    const out = liveOutputsRef.current;
    if (!out) return;
    const { aorticStenosis, mitralRegurgitation } = useHeartStore.getState().valves;
    VALVES.forEach((v, vi) => {
      const prev = shown.current[vi];
      // Cusps swing over a few tens of milliseconds rather than snapping.
      shown.current[vi] += (out.valves[v.name] - prev) * Math.min(1, delta * 22);
      let open = shown.current[vi];
      if (prev > 0.25 && open <= 0.25) flash.current[vi] = 1;
      flash.current[vi] = Math.max(0, flash.current[vi] - delta * 3.5);
      if (v.name === "mitral") open = Math.max(open, mitralRegurgitation * 0.25);
      const angle = open * MAX_OPEN;
      for (const h of hinges.current[vi]) if (h) h.rotation.x = -angle;

      const m = mats[vi];
      m.cusp.color.copy(closedColor).lerp(openColor, open);
      if (v.name === "aortic" && aorticStenosis) m.cusp.color.lerp(CALCIFIED, aorticStenosis * 0.8);
      if (v.name === "mitral" && mitralRegurgitation) m.cusp.color.lerp(LEAKY, mitralRegurgitation * 0.6);
      if (glow) m.cusp.opacity = 0.4 + 0.35 * open;
      m.ring.color.copy(ringBase).lerp(ringFlash, flash.current[vi]);
      const ring = rings.current[vi];
      if (ring) ring.scale.setScalar(1 + flash.current[vi] * 0.12);

      // Chordae run from near each cusp's tip to its papillary muscle.
      const cg = chordGeos[vi];
      if (cg) {
        const pos = cg.attributes.position as THREE.BufferAttribute;
        let k = 0;
        for (let c = 0; c < v.cusps; c++) {
          papillaryTip(v, c, tip);
          for (let j = 0; j < CHORDS_PER_CUSP; j++) {
            cuspPointInValve(v, c, -0.55 + 0.55 * j, 0.8, angle, tmp);
            pos.setXYZ(k++, tmp.x, tmp.y, tmp.z);
            pos.setXYZ(k++, tip.x, tip.y, tip.z);
          }
        }
        pos.needsUpdate = true;
        cg.computeBoundingSphere();
      }
    });
  });

  return (
    <group>
      {VALVES.map((v, vi) => (
        <group key={v.name} position={v.position} quaternion={orient[vi]}>
          <mesh
            material={mats[vi].ring}
            ref={(m) => {
              rings.current[vi] = m;
            }}
          >
            <torusGeometry args={[v.radius, v.radius * (glow ? 0.06 : 0.1), 10, 48]} />
          </mesh>
          {Array.from({ length: v.cusps }).map((_, ci) => (
            <group key={ci} rotation={[0, 0, (ci * 2 * Math.PI) / v.cusps]}>
              <group position={[0, v.radius, 0]}>
                <group
                  ref={(g) => {
                    hinges.current[vi][ci] = g;
                  }}
                >
                  <mesh geometry={geos[vi].cusp} material={mats[vi].cusp} />
                  <lineLoop geometry={geos[vi].edge} material={edgeMat} />
                </group>
              </group>
            </group>
          ))}
          {chordGeos[vi] && (
            <>
              <lineSegments geometry={chordGeos[vi]!} material={chordMat} />
              {Array.from({ length: v.cusps }).map((_, ci) => {
                const at = papillaryTip(v, ci);
                // A small cone of papillary muscle, its tip where the chordae meet.
                return (
                  <mesh key={ci} position={[at.x, at.y, at.z + v.radius * 0.28]} rotation={[-Math.PI / 2, 0, 0]}>
                    <coneGeometry args={[v.radius * 0.16, v.radius * 0.56, 12]} />
                    {glow ? (
                      <meshBasicMaterial color={s.papillary} transparent opacity={0.6} blending={THREE.AdditiveBlending} depthWrite={false} />
                    ) : (
                      <meshStandardMaterial color={s.papillary} roughness={0.5} />
                    )}
                  </mesh>
                );
              })}
            </>
          )}
        </group>
      ))}
      {showLabels &&
        VALVES.map((v) => (
          <Html
            key={`label-${v.name}`}
            position={[v.position[0] + v.labelAt[0], v.position[1] + v.labelAt[1], v.position[2] + v.labelAt[2]]}
            center
            distanceFactor={3.2}
            occlude={false}
          >
            <div className="holo-label">{v.label}</div>
          </Html>
        ))}
    </group>
  );
}
