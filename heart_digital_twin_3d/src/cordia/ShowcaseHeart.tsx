import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { bodyPoint, dirOf, RealisticHeart } from "../components/Heart3D/RealisticHeart";
import { ShellCutDriver } from "../components/Heart3D/shellCut";
import { VIEWS } from "../components/Heart3D/Heart3D";

/**
 * The dashboard's heart: the realistic twin in its illustrated colours on a
 * glowing holographic platform, turning a full 360 degrees, with leader-line
 * labels that stay pinned to each structure as it turns. The heart beats
 * with the page's simulation (liveOutputsRef), so it follows the controls.
 */

export type ShowcaseView = "front" | "left" | "right" | "back";

interface Label {
  name: string;
  tone: "red" | "blue" | "gold";
  /** Point on the structure, in the heart model's own coordinates. */
  at: THREE.Vector3;
}

// RealisticHeart sits at y -0.12, scale 0.88, inside its own group, which this page lifts by 0.1.
const toWorld = (p: THREE.Vector3) => p.clone().multiplyScalar(0.88).add(new THREE.Vector3(0, -0.02, 0));
const onBody = (polar: number, az: number) => bodyPoint(dirOf(polar, az)).multiplyScalar(1.02);

const LABELS: Label[] = (
  [
  { name: "Aorta", tone: "red", at: new THREE.Vector3(0.1, 1.3, -0.12) },
  { name: "Superior Vena Cava", tone: "blue", at: new THREE.Vector3(-0.5, 1.02, 0.08) },
  { name: "Pulmonary Artery", tone: "blue", at: new THREE.Vector3(0.52, 0.97, 0.05) },
  { name: "Left Atrium", tone: "red", at: new THREE.Vector3(0.58, 0.36, 0.26) },
  { name: "Right Atrium", tone: "blue", at: new THREE.Vector3(-0.6, 0.3, 0.2) },
  { name: "Left Ventricle", tone: "red", at: onBody(1.8, 0.5) },
  { name: "Right Ventricle", tone: "blue", at: onBody(1.8, 2.2) },
  { name: "Coronary Artery", tone: "gold", at: onBody(1.72, 1.47) },
  ] as Label[]
).map((l) => ({ ...l, at: toWorld(l.at) }));

const CENTER = new THREE.Vector3(0, -0.18, 0);

/** Projects each label's point to the screen and lays the labels out in two columns beside the heart. */
function LabelLayout({ els, lines, show }: { els: (HTMLDivElement | null)[]; lines: (SVGPolylineElement | null)[]; show: boolean }) {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);
  const toCam = useMemo(() => new THREE.Vector3(), []);
  const side = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    if (!show) return;
    const w = size.width;
    const h = size.height;
    const project = (p: THREE.Vector3) => {
      v.copy(p).project(camera);
      return { x: ((v.x + 1) / 2) * w, y: ((1 - v.y) / 2) * h };
    };
    const c = project(CENTER);
    // The heart's half-width on screen sets how far out the label columns sit.
    toCam.copy(camera.position).sub(CENTER).normalize();
    side.crossVectors(toCam, camera.up).normalize().multiplyScalar(1.1).add(CENTER);
    const half = Math.abs(project(side).x - c.x);
    const gap = Math.min(half + 24, w / 2 - 12);
    const items = LABELS.map((l, i) => {
      const s = project(l.at);
      // Facing the viewer, or round the back of the heart?
      const facing = v.copy(l.at).sub(CENTER).normalize().dot(toCam) > -0.15;
      return { i, s, facing, left: s.x < c.x, y: s.y };
    });
    // Balance the columns: at most five labels a side, moving the ones nearest the other side.
    const byX = [...items].sort((a, b) => a.s.x - b.s.x);
    byX.forEach((it, k) => {
      if (it.left && k >= 5) it.left = false;
      if (!it.left && k < byX.length - 5) it.left = true;
    });
    for (const left of [true, false]) {
      const col = items.filter((it) => it.left === left).sort((a, b) => a.s.y - b.s.y);
      // Keep labels at their point's height, pushed apart so none overlap.
      let last = -Infinity;
      for (const it of col) {
        it.y = Math.max(it.s.y, last + 34);
        last = it.y;
      }
      const overflow = last - (h - 16);
      if (overflow > 0) col.forEach((it) => (it.y -= overflow));
    }
    for (const it of items) {
      const el = els[it.i];
      const line = lines[it.i];
      if (!el || !line) continue;
      // Keep the whole label on screen.
      const lx = it.left ? Math.max(c.x - gap, el.offsetWidth + 6) : Math.min(c.x + gap, w - el.offsetWidth - 6);
      el.style.transform = `translate(${it.left ? `calc(${lx}px - 100%)` : `${lx}px`}, ${it.y - 12}px)`;
      el.classList.toggle("left", it.left);
      el.style.opacity = it.facing ? "1" : "0.35";
      line.setAttribute("points", `${lx + (it.left ? 6 : -6)},${it.y} ${it.left ? lx + 22 : lx - 22},${it.y} ${it.s.x},${it.s.y}`);
      line.style.opacity = it.facing ? "0.9" : "0.25";
    }
  });
  return null;
}

/** The glowing platform the heart floats over: a disc, rings and a slowly turning ring of ticks. */
function Platform() {
  const ticks = useRef<THREE.Group>(null);
  const glow = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d")!;
    const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
    grad.addColorStop(0, "rgba(90,190,255,0.55)");
    grad.addColorStop(0.45, "rgba(40,130,255,0.22)");
    grad.addColorStop(1, "rgba(20,80,200,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(c);
  }, []);
  useFrame((_, dt) => {
    if (ticks.current) ticks.current.rotation.y += dt * 0.15;
  });
  const glowMat = { transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false } as const;
  return (
    <group position={[0, -1.42, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[3.1, 3.1]} />
        <meshBasicMaterial map={glow} {...glowMat} />
      </mesh>
      {[0.66, 0.94, 1.24].map((r, i) => (
        <mesh key={r} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[r, r + (i === 1 ? 0.035 : 0.018), 128]} />
          <meshBasicMaterial color={i === 1 ? "#5cc8ff" : "#2f8cff"} opacity={i === 1 ? 0.95 : 0.6} {...glowMat} side={THREE.DoubleSide} />
        </mesh>
      ))}
      <group ref={ticks}>
        {Array.from({ length: 72 }, (_, k) => {
          const a = (k / 72) * Math.PI * 2;
          const long = k % 6 === 0;
          return (
            <mesh key={k} position={[Math.cos(a) * 1.1, 0, Math.sin(a) * 1.1]} rotation={[-Math.PI / 2, 0, -a]}>
              <planeGeometry args={[long ? 0.09 : 0.05, 0.012]} />
              <meshBasicMaterial color="#7fd6ff" opacity={long ? 0.9 : 0.5} {...glowMat} side={THREE.DoubleSide} />
            </mesh>
          );
        })}
      </group>
    </group>
  );
}

function Lights() {
  return (
    <>
      <hemisphereLight args={["#ffffff", "#1a2440", 0.9]} />
      <directionalLight position={[2.5, 3.5, 3.5]} intensity={2.6} color="#fff6f2" />
      <directionalLight position={[-3, 1.5, 2]} intensity={0.9} color="#ffe9e4" />
      {/* Blue rim light from behind, as in the reference render */}
      <directionalLight position={[-2.5, 1, -3.5]} intensity={2.2} color="#4d9bff" />
      <directionalLight position={[2.5, 0.5, -3]} intensity={1.4} color="#3a7dff" />
      <pointLight position={[0.4, 0.8, 2.8]} intensity={7} distance={8} decay={2} color="#ffffff" />
      <pointLight position={[0, -1.4, 0.4]} intensity={4} distance={4} decay={2} color="#3aa0ff" />
    </>
  );
}

/** Moves the camera to a requested view and distance. */
function Rig({ view, distance }: { view: { name: ShowcaseView; n: number }; distance: number }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const aspect = useThree((s) => s.size.width / Math.max(1, s.size.height));
  useEffect(() => {
    const [theta, phi] = VIEWS[view.name];
    const t = controls?.target ?? CENTER;
    // Step back on narrow (portrait) screens so the whole heart fits across.
    const r = distance * Math.pow(Math.max(1, 1.1 / aspect), 0.75);
    camera.position.set(t.x + r * Math.sin(phi) * Math.sin(theta), t.y + r * Math.cos(phi), t.z + r * Math.sin(phi) * Math.cos(theta));
    camera.lookAt(t);
    controls?.update();
  }, [view, distance, aspect, camera, controls]);
  return null;
}

export function ShowcaseHeart({
  rotate,
  labels,
  view,
  distance,
}: {
  rotate: boolean;
  labels: boolean;
  view: { name: ShowcaseView; n: number };
  distance: number;
}) {
  const els = useRef<(HTMLDivElement | null)[]>([]);
  const lines = useRef<(SVGPolylineElement | null)[]>([]);
  return (
    <div className="showcase">
      <Canvas camera={{ position: [0, 0.4, 4.7], fov: 40 }} gl={{ antialias: true, localClippingEnabled: true, alpha: true, preserveDrawingBuffer: true }} dpr={[1, 2]}>
        <Lights />
        <Suspense fallback={null}>
          <ShellCutDriver />
          <group position={[0, 0.1, 0]}>
            <RealisticHeart look="illustrated" shadow={false} />
          </group>
          <Platform />
        </Suspense>
        <OrbitControls
          makeDefault
          enablePan={false}
          enableDamping
          minDistance={2.4}
          maxDistance={8}
          target={[CENTER.x, CENTER.y, CENTER.z]}
          autoRotate={rotate}
          autoRotateSpeed={2}
        />
        <Rig view={view} distance={distance} />
        <LabelLayout els={els.current} lines={lines.current} show={labels} />
      </Canvas>
      {labels && (
        <>
          <svg className="leaders" aria-hidden="true">
            {LABELS.map((l, i) => (
              <polyline key={l.name} ref={(e) => void (lines.current[i] = e)} className={`tone-${l.tone}`} points="0,0 0,0" />
            ))}
          </svg>
          {LABELS.map((l, i) => (
            <div key={l.name} ref={(e) => void (els.current[i] = e)} className={`part-label tone-${l.tone}`}>
              <i />
              {l.name}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
