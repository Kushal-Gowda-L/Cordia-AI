import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { liveOutputsRef } from "../../simulation/liveOutputsRef";
import { useHeartStore } from "../../state/store";
import { SHELL_CUT } from "./shellCut";

/**
 * The four chambers inside the model hearts (realistic, hologram, low-poly):
 * the right atrium and ventricle (oxygen-poor, blue) and the left atrium and
 * ventricle (oxygen-rich, red), each filling and emptying with its own
 * simulated volume, so the pumping inside matches the numbers. They sit
 * under the valves: the mitral valve opens into the left ventricle, the
 * tricuspid into the right, and the aortic and pulmonary valves leave from
 * their tops. Cutting the heart open slices through them too, showing each
 * cavity in section.
 */

type V3 = [number, number, number];
type Key = "vRA" | "vRV" | "vLA" | "vLV";

interface ChamberSpec {
  short: string;
  name: string;
  volume: Key;
  /** Volume (mL) at which the chamber is drawn at its modelled size. */
  reference: number;
  position: V3;
  rotation: V3;
  size: V3;
  /** Ventricles taper to a point toward the apex. */
  ventricle: boolean;
  oxygenated: boolean;
}

// In the model hearts' own coordinates (the apex points down and to the viewer's right).
const CHAMBERS: ChamberSpec[] = [
  { short: "RA", name: "Right atrium", volume: "vRA", reference: 55, position: [-0.44, 0.3, 0.0], rotation: [0, 0, 0.2], size: [0.24, 0.23, 0.24], ventricle: false, oxygenated: false },
  { short: "LA", name: "Left atrium", volume: "vLA", reference: 55, position: [0.28, 0.4, -0.33], rotation: [0, 0, 0], size: [0.32, 0.19, 0.22], ventricle: false, oxygenated: true },
  { short: "RV", name: "Right ventricle", volume: "vRV", reference: 120, position: [-0.2, -0.33, 0.16], rotation: [0.15, 0, 0.3], size: [0.22, 0.42, 0.17], ventricle: true, oxygenated: false },
  { short: "LV", name: "Left ventricle", volume: "vLV", reference: 120, position: [0.3, -0.42, -0.1], rotation: [0, 0, 0.35], size: [0.24, 0.5, 0.24], ventricle: true, oxygenated: true },
];

function makeChamber(c: ChamberSpec, segments: [number, number]): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, segments[0], segments[1]);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    // A ventricle narrows toward its apex; an atrium is a rounded pouch.
    const taper = c.ventricle && v.y < 0 ? 1 - 0.55 * Math.pow(-v.y, 1.4) : 1;
    pos.setXYZ(i, v.x * c.size[0] * taper, v.y * c.size[1], v.z * c.size[2] * taper);
  }
  geo.computeVertexNormals();
  return geo;
}

export type ChamberStyle = "glow" | "mesh" | "tissue";

const COLORS = {
  glow: { red: "#ff4d6d", blue: "#4da6ff" },
  mesh: { red: "#ff5a78", blue: "#5ab0ff" },
  tissue: { red: "#b3121f", blue: "#4b3f8f" },
};

export function ModelChambers({ style }: { style: ChamberStyle }) {
  const showLabels = useHeartStore((s) => s.toggles.labels);
  const geos = useMemo(() => CHAMBERS.map((c) => makeChamber(c, style === "mesh" ? [10, 8] : [40, 28])), [style]);
  const wires = useMemo(() => (style === "mesh" ? geos.map((g) => new THREE.WireframeGeometry(g)) : null), [geos, style]);
  const groups = useRef<(THREE.Group | null)[]>([]);
  const shown = useRef(CHAMBERS.map(() => 1));

  useFrame((_, delta) => {
    const out = liveOutputsRef.current;
    if (!out) return;
    CHAMBERS.forEach((c, i) => {
      const target = THREE.MathUtils.clamp(Math.pow(Math.max(out[c.volume], 1) / c.reference, 1 / 3), 0.55, 1.25);
      shown.current[i] += (target - shown.current[i]) * Math.min(1, delta * 20);
      const g = groups.current[i];
      if (!g) return;
      const k = shown.current[i];
      // Ventricles shorten toward their base and narrow as they empty.
      if (c.ventricle) g.scale.set(k, 0.75 + 0.25 * k, k);
      else g.scale.setScalar(k);
    });
  });

  const material = (c: ChamberSpec) => {
    const color = c.oxygenated ? COLORS[style].red : COLORS[style].blue;
    if (style === "glow")
      return (
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.28}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          side={THREE.DoubleSide}
          toneMapped={false}
          clippingPlanes={SHELL_CUT}
        />
      );
    if (style === "mesh")
      return (
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.45}
          flatShading
          transparent
          opacity={0.42}
          depthWrite={false}
          side={THREE.DoubleSide}
          clippingPlanes={SHELL_CUT}
        />
      );
    // Blood seen through the opened heart: glossy and slightly see-through.
    return (
      <meshPhysicalMaterial
        color={color}
        roughness={0.25}
        clearcoat={1}
        clearcoatRoughness={0.15}
        transparent
        opacity={0.82}
        side={THREE.DoubleSide}
        clippingPlanes={SHELL_CUT}
      />
    );
  };

  return (
    <group>
      {CHAMBERS.map((c, i) => (
        <group key={c.short} position={c.position} rotation={c.rotation}>
          <group
            ref={(g) => {
              groups.current[i] = g;
            }}
          >
            <mesh geometry={geos[i]} renderOrder={-1}>
              {material(c)}
            </mesh>
            {wires && (
              <lineSegments geometry={wires[i]}>
                <lineBasicMaterial
                  color={c.oxygenated ? "#ffb3c2" : "#b8deff"}
                  transparent
                  opacity={0.55}
                  blending={THREE.AdditiveBlending}
                  depthWrite={false}
                  clippingPlanes={SHELL_CUT}
                />
              </lineSegments>
            )}
          </group>
          {showLabels && (
            <Html center distanceFactor={3.2} occlude={false}>
              <div className="holo-label" title={c.name}>
                {c.short}
              </div>
            </Html>
          )}
        </group>
      ))}
    </group>
  );
}
