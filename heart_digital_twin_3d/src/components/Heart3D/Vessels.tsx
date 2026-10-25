import { useMemo } from "react";
import * as THREE from "three";
import { VESSELS } from "./heartGeometry";
import { useHeartStore } from "../../state/store";

export function Vessels() {
  const toggles = useHeartStore((s) => s.toggles);
  const viewMode = useHeartStore((s) => s.viewMode);

  const tubes = useMemo(
    () =>
      VESSELS.map((v) => {
        const curve = new THREE.CatmullRomCurve3(v.points.map((p) => new THREE.Vector3(...p)));
        const geo = new THREE.TubeGeometry(curve, 24, v.radius, 12, false);
        return { ...v, geo, curve };
      }),
    []
  );

  if (!toggles.vessels) return null;
  const dimmed = viewMode === "electrical" || viewMode === "pressure";
  // Solid vessel walls in the muted anatomical hues of a textbook plate
  // (dusty red for oxygenated, mauve-blue for deoxygenated) -- except in
  // blood-flow mode, where they go translucent so the particles inside
  // are visible.
  const translucent = viewMode === "bloodFlow" || dimmed;

  return (
    <group>
      {tubes.map((v) => (
        <mesh key={v.name} geometry={v.geo}>
          <meshStandardMaterial
            color={v.oxygenated ? "#b4565c" : "#6b7bad"}
            transparent={translucent}
            opacity={dimmed ? 0.2 : translucent ? 0.45 : 1}
            roughness={0.55}
            metalness={0.05}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

export function vesselCurves() {
  return VESSELS.map((v) => ({
    name: v.name,
    oxygenated: v.oxygenated,
    curve: new THREE.CatmullRomCurve3(v.points.map((p) => new THREE.Vector3(...p))),
  }));
}
