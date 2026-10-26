import { useEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useHeartStore } from "../../state/store";

// Front of the heart (incl. the pulmonary trunk) to just behind the ventricles.
const FRONT_Z = 1.25;
const BACK_Z = -0.75;

/**
 * Slices the whole scene front-to-back with one clipping plane, like a
 * dissection: everything nearer the viewer than the plane is removed, so the
 * chambers, valves and walls show in cross-section. The plane is fixed to the
 * heart, so orbiting shows the cut from any side.
 */
export function CutPlane() {
  const gl = useThree((s) => s.gl);
  const cutDepth = useHeartStore((s) => s.xray.cutDepth);
  // Keeps points with z <= constant: normal (0, 0, -1).
  const plane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 0, -1), FRONT_Z), []);

  useEffect(() => {
    plane.constant = THREE.MathUtils.lerp(FRONT_Z, BACK_Z, cutDepth);
    gl.clippingPlanes = cutDepth > 0 ? [plane] : [];
  }, [gl, plane, cutDepth]);

  useEffect(() => () => void (gl.clippingPlanes = []), [gl]);
  return null;
}
