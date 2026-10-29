import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useHeartStore } from "../../state/store";

/**
 * "Cut away front" for the model hearts (realistic, hologram, low-poly): one
 * clipping plane facing the viewer, applied only to the heart's outer shell,
 * so slicing it away opens the heart while the valves inside stay whole.
 * Unlike the clinical views' CutPlane it follows the camera, so the open
 * side always faces you while the heart turns.
 */
export const SHELL_CUT_PLANE = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e6);
export const SHELL_CUT = [SHELL_CUT_PLANE];

const FRONT = 1.0; // just in front of the heart
const BACK = -0.35; // past the valves, short of the back wall

const toCamera = new THREE.Vector3();

/** Keeps the plane between the camera and the heart's centre, at the slider's depth. */
export function ShellCutDriver() {
  useFrame(({ camera }) => {
    const depth = useHeartStore.getState().xray.cutDepth;
    if (depth <= 0) {
      SHELL_CUT_PLANE.constant = 1e6;
      return;
    }
    toCamera.copy(camera.position).normalize();
    // Keeps points with dot(n, p) + c >= 0, i.e. those farther from the camera than the cut.
    SHELL_CUT_PLANE.normal.copy(toCamera).negate();
    SHELL_CUT_PLANE.constant = THREE.MathUtils.lerp(FRONT, BACK, depth);
  });
  return null;
}
