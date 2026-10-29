import * as THREE from "three";

/**
 * Procedural anatomy, modelled on the standard anterior cutaway view used
 * in cardiology textbooks (the "major vessels and structures" plate):
 * each chamber is its own thick-walled shell opened at the front, with a
 * left ventricular wall visibly thicker than the right, papillary muscles
 * and chordae tendineae inside the ventricles, atrial auricles, and the
 * aorta arching behind the pulmonary trunk that crosses in front of it.
 *
 * No external mesh assets are used -- no licensed or public-domain
 * segmented heart mesh was available to bundle -- so every shape here is
 * generated from parametric math. This is a stylized but anatomically
 * organised approximation, not a medical-grade reconstruction. Orientation
 * follows the clinical convention: viewed from the front, the heart's own
 * right side (RA, RV, venae cavae) is on the viewer's LEFT.
 */

export type Vec3 = [number, number, number];

export interface ThickShellOptions {
  radii: Vec3;
  thickness: number;
  gap?: number; // half-angle of the anterior opening (radians)
  gapCenter?: number; // azimuth the opening faces; default +Z (toward viewer)
  phiStart?: number; // polar start -- leaves the chamber's base open at the valve
  phiEnd?: number;
  apexTaper?: number;
  uSeg?: number;
  vSeg?: number;
}

/**
 * A chamber wall with real thickness: an outer myocardial surface, an
 * inner endocardial surface, and rim faces joining them along every cut
 * edge. The rim faces are the exposed wall cross-section that makes a
 * cutaway read as a cutaway, and they show the LV/RV wall-thickness
 * difference at a glance.
 */
export function makeThickShell(o: ThickShellOptions): THREE.BufferGeometry {
  const gap = o.gap ?? 0.95;
  const gapCenter = o.gapCenter ?? Math.PI / 2;
  const phiStart = o.phiStart ?? 0.5;
  const phiEnd = o.phiEnd ?? Math.PI * 0.985;
  const apexTaper = o.apexTaper ?? 0.25;
  const uSeg = o.uSeg ?? 40;
  const vSeg = o.vSeg ?? 26;
  const thetaStart = gapCenter + gap;
  const thetaEnd = gapCenter - gap + Math.PI * 2;

  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];

  const push = (u: number, v: number, inner: boolean) => {
    const theta = thetaStart + (thetaEnd - thetaStart) * u;
    const phi = phiStart + (phiEnd - phiStart) * v;
    const t = inner ? o.thickness : 0;
    const rx = Math.max(0.02, o.radii[0] - t);
    const ry = Math.max(0.02, o.radii[1] - t);
    const rz = Math.max(0.02, o.radii[2] - t);
    const taper = 1 - apexTaper * Math.max(0, (phi - Math.PI / 2) / (Math.PI / 2));
    pos.push(
      Math.sin(phi) * Math.cos(theta) * rx * taper,
      Math.cos(phi) * ry,
      Math.sin(phi) * Math.sin(theta) * rz * taper
    );
    uv.push(u * 2, v * 2);
  };

  const rowLen = uSeg + 1;
  const gridSize = rowLen * (vSeg + 1);
  for (let v = 0; v <= vSeg; v++) for (let u = 0; u <= uSeg; u++) push(u / uSeg, v / vSeg, false);
  for (let v = 0; v <= vSeg; v++) for (let u = 0; u <= uSeg; u++) push(u / uSeg, v / vSeg, true);

  const quad = (a: number, b: number, c: number, d: number) => {
    idx.push(a, b, d, a, d, c);
  };

  for (const off of [0, gridSize]) {
    for (let v = 0; v < vSeg; v++) {
      for (let u = 0; u < uSeg; u++) {
        const a = off + v * rowLen + u;
        quad(a, a + 1, a + rowLen, a + rowLen + 1);
      }
    }
  }
  // Cut-edge rims (exposed wall thickness) down both sides of the opening
  for (let v = 0; v < vSeg; v++) {
    const o0 = v * rowLen;
    const i0 = gridSize + v * rowLen;
    quad(o0, i0, o0 + rowLen, i0 + rowLen);
    const o1 = v * rowLen + uSeg;
    const i1 = gridSize + v * rowLen + uSeg;
    quad(o1, i1, o1 + rowLen, i1 + rowLen);
  }
  // Rims at the base annulus and at the apex
  for (let u = 0; u < uSeg; u++) {
    quad(u, u + 1, gridSize + u, gridSize + u + 1);
    const vb = vSeg * rowLen;
    quad(vb + u, vb + u + 1, gridSize + vb + u, gridSize + vb + u + 1);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

const MYO_RADIUS = 1.35;

/** Maps a point on a sphere of radius MYO_RADIUS onto the heart's outer surface. */
function deformMyocardium(v: THREE.Vector3): THREE.Vector3 {
  const ny = v.y / MYO_RADIUS;
  const apexTaper = ny < 0 ? 1 + 1.8 * Math.pow(-ny, 1.6) : 1;
  let x = (v.x / apexTaper) * 0.85;
  const z = (v.z / apexTaper) * 0.95;
  x *= x < 0 ? 0.86 : 1.12; // LV side bulges more than the RV side
  return v.set(x, v.y * 1.15 - 0.15, z);
}

/** Intact external myocardium, used by the "Outside" (uncut) and risk views. */
export function makeMyocardiumGeometry(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(MYO_RADIUS, 64, 48);
  const p = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    deformMyocardium(v.fromBufferAttribute(p, i));
    p.setXYZ(i, v.x, v.y, v.z);
  }
  geo.computeVertexNormals();
  return geo;
}

/**
 * A point on the outer heart surface, by direction: polar 0 is the top,
 * PI the apex; azimuth PI/2 faces the viewer (anterior), smaller azimuths
 * turn toward the heart's left (viewer's right). `lift` floats the point
 * just off the surface so a vessel sits on the muscle rather than in it.
 */
export function myocardiumSurfacePoint(polar: number, azimuth: number, lift = 1.02): THREE.Vector3 {
  const r = MYO_RADIUS * lift;
  return deformMyocardium(
    new THREE.Vector3(Math.sin(polar) * Math.cos(azimuth) * r, Math.cos(polar) * r, Math.sin(polar) * Math.sin(azimuth) * r)
  );
}

export interface CoronarySpec {
  name: string;
  /** Path over the heart surface as [polar, azimuth] pairs (see myocardiumSurfacePoint). */
  path: [number, number][];
  radius: number;
  /** Territory the risk overlay weights most heavily (the LAD's). */
  primary: boolean;
}

// The coronary arteries on the outer surface. The left main splits into the
// left anterior descending (down the front groove between the ventricles to
// the apex, with a diagonal branch over the LV) and the circumflex (round
// the left side in the groove between atrium and ventricle). The right
// coronary runs round the right side in the same groove. Real coronary
// paths vary between people; these follow the common right-dominant pattern.
export const CORONARIES: CoronarySpec[] = [
  {
    name: "Left anterior descending artery",
    path: [[0.95, 1.4], [1.3, 1.5], [1.75, 1.52], [2.2, 1.5], [2.6, 1.42], [2.95, 1.3]],
    radius: 0.03,
    primary: true,
  },
  { name: "Diagonal branch", path: [[1.4, 1.48], [1.7, 1.25], [2.05, 1.08], [2.35, 1.0]], radius: 0.02, primary: true },
  { name: "Left circumflex artery", path: [[0.95, 1.4], [1.15, 1.1], [1.25, 0.6], [1.3, 0.0], [1.35, -0.6]], radius: 0.026, primary: false },
  { name: "Right coronary artery", path: [[1.05, 1.72], [1.3, 2.05], [1.38, 2.6], [1.45, 3.2], [1.6, 3.9]], radius: 0.028, primary: false },
  { name: "Marginal branch", path: [[1.36, 2.5], [1.8, 2.45], [2.2, 2.3]], radius: 0.018, primary: false },
];

/**
 * How strongly each outer-surface vertex belongs to the left anterior
 * descending artery's territory (front of the LV, front of the septum and
 * the apex) -- the territory the risk overlay shades most. 0..1.
 */
export function ladTerritoryWeight(p: THREE.Vector3): number {
  const smooth = (e0: number, e1: number, x: number) => {
    const t = THREE.MathUtils.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  };
  const anterior = smooth(-0.35, 0.45, p.z);
  const leftSide = smooth(-0.55, 0.15, p.x);
  const ventricular = smooth(0.55, 0.05, p.y);
  const apex = smooth(-0.9, -1.4, p.y);
  return Math.max(anterior * leftSide * ventricular, apex);
}

/** Small ear-like atrial appendage (auricle), as labelled in the reference plate. */
export function makeAuricleGeometry(): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(0.115, 20, 16);
  const p = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const lobe = 1 + 0.35 * Math.sin(v.y * 26); // scalloped, lobed edge
    p.setXYZ(i, v.x * 1.5 * lobe, v.y * 0.75, v.z * 0.7 * lobe);
  }
  geo.computeVertexNormals();
  return geo;
}

export interface ChamberSpec {
  name: "LA" | "RA" | "LV" | "RV";
  center: Vec3;
  radii: Vec3;
  thickness: number;
  gap: number;
  phiStart: number;
  apexTaper: number;
  colorOxygenated: boolean;
  poolOffset: number;
}

export const CHAMBERS: ChamberSpec[] = [
  {
    name: "RA",
    center: [-0.52, 0.46, 0.02],
    radii: [0.31, 0.3, 0.3],
    thickness: 0.035, // atrial walls are thin
    gap: 0.9,
    phiStart: 0.45,
    apexTaper: 0.1,
    colorOxygenated: false,
    poolOffset: -0.03,
  },
  {
    name: "LA",
    center: [0.48, 0.5, -0.06],
    radii: [0.29, 0.28, 0.28],
    thickness: 0.04,
    gap: 0.85,
    phiStart: 0.45,
    apexTaper: 0.1,
    colorOxygenated: true,
    poolOffset: -0.03,
  },
  {
    name: "RV",
    center: [-0.34, -0.26, 0.04],
    radii: [0.36, 0.52, 0.34],
    thickness: 0.055, // thin-walled, low-pressure pump
    gap: 0.9,
    phiStart: 0.55,
    apexTaper: 0.35,
    colorOxygenated: false,
    poolOffset: -0.04,
  },
  {
    name: "LV",
    center: [0.32, -0.38, -0.02],
    radii: [0.4, 0.64, 0.38],
    thickness: 0.14, // ~2.5x the RV: the single most recognisable wall difference
    gap: 0.85,
    phiStart: 0.5,
    apexTaper: 0.3,
    colorOxygenated: true,
    poolOffset: -0.05,
  },
];

export const AURICLES: { position: Vec3; rotation: Vec3; oxygenated: boolean }[] = [
  { position: [-0.76, 0.52, 0.16], rotation: [0, 0.5, 0.5], oxygenated: false },
  { position: [0.7, 0.56, 0.1], rotation: [0, -0.6, -0.5], oxygenated: true },
];

/** Papillary muscles and the chordae tendineae anchoring the AV valves. */
export const PAPILLARY: { position: Vec3; anchor: Vec3; chords: number }[] = [
  { position: [-0.38, -0.44, 0.04], anchor: [-0.45, 0.12, -0.02], chords: 5 },
  { position: [-0.22, -0.48, 0.0], anchor: [-0.45, 0.12, -0.02], chords: 4 },
  { position: [0.2, -0.62, -0.02], anchor: [0.41, 0.13, -0.05], chords: 5 },
  { position: [0.42, -0.58, -0.06], anchor: [0.41, 0.13, -0.05], chords: 5 },
];

export interface VesselSpec {
  name: string;
  points: Vec3[];
  radius: number;
  oxygenated: boolean;
}

// The aorta ascends from the LV, then arches POSTERIORLY and toward the
// patient's LEFT (the viewer's right), giving off -- right to left -- the
// brachiocephalic trunk, left common carotid and left subclavian, before
// descending behind the heart. The pulmonary trunk leaves the RV, crosses
// ANTERIOR to the ascending aorta and bifurcates beneath the arch: the
// left pulmonary artery runs to the viewer's right, the right pulmonary
// artery passes behind the ascending aorta and SVC to the viewer's left.
export const VESSELS: VesselSpec[] = [
  {
    name: "Superior vena cava",
    points: [
      [-0.6, 1.22, -0.02],
      [-0.58, 0.92, 0.0],
      [-0.54, 0.62, 0.0],
    ],
    radius: 0.1,
    oxygenated: false,
  },
  {
    name: "Inferior vena cava",
    points: [
      [-0.6, -0.95, -0.3],
      [-0.64, -0.35, -0.36],
      [-0.58, 0.22, -0.16],
    ],
    radius: 0.11,
    oxygenated: false,
  },
  {
    name: "Pulmonary trunk",
    points: [
      [-0.27, 0.2, 0.12],
      [-0.2, 0.5, 0.2],
      [0.0, 0.75, 0.16],
      [0.18, 0.86, 0.04],
    ],
    radius: 0.115,
    oxygenated: false,
  },
  {
    name: "Left pulmonary artery",
    points: [
      [0.18, 0.86, 0.04],
      [0.45, 0.9, -0.04],
      [0.8, 0.86, -0.14],
    ],
    radius: 0.075,
    oxygenated: false,
  },
  {
    name: "Right pulmonary artery",
    points: [
      [0.18, 0.86, 0.04],
      [-0.1, 0.9, -0.26],
      [-0.5, 0.86, -0.32],
      [-0.86, 0.8, -0.34],
    ],
    radius: 0.075,
    oxygenated: false,
  },
  {
    name: "Aorta",
    points: [
      [0.16, 0.22, -0.06],
      [0.1, 0.55, -0.02],
      [0.04, 0.95, -0.06],
      [0.12, 1.14, -0.2],
      [0.34, 1.1, -0.36],
      [0.46, 0.8, -0.54],
      [0.46, 0.2, -0.56],
      [0.42, -0.6, -0.54],
    ],
    radius: 0.105,
    oxygenated: true,
  },
  {
    name: "Brachiocephalic trunk",
    points: [
      [0.04, 1.12, -0.14],
      [-0.04, 1.44, -0.12],
    ],
    radius: 0.042,
    oxygenated: true,
  },
  {
    name: "Left common carotid artery",
    points: [
      [0.18, 1.15, -0.24],
      [0.18, 1.46, -0.22],
    ],
    radius: 0.034,
    oxygenated: true,
  },
  {
    name: "Left subclavian artery",
    points: [
      [0.3, 1.12, -0.32],
      [0.38, 1.42, -0.32],
    ],
    radius: 0.034,
    oxygenated: true,
  },
  {
    name: "Left pulmonary veins (upper)",
    points: [
      [0.95, 0.66, -0.26],
      [0.72, 0.6, -0.18],
      [0.6, 0.56, -0.12],
    ],
    radius: 0.058,
    oxygenated: true,
  },
  {
    name: "Left pulmonary veins (lower)",
    points: [
      [0.92, 0.38, -0.3],
      [0.72, 0.4, -0.22],
      [0.58, 0.44, -0.14],
    ],
    radius: 0.058,
    oxygenated: true,
  },
  {
    name: "Right pulmonary veins",
    points: [
      [-0.86, 0.5, -0.42],
      [-0.3, 0.5, -0.38],
      [0.24, 0.5, -0.24],
    ],
    radius: 0.058,
    oxygenated: true,
  },
];

// One valve leaflet/cusp: a petal hinged along local y=0 (the annulus
// edge), tapering to a rounded free edge at y=length -- rotating the
// resulting mesh around its own local X axis folds it away from the flat,
// flow-blocking "closed" position, the same hinge kinematics real valve
// cusps use.
export function makeLeafletGeometry(length = 0.075, width = 0.07): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  shape.moveTo(-width / 2, 0);
  shape.quadraticCurveTo(-width / 2, length, 0, length);
  shape.quadraticCurveTo(width / 2, length, width / 2, 0);
  shape.lineTo(-width / 2, 0);
  return new THREE.ShapeGeometry(shape, 8);
}

export interface ValveSpec {
  name: "tricuspid" | "mitral" | "pulmonic" | "aortic";
  position: Vec3;
  rotation: Vec3;
  leaflets: number;
  scale: number;
}

// rotation orients each valve's flow axis (local Z of its leaflet fan)
// roughly vertically -- a simplification (real valve planes are tilted)
// stated in README.md. The pulmonic and aortic valves sit at the roots of
// the pulmonary trunk and aorta above.
export const VALVES: ValveSpec[] = [
  { name: "tricuspid", position: [-0.45, 0.14, -0.02], rotation: [-Math.PI / 2, 0, 0.3], leaflets: 3, scale: 1.5 },
  { name: "mitral", position: [0.41, 0.15, -0.05], rotation: [-Math.PI / 2, 0, -0.2], leaflets: 2, scale: 1.4 },
  { name: "pulmonic", position: [-0.27, 0.2, 0.1], rotation: [-Math.PI / 2, 0, 0.6], leaflets: 3, scale: 1.05 },
  { name: "aortic", position: [0.19, 0.2, -0.06], rotation: [-Math.PI / 2, 0, -0.6], leaflets: 3, scale: 1.05 },
];

// SA node -> AV node -> Bundle of His -> bundle branches -> Purkinje
// fibers, positioned relative to real conduction-system anatomy: SA node
// high on the right atrium, AV node at the atrioventricular junction,
// bundle branches running down either side of the septum to the apex.
export const CONDUCTION_PATH: { name: string; position: Vec3; tAt: number }[] = [
  { name: "SA node", position: [-0.6, 0.72, 0.12], tAt: 0.0 },
  { name: "AV node", position: [-0.12, 0.1, 0.02], tAt: 0.1 },
  { name: "Bundle of His", position: [-0.02, -0.12, 0.0], tAt: 0.12 },
  { name: "Left bundle branch", position: [0.18, -0.55, -0.03], tAt: 0.135 },
  { name: "Right bundle branch", position: [-0.2, -0.48, 0.04], tAt: 0.135 },
  { name: "Purkinje fibers (LV apex)", position: [0.28, -0.92, -0.03], tAt: 0.15 },
  { name: "Purkinje fibers (RV apex)", position: [-0.3, -0.72, 0.04], tAt: 0.15 },
];

export function buildFlowCurve(points: Vec3[]): THREE.CatmullRomCurve3 {
  return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
}
