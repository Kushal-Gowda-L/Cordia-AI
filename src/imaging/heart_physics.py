"""Multi-physics heart twin on the patient's own MRI mesh: electrical
activation, muscle (wall thickness and thickening), blood pressure and flow
(pressure-volume loop), and epicardial fat.

Everything here is a lightweight, illustrative model of the kind used in
teaching and in early-stage digital-twin research. None of it is a clinical
simulation or validated against this patient. Each part says what it
assumes:

Electrical (eikonal-style activation map)
  The left-ventricular myocardium mesh becomes a graph (mesh edges plus every
  pair of vertices within GRAPH_RADIUS_MM, so the wave can cross the wall).
  Each edge's travel time uses an anisotropic conduction velocity: faster
  along the muscle fibre than across it. Fibres follow the rule-based helix
  of Streeter et al. (1969): +60 deg at the endocardium turning to -60 deg
  at the epicardium. A fast endocardial layer stands in for the Purkinje
  network. Activation starts at three left-septal sites (Durrer et al.,
  1970) and the earliest arrival time at every vertex is found with
  Dijkstra's algorithm -- the graph form of the eikonal equation. The right
  ventricle's muscle is not segmented by ACDC, so only the LV is modelled.

Muscle
  Wall thickness at every vertex from the mesh (distance from the
  endocardial to the epicardial surface). Measured systolic wall thickening
  needs the end-systolic segmentation too: ray casting from the cavity
  centre on each short-axis slice at ED and ES, reported per AHA-style
  segment and mapped onto the mesh.

Blood flow (0D lumped-parameter circulation)
  Time-varying elastance left ventricle and left atrium, mitral and aortic
  valves as one-way resistors, three-element Windkessel aorta, fixed
  pulmonary-venous filling pressure. The ventricle's elastance is the
  average of every vertex's local twitch started at its own activation time,
  so a slower or dyssynchronous activation (LBBB, scar) really does weaken
  the beat -- the electromechanical coupling. The model is first calibrated
  to the patient's measured EDV, ESV, blood pressure and heart rate, then the
  what-if changes are applied with that calibration held fixed.

Fat
  Epicardial adipose tissue volume from CT is the voxels inside the
  pericardial sac with -190 to -30 HU (the standard window). That needs a
  CT with a pericardium segmentation, which CordiaAI's CT branch (coronary
  lumen only) does not produce, so without one the twin says "not measured".

Out of scope, stated in README: 3D finite-element mechanics (stress/strain
in the wall), monodomain/bidomain ionic electrophysiology, 3D CFD of the
blood inside the chambers, and the right heart and atria as 3D structures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ILLUSTRATIVE_NOTE = (
    "Illustrative model, not a clinical simulation. Built on the patient's MRI shape and calibrated to "
    "their measured volumes and blood pressure; the physics are simplified and not validated for this patient."
)

# ------------------------------------------------------------------ constants
GRAPH_RADIUS_MM = 6.0
MIDWALL_FRACTIONS = (0.25, 0.5, 0.75)
ENDO_DISTANCE_MM = 2.5  # myocardium vertices this close to the LV cavity surface are endocardial
HELIX_ENDO_DEG, HELIX_EPI_DEG = 60.0, -60.0  # Streeter rule-based fibres
# mm/ms (= m/s). The upper end of measured ventricular values: zig-zag paths on a point graph run
# a little longer than straight ones, and with these the LBBB case lands in the reported
# 150-200 ms LV activation range and the normal case at 60-80 ms.
CV_FIBRE, CV_CROSS, CV_PURKINJE = 0.8, 0.4, 2.5
PURKINJE_MAX_LEVEL = 0.85  # the Purkinje network thins out towards the base
MMHG_ML_TO_J = 1.333e-4

# Durrer's earliest LV breakthrough sites: (angle from septum in degrees, level 0 = apex .. 1 = base)
LV_SEPTAL_SITES = [(0.0, 0.5), (55.0, 0.6), (-55.0, 0.35)]
CONDUCTION_MODES = ("normal", "lbbb", "rv_pacing", "crt")
RIGHT_BUNDLE_DELAY_MS = 15.0  # RV septal surface activation in LBBB / start of RV pacing


# ------------------------------------------------------------------ geometry
def mesh_volume_ml(mesh: dict) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem), in ml."""
    v = np.asarray(mesh["vertices"], dtype=float)
    f = np.asarray(mesh["faces"], dtype=int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0 / 1000.0)


@dataclass
class LVGeometry:
    vertices: np.ndarray  # (N, 3) mm: the myocardium mesh's vertices first, then mid-wall points
    faces: np.ndarray
    axis: np.ndarray  # unit vector apex -> base
    centre: np.ndarray  # a point on the long axis
    level: np.ndarray  # 0 apex .. 1 base
    angle_deg: np.ndarray  # around the long axis, 0 = towards the septum (RV)
    transmural: np.ndarray  # 0 endocardium .. 1 epicardium
    thickness_mm: np.ndarray
    radial: np.ndarray  # (N, 3) unit, outward from the axis
    fibre: np.ndarray  # (N, 3) unit fibre direction
    helix_deg: np.ndarray
    edges: np.ndarray  # (E, 2)
    edge_vec: np.ndarray  # (E, 3)
    extra: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def build_geometry(meshes: dict, slice_axis: int = 2) -> LVGeometry:
    """meshes: {"LV myocardium", "LV cavity", optional "RV cavity"} as returned by label_meshes.
    The long axis is the short-axis stack's slice normal (ACDC is acquired that way)."""
    myo = np.asarray(meshes["LV myocardium"]["vertices"], dtype=float)
    faces = np.asarray(meshes["LV myocardium"]["faces"], dtype=int)
    cav = np.asarray(meshes["LV cavity"]["vertices"], dtype=float)

    axis = np.zeros(3)
    axis[slice_axis] = 1.0
    centre = cav.mean(0)
    # Base is the end where the cavity is wider (the ventricle tapers to the apex).
    h_cav = (cav - centre) @ axis
    radial_extent = np.linalg.norm((cav - centre) - np.outer(h_cav, axis), axis=1)
    lo, hi = np.quantile(h_cav, [0.25, 0.75])
    if radial_extent[h_cav <= lo].mean() > radial_extent[h_cav >= hi].mean():
        axis = -axis

    # Endocardium vs epicardium, then local wall thickness.
    d_cav, _ = cKDTree(cav).query(myo)
    endo_s = d_cav < ENDO_DISTANCE_MM
    if endo_s.all() or not endo_s.any():
        raise ValueError("Could not tell the endocardial from the epicardial surface of this mesh")
    d_endo, i_endo = cKDTree(myo[endo_s]).query(myo)
    d_epi, _ = cKDTree(myo[~endo_s]).query(myo)
    thickness_s = np.where(endo_s, d_epi, d_endo)
    transmural_s = np.where(endo_s, 0.0, 1.0)

    # The mesh is two surfaces. Fill the wall with points between each epicardial vertex and its
    # nearest endocardial one, so the wave can travel through mid-wall muscle, where the fibres
    # run circumferentially and conduction around the ventricle is fastest.
    epi_idx = np.flatnonzero(~endo_s)
    a, b = myo[endo_s][i_endo[epi_idx]], myo[epi_idx]
    inner = [a + f * (b - a) for f in MIDWALL_FRACTIONS]
    pts = np.concatenate([myo, *inner])
    transmural = np.concatenate([transmural_s, np.repeat(MIDWALL_FRACTIONS, len(epi_idx))])
    thickness = np.concatenate([thickness_s, np.tile(thickness_s[epi_idx], len(MIDWALL_FRACTIONS))])
    endo = np.concatenate([endo_s, np.zeros(len(epi_idx) * len(MIDWALL_FRACTIONS), bool)])

    h = (pts - centre) @ axis
    level = (h - h.min()) / max(h.max() - h.min(), 1e-6)
    rel = (pts - centre) - np.outer(h, axis)
    radial = _unit(rel)
    if "RV cavity" in meshes:
        rv = np.asarray(meshes["RV cavity"]["vertices"], dtype=float).mean(0) - centre
    else:  # no RV: pick any in-plane direction, the septum label is then arbitrary
        rv = np.cross(axis, [1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.cross(axis, [0, 1.0, 0])
    septum = _unit(rv - (rv @ axis) * axis)
    other = np.cross(axis, septum)
    angle = np.degrees(np.arctan2(radial @ other, radial @ septum))

    helix = HELIX_ENDO_DEG + (HELIX_EPI_DEG - HELIX_ENDO_DEG) * transmural
    circ = _unit(np.cross(axis[None, :], radial))
    hr = np.radians(helix)[:, None]
    fibre = _unit(np.cos(hr) * circ + np.sin(hr) * axis[None, :])

    pairs = cKDTree(pts).query_pairs(GRAPH_RADIUS_MM, output_type="ndarray")
    mesh_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(np.concatenate([pairs, mesh_edges]), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return LVGeometry(pts, faces, axis, centre, level, angle, transmural, thickness, radial, fibre, helix,
                      edges, pts[edges[:, 1]] - pts[edges[:, 0]],
                      {"endo": endo, "septum": septum, "n_surface": len(myo)})


# ------------------------------------------------------------------ electrical
def _site(geo: LVGeometry, angle_deg: float, level: float, surface: str) -> int:
    """Vertex nearest to (angle, level) on the endocardial or epicardial surface."""
    on = geo.extra["endo"] if surface == "endo" else geo.transmural >= 1.0
    dang = np.radians((geo.angle_deg - angle_deg + 180) % 360 - 180)
    score = dang**2 + (2.0 * (geo.level - level)) ** 2 + np.where(on, 0, 1e6)
    return int(np.argmin(score))


def scar_mask(geo: LVGeometry, angle_deg: float, level: float, radius_mm: float) -> np.ndarray:
    """A transmural scar: every wall vertex within radius_mm of a point on the epicardium."""
    centre = geo.vertices[_site(geo, angle_deg, level, "epi")]
    d = np.linalg.norm(geo.vertices - centre, axis=1)
    # also take the endocardial side beneath it
    return d <= radius_mm + float(np.median(geo.thickness_mm))


def activation_times(geo: LVGeometry, mode: str = "normal", cv_scale: float = 1.0,
                     scar: np.ndarray | None = None, scar_cv_factor: float = 0.15,
                     crt_site: tuple[float, float] = (180.0, 0.5)) -> np.ndarray:
    """Earliest activation time (ms, 0 = first breakthrough) at every myocardium vertex.

    normal    -- three left-septal Purkinje breakthroughs plus a fast endocardial layer
    lbbb      -- left bundle blocked: the wave enters from the RV side of the septum and
                 crosses the LV through working muscle only (no fast layer)
    rv_pacing -- paced from the RV apex (septal side, near the apex), no fast layer
    crt       -- LBBB plus a second, LV lateral epicardial pacing lead (cardiac
                 resynchronisation therapy)
    """
    if mode not in CONDUCTION_MODES:
        raise ValueError(f"mode must be one of {CONDUCTION_MODES}")
    e, ev = geo.edges, geo.edge_vec
    length2 = np.einsum("ij,ij->i", ev, ev)
    along2 = 0.5 * ((np.einsum("ij,ij->i", ev, geo.fibre[e[:, 0]])) ** 2 + (np.einsum("ij,ij->i", ev, geo.fibre[e[:, 1]])) ** 2)
    along2 = np.minimum(along2, length2)
    cvf, cvc = CV_FIBRE * cv_scale, CV_CROSS * cv_scale
    t_edge = np.sqrt(along2 / cvf**2 + (length2 - along2) / cvc**2)

    endo = geo.extra["endo"]
    if mode == "normal":
        fast = endo[e[:, 0]] & endo[e[:, 1]] & (geo.level[e[:, 0]] < PURKINJE_MAX_LEVEL) & (geo.level[e[:, 1]] < PURKINJE_MAX_LEVEL)
        t_edge = np.where(fast, np.minimum(t_edge, np.sqrt(length2) / (CV_PURKINJE * cv_scale)), t_edge)
    if scar is not None and scar.any():
        slow = scar[e[:, 0]] | scar[e[:, 1]]
        t_edge = np.where(slow, t_edge / scar_cv_factor, t_edge)

    seeds: list[tuple[int, float]] = []
    if mode == "normal":
        seeds = [(_site(geo, a, lv, "endo"), 0.0) for a, lv in LV_SEPTAL_SITES]
    elif mode in ("lbbb", "crt"):
        seeds = [(_site(geo, 0.0, 0.5, "epi"), RIGHT_BUNDLE_DELAY_MS), (_site(geo, 0.0, 0.3, "epi"), RIGHT_BUNDLE_DELAY_MS)]
        if mode == "crt":
            seeds.append((_site(geo, crt_site[0], crt_site[1], "epi"), RIGHT_BUNDLE_DELAY_MS))
    elif mode == "rv_pacing":
        seeds = [(_site(geo, 0.0, 0.1, "epi"), 0.0)]

    n = len(geo.vertices)
    src = n  # a virtual source joined to each seed by an edge of length = its start time
    rows = np.concatenate([e[:, 0], e[:, 1], [src] * len(seeds)])
    cols = np.concatenate([e[:, 1], e[:, 0], [s for s, _ in seeds]])
    vals = np.concatenate([t_edge, t_edge, [max(d, 1e-6) for _, d in seeds]])
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    at = dijkstra(graph, directed=True, indices=src)[:n]
    at = at - np.nanmin(at[np.isfinite(at)])
    return at


def repolarisation_times(geo: LVGeometry, at: np.ndarray, heart_rate: float) -> np.ndarray:
    """Activation + action potential duration. APD is longer at the endocardium than the
    epicardium (the transmural gradient behind the upright T wave) and shortens with heart rate
    (Bazett-style sqrt(RR) scaling)."""
    rr_s = 60.0 / heart_rate
    apd = (290.0 - 40.0 * geo.transmural) * math.sqrt(rr_s)
    return at + apd


# ------------------------------------------------------------------ muscle (measured)
def ray_wall_thickness(mask2d: np.ndarray, spacing2d, myo_label: int, cav_label: int,
                       n_angles: int = 72) -> np.ndarray | None:
    """Wall thickness (mm) along n_angles rays cast from the LV cavity centre on one slice."""
    cav = mask2d == cav_label
    myo = mask2d == myo_label
    if cav.sum() < 5 or myo.sum() < 5:
        return None
    ci, cj = np.argwhere(cav).mean(0)
    step = 0.25  # pixels
    r = np.arange(0, max(mask2d.shape), step)
    out = np.zeros(n_angles)
    for k, th in enumerate(np.linspace(0, 2 * np.pi, n_angles, endpoint=False)):
        ii = np.round(ci + r * np.cos(th)).astype(int)
        jj = np.round(cj + r * np.sin(th)).astype(int)
        ok = (ii >= 0) & (ii < mask2d.shape[0]) & (jj >= 0) & (jj < mask2d.shape[1])
        hits = myo[ii[ok], jj[ok]]
        pix = math.hypot(spacing2d[0] * math.cos(th), spacing2d[1] * math.sin(th))
        out[k] = hits.sum() * step * pix
    return out


AHA_RINGS = {"basal": ["anteroseptal", "anterior", "anterolateral", "inferolateral", "inferior", "inferoseptal"],
             "mid": ["anteroseptal", "anterior", "anterolateral", "inferolateral", "inferior", "inferoseptal"],
             "apical": ["septal", "anterior", "lateral", "inferior"]}


def regional_thickening(ed_mask: np.ndarray, es_mask: np.ndarray, spacing_mm, myo_label: int = 2,
                        cav_label: int = 3, rv_label: int = 1, n_angles: int = 72) -> dict:
    """Measured wall thickness at ED and ES on every short-axis slice, the systolic thickening
    (ES - ED) / ED, and the AHA-style 16-segment summary.

    Segments are anchored on the RV centre (the septum faces it). Which way round 'anterior' and
    'inferior' fall depends on the image orientation, so they may be mirrored."""
    per_slice = []
    for k in range(ed_mask.shape[-1]):
        ted = ray_wall_thickness(ed_mask[..., k], spacing_mm[:2], myo_label, cav_label, n_angles)
        tes = ray_wall_thickness(es_mask[..., k], spacing_mm[:2], myo_label, cav_label, n_angles)
        if ted is None or tes is None:
            per_slice.append(None)
            continue
        cav = np.argwhere(ed_mask[..., k] == cav_label).mean(0)
        rv_pts = np.argwhere(ed_mask[..., k] == rv_label)
        if len(rv_pts):
            d = (rv_pts.mean(0) - cav) * np.asarray(spacing_mm[:2])
            septum_angle = math.degrees(math.atan2(d[1], d[0]))
        else:
            septum_angle = 0.0
        thick = np.where(ted > 0, 100.0 * (tes - ted) / np.maximum(ted, 1e-6), np.nan)
        per_slice.append({"slice": k, "centre_px": cav.tolist(), "septum_angle_deg": septum_angle,
                          "ed_mm": ted.round(2).tolist(), "es_mm": tes.round(2).tolist(),
                          "thickening_pct": np.round(thick, 1).tolist()})

    valid = [s for s in per_slice if s is not None]
    segments = []
    if valid:
        # base = the end where the cavity is largest
        areas = [(ed_mask[..., s["slice"]] == cav_label).sum() for s in valid]
        order = valid if areas[0] >= areas[-1] else valid[::-1]
        thirds = np.array_split(np.arange(len(order)), 3)
        for ring, idx in zip(("basal", "mid", "apical"), thirds):
            names = AHA_RINGS[ring]
            n_sec = len(names)
            acc = {nm: ([], []) for nm in names}
            for i in idx:
                s = order[i]
                for a, (ed_t, th) in enumerate(zip(s["ed_mm"], s["thickening_pct"])):
                    ang = (360.0 * a / n_angles - s["septum_angle_deg"]) % 360
                    sec = int(((ang + 360.0 / n_sec / 2) % 360) // (360.0 / n_sec))
                    acc[names[sec]][0].append(ed_t)
                    if th == th:
                        acc[names[sec]][1].append(th)
            for nm in names:
                eds, ths = acc[nm]
                segments.append({"ring": ring, "segment": nm,
                                 "ed_thickness_mm": round(float(np.mean(eds)), 1) if eds else None,
                                 "thickening_pct": round(float(np.median(ths)), 1) if ths else None})
    return {"slices": per_slice, "segments": segments,
            "note": "Measured from the ED and ES segmentations by ray casting from the cavity centre. "
                    "Normal systolic thickening is roughly 30-60%; below ~10% suggests a hypokinetic segment."}


def thickening_at_vertices(vertices: np.ndarray, regional: dict, spacing_mm, slice_axis: int = 2) -> list:
    """Map the per-slice, per-angle thickening onto mesh vertices (nearest slice and ray)."""
    slices = {s["slice"]: s for s in regional["slices"] if s is not None}
    out = []
    for v in np.asarray(vertices, dtype=float):
        k = int(round(v[slice_axis] / spacing_mm[slice_axis]))
        if not slices:
            out.append(None)
            continue
        k = min(slices, key=lambda s: abs(s - k))
        s = slices[k]
        di, dj = v[0] / spacing_mm[0] - s["centre_px"][0], v[1] / spacing_mm[1] - s["centre_px"][1]
        ang = math.atan2(dj * spacing_mm[1], di * spacing_mm[0]) % (2 * math.pi)
        n = len(s["thickening_pct"])
        val = s["thickening_pct"][int(round(ang / (2 * math.pi) * n)) % n]
        out.append(None if val != val else val)
    return out


# ------------------------------------------------------------------ fat
EAT_HU = (-190.0, -30.0)


def epicardial_fat(ct_hu: np.ndarray, pericardium_mask: np.ndarray, spacing_mm) -> dict:
    """Epicardial adipose tissue: voxels inside the pericardial sac in the fat HU window."""
    inside = pericardium_mask.astype(bool)
    fat = inside & (ct_hu >= EAT_HU[0]) & (ct_hu <= EAT_HU[1])
    vox_ml = float(np.prod(spacing_mm)) / 1000.0
    return {"eat_volume_ml": round(float(fat.sum() * vox_ml), 1),
            "eat_mean_hu": round(float(ct_hu[fat].mean()), 1) if fat.any() else None,
            "method": "Voxels inside the pericardium with -190 to -30 HU", "measured": True}


FAT_NOT_MEASURED = {
    "measured": False,
    "reason": "Epicardial fat is measured on CT inside the pericardium. CordiaAI's CT branch segments the "
              "coronary lumen only, and cine MRI does not separate fat this way, so fat is not measured. "
              "The viewer's fat layer is an illustrative drawing, not this patient's fat.",
}


# ------------------------------------------------------------------ blood flow (0D)
@dataclass
class Circulation:
    hr: float = 72.0
    ees: float = 2.3  # mmHg/ml, end-systolic elastance (contractility)
    v0: float = 10.0  # ml, ESPVR intercept
    edp_a: float = 0.3  # mmHg, EDPVR P = A (exp(B (V - V0)) - 1)
    edp_b: float = 0.025  # 1/ml
    r_sys: float = 1.0  # mmHg s/ml, systemic resistance
    c_art: float = 1.4  # ml/mmHg, arterial compliance
    z_c: float = 0.04  # mmHg s/ml, aortic characteristic impedance
    r_av: float = 0.003
    r_mv: float = 0.008
    r_pv: float = 0.01
    p_pv: float = 10.0  # mmHg, pulmonary venous (filling) pressure
    p_ven: float = 0.0
    la_emin: float = 0.15
    la_emax: float = 0.45
    la_v0: float = 4.0
    pr_ms: float = 160.0


def twitch(tau: np.ndarray, rr_s: float) -> np.ndarray:
    """Normalised local active elastance after activation (raised-cosine rise and fall)."""
    tp = 0.27 * math.sqrt(rr_s)
    tr = 0.5 * tp
    out = np.zeros_like(tau)
    rise = (tau >= 0) & (tau < tp)
    fall = (tau >= tp) & (tau < tp + tr)
    out[rise] = 0.5 * (1 - np.cos(np.pi * tau[rise] / tp))
    out[fall] = 0.5 * (1 + np.cos(np.pi * (tau[fall] - tp) / tr))
    return out


def ventricular_activation_curve(at_ms: np.ndarray, contract: np.ndarray, hr: float, dt: float) -> np.ndarray:
    """Global LV activation e(t) over one beat: the mean of every vertex's twitch started at its
    own activation time. Non-activated or non-contracting vertices add nothing."""
    rr = 60.0 / hr
    t = np.arange(0, rr, dt)
    ok = np.isfinite(at_ms) & (contract > 0)
    if not ok.any():
        return np.zeros_like(t)
    # histogram of activation times, then convolve with the twitch: cheaper than per-vertex
    bins = np.minimum(np.floor(at_ms[ok] / 1000.0 / dt).astype(int), len(t) - 1)
    weights = np.bincount(bins, weights=contract[ok], minlength=len(t))[: len(t)]
    kernel = twitch(t, rr)
    e = np.convolve(weights, kernel)[: len(t)]
    # wrap the tail (a long activation can spill into the next beat)
    tail = np.convolve(weights, kernel)[len(t):]
    e[: len(tail)] += tail[: len(t)]
    return e / len(at_ms)


def simulate_circulation(p: Circulation, e_lv: np.ndarray, dt: float, beats: int = 12) -> dict:
    """Integrate the left-heart circulation to steady state; return the last beat."""
    rr = 60.0 / p.hr
    n = len(e_lv)
    t_beat = np.arange(n) * dt
    tau_a = (t_beat - (rr - p.pr_ms / 1000.0)) % rr
    e_la = twitch(tau_a, 0.11)  # atrial twitch: ~90 ms rise, ~45 ms relaxation
    v_lv, v_la, p_c = 120.0, 60.0, 80.0
    rec = {k: np.zeros(n) for k in ("v_lv", "p_lv", "p_ao", "p_la", "q_av", "q_mv")}
    for beat in range(beats):
        last = beat == beats - 1
        for i in range(n):
            e = e_lv[i]
            p_ed = p.edp_a * (math.exp(min(p.edp_b * (v_lv - p.v0), 50)) - 1.0)
            p_lv = e * p.ees * (v_lv - p.v0) + (1 - e) * p_ed
            p_la = (p.la_emin + (p.la_emax - p.la_emin) * e_la[i]) * (v_la - p.la_v0)
            q_mv = (p_la - p_lv) / p.r_mv if p_la > p_lv else 0.0
            q_av = (p_lv - p_c) / (p.r_av + p.z_c) if p_lv > p_c else 0.0
            q_pv = (p.p_pv - p_la) / p.r_pv
            if last:
                rec["v_lv"][i], rec["p_lv"][i], rec["p_ao"][i] = v_lv, p_lv, p_c + p.z_c * q_av
                rec["p_la"][i], rec["q_av"][i], rec["q_mv"][i] = p_la, q_av, q_mv
            v_lv += dt * (q_mv - q_av)
            v_la += dt * (q_pv - q_mv)
            p_c += dt * (q_av - (p_c - p.p_ven) / p.r_sys) / p.c_art
            v_lv = max(v_lv, 1.0)
            v_la = max(v_la, 1.0)
    rec["t_ms"] = t_beat * 1000.0
    return rec


def summarise(rec: dict, hr: float, dt: float) -> dict:
    v, p, pao = rec["v_lv"], rec["p_lv"], rec["p_ao"]
    edv, esv = float(v.max()), float(v.min())
    sv = edv - esv
    work = 0.5 * abs(float(np.dot(v, np.roll(p, -1)) - np.dot(p, np.roll(v, -1))))
    q_mv = rec["q_mv"]
    # E wave: early filling peak; A wave: atrial-kick peak in the last part of diastole
    n = len(q_mv)
    late = int(n * 0.75)
    e_peak, a_peak = float(q_mv[:late].max()), float(q_mv[late:].max())
    return {
        "edv_ml": round(edv, 1), "esv_ml": round(esv, 1), "sv_ml": round(sv, 1),
        "ef_percent": round(100 * sv / edv, 1) if edv > 0 else None,
        "sbp_mmhg": round(float(pao.max()), 1), "dbp_mmhg": round(float(pao.min()), 1),
        "map_mmhg": round(float(pao.mean()), 1),
        "lv_peak_pressure_mmhg": round(float(p.max()), 1),
        "cardiac_output_l_min": round(sv * hr / 1000.0, 2),
        "stroke_work_j": round(work * MMHG_ML_TO_J, 3),
        "dpdt_max_mmhg_s": round(float(np.max(np.diff(p)) / dt), 0),
        "mitral_e_a_ratio": round(e_peak / a_peak, 2) if a_peak > 1 else None,
        "peak_aortic_flow_ml_s": round(float(rec["q_av"].max()), 0),
        "mean_la_pressure_mmhg": round(float(rec["p_la"].mean()), 1),
    }


def calibrate(targets: dict, e_lv: np.ndarray, dt: float, iterations: int = 12) -> tuple[Circulation, dict]:
    """Fit contractility, filling pressure, resistance and compliance so the simulated beat
    matches the measured EDV, ESV, systolic/diastolic BP and heart rate."""
    edv, esv = targets["edv_ml"], targets["esv_ml"]
    sbp, dbp, hr = targets["sbp_mmhg"], targets["dbp_mmhg"], targets["hr_bpm"]
    sv = max(edv - esv, 5.0)
    map_t = dbp + (sbp - dbp) / 3.0
    p = Circulation(hr=hr)
    p.edp_a = 8.0 / (math.exp(p.edp_b * (edv - p.v0)) - 1.0)  # end-diastolic pressure ~8 mmHg at the measured EDV
    p.ees = 0.9 * sbp / max(esv - p.v0, 5.0) / max(e_lv.max(), 0.3)
    p.r_sys = map_t / (sv * hr / 60.0)
    p.c_art = 0.8 * sv / max(sbp - dbp, 10.0)
    for _ in range(iterations):
        s = summarise(simulate_circulation(p, e_lv, dt, beats=8), hr, dt)
        p.ees *= _clamp((s["esv_ml"] - p.v0) / max(esv - p.v0, 2.0), 0.6, 1.6)
        p.p_pv = _clamp(p.p_pv * (edv / s["edv_ml"]) ** 1.5, 2.0, 40.0)
        # compare like with like: the clinical MAP formula on both sides
        map_m = s["dbp_mmhg"] + (s["sbp_mmhg"] - s["dbp_mmhg"]) / 3.0
        p.r_sys *= _clamp(map_t / map_m, 0.6, 1.6)
        p.c_art *= _clamp(((s["sbp_mmhg"] - s["dbp_mmhg"]) / max(sbp - dbp, 5.0)) ** 0.8, 0.6, 1.6)
    s = summarise(simulate_circulation(p, e_lv, dt), hr, dt)
    fit = {k: {"target": round(float(targets[t]), 1), "model": s[k]}
           for k, t in (("edv_ml", "edv_ml"), ("esv_ml", "esv_ml"), ("sbp_mmhg", "sbp_mmhg"), ("dbp_mmhg", "dbp_mmhg"))}
    return p, fit


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------------------------------------------ the whole twin
DEFAULT_WHATIF = {
    "heart_rate": None,  # bpm; None keeps the measured rate
    "contractility": 1.0,  # x calibrated end-systolic elastance
    "resistance": 1.0,  # x systemic vascular resistance (afterload)
    "arterial_stiffness": 1.0,  # x 1/compliance
    "filling_pressure": 1.0,  # x pulmonary venous pressure (preload)
    "diastolic_stiffness": 1.0,  # x EDPVR stiffness (e.g. hypertrophy, HFpEF)
    "aortic_stenosis": 0.0,  # 0 none .. 1 severe
    "conduction": "normal",
    "conduction_velocity": 1.0,  # x conduction velocity (diffuse fibrosis < 1)
    "scar": None,  # {"angle_deg": 180, "level": 0.5, "radius_mm": 15}
}


def run_twin(meshes: dict, targets: dict, whatif: dict | None = None, dt: float = 0.0005,
             geometry: LVGeometry | None = None, calibration: tuple | None = None) -> dict:
    """targets: {"edv_ml", "esv_ml", "sbp_mmhg", "dbp_mmhg", "hr_bpm"} measured for the patient.
    Calibrates on the patient's own (normal-conduction) activation, then applies the what-ifs."""
    w = {**DEFAULT_WHATIF, **(whatif or {})}
    geo = geometry or build_geometry(meshes)
    hr0 = targets["hr_bpm"]
    hr = float(w["heart_rate"] or hr0)

    if calibration is None:
        at0 = activation_times(geo, "normal")
        e0 = ventricular_activation_curve(at0, np.ones(len(at0)), hr0, dt)
        calibration = calibrate(targets, e0, dt)
    base, fit = calibration

    scar = None
    if w["scar"]:
        sc = w["scar"]
        scar = scar_mask(geo, float(sc.get("angle_deg", 180)), float(sc.get("level", 0.5)), float(sc.get("radius_mm", 15)))
    at = activation_times(geo, w["conduction"], float(w["conduction_velocity"]), scar)
    contract = np.ones(len(at)) if scar is None else np.where(scar, 0.0, 1.0)
    rt = repolarisation_times(geo, at, hr)

    p = Circulation(**{k: getattr(base, k) for k in Circulation.__dataclass_fields__})
    p.hr = hr
    p.ees *= w["contractility"]
    p.r_sys *= w["resistance"]
    p.c_art /= w["arterial_stiffness"]
    p.p_pv *= w["filling_pressure"]
    p.edp_a *= w["diastolic_stiffness"]
    p.r_av *= 1.0 + 60.0 * float(w["aortic_stenosis"]) ** 2
    # Faster heart rates shorten the PR interval a little
    p.pr_ms = base.pr_ms * math.sqrt(60.0 / hr / (60.0 / hr0))
    e_lv = ventricular_activation_curve(at, contract, hr, dt)
    rec = simulate_circulation(p, e_lv, dt)
    summary = summarise(rec, hr, dt)

    ns = geo.extra["n_surface"]
    finite = np.isfinite(at)
    healthy = finite & (contract > 0)
    surf = slice(0, ns)
    step = max(1, int(0.004 / dt))  # 4 ms samples for the browser
    return {
        "note": ILLUSTRATIVE_NOTE,
        "whatif": w,
        "electrical": {
            "activation_ms": np.where(finite, np.round(at, 1), -1)[surf].tolist(),
            "repolarisation_ms": np.where(finite, np.round(rt, 1), -1)[surf].tolist(),
            "lv_total_activation_ms": round(float(at[healthy].max()), 1),
            "qrs_estimate_ms": round(float(at[healthy].max()) + 10.0, 0),
            "lv_repolarisation_end_ms": round(float(rt[healthy].max()), 0),
            "mode": w["conduction"],
            "note": "LV only (ACDC does not segment RV muscle). QRS estimate = LV total activation of the "
                    "healthy muscle + 10 ms; a real QRS also includes the RV.",
        },
        "muscle": {
            "wall_thickness_mm": np.round(geo.thickness_mm[surf], 1).tolist(),
            "transmural": np.round(geo.transmural[surf], 2).tolist(),
            "helix_angle_deg": np.round(geo.helix_deg[surf], 0).tolist(),
            "level": np.round(geo.level[surf], 3).tolist(),
            "angle_deg": np.round(geo.angle_deg[surf], 1).tolist(),
            "radial": np.round(geo.radial[surf], 3).tolist(),
            "axis": geo.axis.round(4).tolist(),
            "centre": geo.centre.round(2).tolist(),
            "contracts": contract[surf].astype(int).tolist(),
            "median_wall_thickness_mm": round(float(np.median(geo.thickness_mm[surf][geo.transmural[surf] >= 1])), 1),
        },
        "hemodynamics": {
            "summary": summary,
            "calibration_fit": fit,
            "parameters": {k: round(float(getattr(p, k)), 4) for k in
                           ("hr", "ees", "v0", "edp_a", "edp_b", "r_sys", "c_art", "z_c", "r_av", "p_pv")},
            "curves": {k: np.round(v[::step], 2).tolist() for k, v in rec.items()},
            "activation_curve": np.round(e_lv[::step], 4).tolist(),
            "dt_ms": step * dt * 1000.0,
        },
    }
