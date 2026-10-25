"""
CordiaAI — Digital Twin of a Heart (standalone interactive prototype)
=======================================================================

A real-time cardiovascular digital twin: an anatomical longitudinal-section
(cutaway) view of a four-chamber heart that visibly beats, driven by an
*actual* physiological simulation rather than a canned animation, alongside
a live ECG and a live pressure-volume (PV) loop, all responding immediately
to three sliders: Heart Rate, Blood Pressure, and Myocardial Stiffness.

THE PHYSIOLOGY, IN PLAIN TERMS
-------------------------------
1. Time-varying elastance (Suga-Sagawa) model for the left ventricle.
   The ventricle's "stiffness" E(t) oscillates every heartbeat between a
   low value (Emin, relaxed/compliant, diastole) and a high value (Emax,
   contracted/stiff, systole). Pressure is then simply:
       P_LV(t) = E(t) * (V_LV(t) - V_dead)
   This single equation is *why* the same volume produces low pressure
   during filling and high pressure during ejection -- it's the standard
   textbook model of how a ventricle actually works, not a fitted curve.

2. Two heart valves, modeled as one-way flow resistors (a diode with a
   resistance), so blood only flows the correct direction, only when
   the driving pressure difference is positive:
       Q_mitral(t) = max(0, (P_LA - P_LV) / R_mitral)   -- filling
       Q_aortic(t) = max(0, (P_LV - P_Ao) / R_aortic)   -- ejection
   Ventricular volume is then just conservation of flow:
       dV_LV/dt = Q_mitral - Q_aortic

3. A 3-element Windkessel model for the arterial system (the "Blood
   Pressure" the aorta actually sees), integrated alongside the heart:
       dP_Ao/dt = (Q_aortic - P_Ao / R_systemic) / C_arterial
   Raising the Blood Pressure slider raises R_systemic (systemic
   vascular resistance) -- exactly how real hypertension raises blood
   pressure: the heart works against more resistance, not by "turning up
   a dial on pressure" directly.

4. Myocardial Stiffness raises Emax (stronger, stiffer contraction) AND
   Emin (harder to fill, restrictive/fibrotic-pattern diastole) --
   modeling disease (e.g. fibrosis, hypertrophic/restrictive
   cardiomyopathy) as reduced diastolic compliance, which is the real
   physiological hallmark of a stiffened heart.

5. The ECG is synthesised with a Gaussian-sum PQRST model (each wave --
   P, Q, R, S, T -- as a Gaussian bump at the right point in the cycle),
   the standard lightweight way to generate a realistic-looking synthetic
   ECG trace without a full electrophysiological cell model. It is
   illustrative of correct ECG *morphology and timing* relative to heart
   rate -- it is not derived from the mechanical simulation above (real
   hearts couple electrical and mechanical activity; this prototype does
   not model that coupling, and says so here rather than implying it does).

HONEST SCOPE NOTE ON THE HEART VISUAL
---------------------------------------
The heart panel is a schematic anatomical **longitudinal-section (cutaway)
view** -- the same "four-chamber view" convention used in cardiology
textbooks and echocardiography -- not a photorealistic, segmented-anatomy
render. A commercial product like Dassault Systemes' Living Heart Project
is built from real segmented CT/MRI geometry and a GPU ray-traced renderer
by a large engineering team; that is a fundamentally different (and much
larger) undertaking than a physics-prototype script, and this doesn't
pretend otherwise. What *is* real: chamber sizes and colour intensity below
are driven directly by the same ODE simulation as the ECG and PV loop --
not independently animated.

INSTALL & RUN
-------------
    pip install numpy scipy matplotlib
    python digital_twin_heart_simulation.py

Requires an interactive matplotlib backend (TkAgg, Qt5Agg, etc.) to show
the sliders and live animation -- this will not render in a plain
headless/non-interactive environment.
"""

from __future__ import annotations

import sys

# ------------------------------------------------------------- Dependencies
try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency 'numpy'. Install it with:\n    pip install numpy")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider
    from matplotlib.patches import Ellipse, Polygon, FancyArrow
except ImportError:
    sys.exit("Missing dependency 'matplotlib'. Install it with:\n    pip install matplotlib")

try:
    from scipy.stats import norm  # only used for a vectorised Gaussian helper
except ImportError:
    sys.exit("Missing dependency 'scipy'. Install it with:\n    pip install scipy")


# =====================================================================
# 1. Cardiac electrical activity -- synthetic PQRST ECG
# =====================================================================
ECG_WAVES = {
    "P": {"a": 0.20, "b": 0.10, "c": 0.020},
    "Q": {"a": -0.10, "b": 0.24, "c": 0.008},
    "R": {"a": 1.20, "b": 0.26, "c": 0.010},
    "S": {"a": -0.30, "b": 0.29, "c": 0.010},
    "T": {"a": 0.35, "b": 0.50, "c": 0.045},
}


def ecg_waveform(t_frac: np.ndarray) -> np.ndarray:
    """t_frac: position within the cardiac cycle, in [0, 1). Returns the
    synthetic ECG voltage (arbitrary units) as a sum of Gaussian waves."""
    ecg = np.zeros_like(t_frac)
    for wave in ECG_WAVES.values():
        d = np.minimum(np.abs(t_frac - wave["b"]), 1 - np.abs(t_frac - wave["b"]))
        ecg += wave["a"] * norm.pdf(d, scale=wave["c"]) * wave["c"] * np.sqrt(2 * np.pi)
    return ecg


# =====================================================================
# 2. Cardiac mechanics -- time-varying elastance model
# =====================================================================
def normalized_elastance(t_frac: float, systole_frac: float = 0.30, relax_frac: float = 0.15) -> float:
    """e(t) in [0, 1]: 0 = fully relaxed (diastole), 1 = fully contracted
    (peak systole). Piecewise-sinusoidal activation -- a standard simplified
    stand-in for the fuller double-Hill elastance curve used in the
    cardiovascular-modelling literature (e.g. Stergiopulos et al. 1996)."""
    if t_frac <= systole_frac:
        return 0.5 * (1 - np.cos(np.pi * t_frac / systole_frac))
    elif t_frac <= systole_frac + relax_frac:
        return 0.5 * (1 + np.cos(np.pi * (t_frac - systole_frac) / relax_frac))
    return 0.0


class HeartModel:
    """The full lumped-parameter cardiovascular simulation: one ventricle
    (time-varying elastance), two valves (flow resistors), one arterial
    Windkessel compartment. State = [V_LV (mL), P_Ao (mmHg)]."""

    def __init__(self):
        self.V_dead = 10.0
        self.P_LA = 10.0
        self.R_mitral = 0.01
        self.R_aortic = 0.006
        self.C_arterial = 1.5

        self.V_LV = 120.0
        self.P_Ao = 80.0
        self.t = 0.0

        self.heart_rate = 75.0
        self.R_systemic = 1.0
        self.stiffness = 1.0

        # Exposed each step() for the visual layer, so chamber flashes /
        # flow-arrow brightness reflect the model's own valve state rather
        # than being guessed from volume alone.
        self.last_Q_mitral = 0.0
        self.last_Q_aortic = 0.0

    @property
    def period(self) -> float:
        return 60.0 / self.heart_rate

    @property
    def Emax(self) -> float:
        return 2.0 * self.stiffness

    @property
    def Emin(self) -> float:
        return 0.06 * self.stiffness

    def elastance(self, t_frac: float) -> float:
        e = normalized_elastance(t_frac)
        return self.Emin + (self.Emax - self.Emin) * e

    def _derivatives(self, V_LV: float, P_Ao: float, t: float) -> tuple[float, float]:
        t_frac = (t % self.period) / self.period
        E = self.elastance(t_frac)
        P_LV = E * (V_LV - self.V_dead)

        Q_mitral = max(0.0, (self.P_LA - P_LV) / self.R_mitral)
        Q_aortic = max(0.0, (P_LV - P_Ao) / self.R_aortic)

        dV_LV = Q_mitral - Q_aortic
        dP_Ao = (Q_aortic - P_Ao / self.R_systemic) / self.C_arterial
        return dV_LV, dP_Ao

    def step(self, dt: float) -> dict:
        V, P, t = self.V_LV, self.P_Ao, self.t

        k1v, k1p = self._derivatives(V, P, t)
        k2v, k2p = self._derivatives(V + 0.5 * dt * k1v, P + 0.5 * dt * k1p, t + 0.5 * dt)
        k3v, k3p = self._derivatives(V + 0.5 * dt * k2v, P + 0.5 * dt * k2p, t + 0.5 * dt)
        k4v, k4p = self._derivatives(V + dt * k3v, P + dt * k3p, t + dt)

        self.V_LV = max(self.V_dead + 1.0, V + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v))
        self.P_Ao = max(0.0, P + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p))
        self.t += dt

        t_frac = (self.t % self.period) / self.period
        E = self.elastance(t_frac)
        P_LV = max(0.0, E * (self.V_LV - self.V_dead))
        self.last_Q_mitral = max(0.0, (self.P_LA - P_LV) / self.R_mitral)
        self.last_Q_aortic = max(0.0, (P_LV - self.P_Ao) / self.R_aortic)

        return {
            "t": self.t,
            "t_frac": t_frac,
            "V_LV": self.V_LV,
            "P_LV": P_LV,
            "P_Ao": self.P_Ao,
            "ecg": float(ecg_waveform(np.array([t_frac]))[0]),
            "mitral_open": self.last_Q_mitral > 0.5,
            "aortic_open": self.last_Q_aortic > 0.5,
        }


LV_REFERENCE_VOLUME = 120.0  # mL, matches HeartModel's initial V_LV


# =====================================================================
# 3. Anatomical longitudinal-section (cutaway) heart geometry
# =====================================================================
# A schematic four-chamber cutaway -- the same convention cardiology
# textbooks and echocardiography use (the "four-chamber view") -- not a
# photorealistic segmented-anatomy render. See the module docstring for
# why that's a deliberate, stated scope boundary rather than a gap.
def heart_outline(scale: float = 1.0, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """The classic parametric 'heart curve', used as the outer myocardium
    silhouette for the cutaway."""
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return (x / 16.0) * scale, (y / 16.0) * scale


# Base (reference-volume) geometry for each chamber's cavity, positioned
# inside the outer silhouette. LV is the largest and extends closest to
# the apex, matching real anatomy (the apex is mostly LV-formed). Colour
# follows the standard convention: red = oxygenated (left heart), blue =
# deoxygenated (right heart) -- like the reference images, not a strict
# clinical-imaging left/right orientation (stated here so it isn't assumed).
CHAMBERS_2D = {
    "LA": {"center": (-0.40, 0.48), "radii": (0.27, 0.23), "color": "#ff5555", "glow": "#ff8888"},
    "RA": {"center": (0.40, 0.48), "radii": (0.25, 0.21), "color": "#4da6ff", "glow": "#8ec9ff"},
    "LV": {"center": (-0.38, -0.32), "radii": (0.40, 0.60), "color": "#ff2b2b", "glow": "#ff6b6b"},
    "RV": {"center": (0.38, -0.14), "radii": (0.32, 0.44), "color": "#2b8fff", "glow": "#6bb6ff"},
}

VALVES_2D = {  # (x, y) marker position, and which chamber pair they gate
    "mitral": {"pos": (-0.39, 0.14), "open_source": "mitral_open"},
    "tricuspid": {"pos": (0.39, 0.20), "open_source": "mitral_open"},  # simplified: shares the LV model's timing
    "aortic": {"pos": (-0.20, -0.90), "open_source": "aortic_open"},
    "pulmonic": {"pos": (0.30, -0.55), "open_source": "aortic_open"},
}


def draw_glow_patch(ax, kind: str, xy, dims, color: str, glow_color: str, alpha: float, zorder: int):
    """Fakes an emissive/glow look (matplotlib has no true bloom) by
    layering progressively larger, more transparent copies of the same
    patch behind a solid core -- a standard, well-known matplotlib trick."""
    patches = []
    for i, (scale, a) in enumerate([(1.55, 0.05), (1.30, 0.09), (1.12, 0.14)]):
        if kind == "ellipse":
            p = Ellipse(xy, dims[0] * 2 * scale, dims[1] * 2 * scale, color=glow_color, alpha=a, zorder=zorder - 1, lw=0)
        else:
            p = Polygon(dims, closed=True, color=glow_color, alpha=a, zorder=zorder - 1, lw=0)
        ax.add_patch(p)
        patches.append(p)
    if kind == "ellipse":
        core = Ellipse(xy, dims[0] * 2, dims[1] * 2, color=color, alpha=alpha, zorder=zorder, lw=0.6, ec=glow_color)
    else:
        core = Polygon(dims, closed=True, color=color, alpha=alpha, zorder=zorder, lw=0.6, ec=glow_color)
    ax.add_patch(core)
    patches.append(core)
    return patches


def draw_heart_section(ax, state: dict):
    """Draws one frame of the longitudinal-section heart, sized and
    coloured from the real simulation state. Clears and redraws the axis
    each call -- simple and robust for a ~20-30 fps prototype."""
    ax.clear()
    ax.set_facecolor("#050a14")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.55, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Longitudinal section — four-chamber view", color="#dfe8f5", fontsize=11, pad=6)

    # Outer myocardial wall, glowing dark-red silhouette
    ox, oy = heart_outline(scale=1.15)
    draw_glow_patch(ax, "polygon", None, np.column_stack([ox, oy]), "#3a0d10", "#8a1f1f", 0.92, zorder=1)

    # Septum
    ax.plot([0, 0], [0.75, -0.95], color="#1a0507", lw=3, zorder=2, solid_capstyle="round")

    # Chamber cavities, radii scaled by the real simulated volume.
    # Area (~ pi*r1*r2) is scaled by the LV volume ratio, so a 2D cavity
    # visibly grows/shrinks with the same number driving the PV loop.
    vol_ratio = state["V_LV"] / LV_REFERENCE_VOLUME
    lv_scale = np.clip(np.sqrt(vol_ratio), 0.65, 1.2)
    other_scale = np.clip(0.55 + 0.45 * lv_scale, 0.65, 1.15)

    pressure_glow = np.clip(state["P_LV"] / 140.0, 0.15, 1.0)  # brighter fill at higher pressure (systole)
    for name, spec in CHAMBERS_2D.items():
        scale = lv_scale if name == "LV" else other_scale
        radii = (spec["radii"][0] * scale, spec["radii"][1] * scale)
        alpha = 0.55 + 0.4 * pressure_glow if name in ("LV", "RV") else 0.65
        draw_glow_patch(ax, "ellipse", spec["center"], radii, spec["color"], spec["glow"], alpha, zorder=3)
        ax.text(spec["center"][0], spec["center"][1], name, color="white", fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=5)

    # Great vessels (simple tube stubs): aorta + pulmonary artery leaving
    # the ventricles, vena cava + pulmonary veins entering the atria.
    vessel_style = dict(color="#20303f", zorder=1, lw=0)
    ax.add_patch(Polygon([(-0.30, 0.75), (-0.16, 0.75), (-0.10, 1.15), (-0.24, 1.15)], **vessel_style))  # SVC into RA-ish
    ax.add_patch(Polygon([(0.30, 0.75), (0.44, 0.75), (0.50, 1.15), (0.36, 1.15)], **vessel_style))
    ax.add_patch(Polygon([(-0.32, -0.85), (-0.14, -0.90), (-0.10, -1.35), (-0.28, -1.32)], color="#5a1414", zorder=1, lw=0))  # aorta
    ax.add_patch(Polygon([(0.22, -0.55), (0.42, -0.55), (0.48, -1.05), (0.28, -1.05)], color="#142f4a", zorder=1, lw=0))  # pulmonary artery

    # Valve markers -- brighten when the model reports that valve is
    # actually passing flow this instant, a real (not decorative) detail.
    for name, spec in VALVES_2D.items():
        is_open = state.get(spec["open_source"], False)
        color = "#e8f7ff" if is_open else "#3a4a5a"
        ax.plot(*spec["pos"], marker="o", markersize=6 if is_open else 4.5,
                color=color, zorder=6, markeredgecolor="#0a1420", markeredgewidth=0.6)

    # Flow arrows -- only drawn brightly while that valve is actually open
    def flow_arrow(x, y, dx, dy, open_flag):
        alpha = 0.9 if open_flag else 0.12
        ax.add_patch(FancyArrow(x, y, dx, dy, width=0.012, head_width=0.05, head_length=0.06,
                                 color="#eaffff", alpha=alpha, zorder=7, length_includes_head=True))

    flow_arrow(-0.39, 0.30, 0, -0.10, state.get("mitral_open", False))
    flow_arrow(0.39, 0.34, 0, -0.10, state.get("mitral_open", False))
    flow_arrow(-0.22, -1.0, -0.06, -0.15, state.get("aortic_open", False))
    flow_arrow(0.32, -0.75, 0.05, -0.18, state.get("aortic_open", False))

    ax.text(0, 1.22, f"LV vol {state['V_LV']:.0f} mL   ·   LV pressure {state['P_LV']:.0f} mmHg",
            color="#9fb4c8", fontsize=8.5, ha="center", family="monospace")


# =====================================================================
# 4. Assemble the interactive dashboard
# =====================================================================
DARK_BG = "#050a14"
PANEL_BG = "#0a1220"


def style_dark_axis(ax, title: str):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="#dfe8f5", fontsize=11)
    ax.tick_params(colors="#7f93a8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#26374a")
    ax.xaxis.label.set_color("#9fb4c8")
    ax.yaxis.label.set_color("#9fb4c8")


def main() -> None:
    model = HeartModel()
    dt = 0.01

    fig = plt.figure(figsize=(13, 8), facecolor=DARK_BG)
    fig.suptitle("CordiaAI — Digital Twin of a Heart (prototype)", fontsize=13, fontweight="bold", color="#eaf6ff")

    ax_heart = fig.add_subplot(2, 2, 1)
    ax_ecg = fig.add_subplot(2, 2, 2)
    ax_pv = fig.add_subplot(2, 2, 3)
    ax_vol = fig.add_subplot(2, 2, 4)
    plt.subplots_adjust(bottom=0.24, hspace=0.4, wspace=0.28)

    style_dark_axis(ax_ecg, "ECG (synthetic PQRST)")
    ax_ecg.set_xlim(0, 1); ax_ecg.set_ylim(-0.5, 1.5)
    ax_ecg.set_xlabel("fraction of cardiac cycle")
    (ecg_line,) = ax_ecg.plot([], [], color="#39e6c1", lw=1.8)
    ecg_history_t: list[float] = []
    ecg_history_v: list[float] = []

    style_dark_axis(ax_pv, "Left-ventricular pressure–volume loop")
    ax_pv.set_xlabel("LV volume (mL)"); ax_pv.set_ylabel("LV pressure (mmHg)")
    ax_pv.set_xlim(0, 200); ax_pv.set_ylim(0, 200)
    (pv_line,) = ax_pv.plot([], [], color="#c77dff", lw=2.0)
    pv_history_v: list[float] = []
    pv_history_p: list[float] = []

    style_dark_axis(ax_vol, "LV volume & aortic pressure over time")
    ax_vol.set_xlabel("time (s)")
    (vol_line,) = ax_vol.plot([], [], color="#ff5c5c", lw=1.5, label="LV volume (mL)")
    (aop_line,) = ax_vol.plot([], [], color="#4da6ff", lw=1.5, label="Aortic pressure (mmHg)")
    leg = ax_vol.legend(loc="upper right", fontsize=8, facecolor=PANEL_BG, edgecolor="#26374a")
    for text in leg.get_texts():
        text.set_color("#dfe8f5")
    ax_vol.set_xlim(0, 4); ax_vol.set_ylim(0, 200)
    time_history: list[float] = []
    vol_history: list[float] = []
    aop_history: list[float] = []

    # ---- Sliders (dark-styled)
    slider_axes = []
    for i, y in enumerate([0.10, 0.06, 0.02]):
        a = plt.axes((0.15, y, 0.3, 0.03), facecolor="#1a2738")
        slider_axes.append(a)
    s_hr = Slider(slider_axes[0], "Heart Rate (BPM)", 40, 180, valinit=model.heart_rate, valstep=1,
                  color="#39e6c1", initcolor="none")
    s_bp = Slider(slider_axes[1], "Blood Pressure (SVR)", 0.5, 2.5, valinit=model.R_systemic, valstep=0.05,
                  color="#ff8a5c", initcolor="none")
    s_stiff = Slider(slider_axes[2], "Myocardial Stiffness", 0.5, 3.0, valinit=model.stiffness, valstep=0.05,
                     color="#c77dff", initcolor="none")
    for s in (s_hr, s_bp, s_stiff):
        s.label.set_color("#dfe8f5")
        s.valtext.set_color("#dfe8f5")

    info_text = fig.text(0.58, 0.05, "", fontsize=9, family="monospace", color="#9fb4c8")

    def update_sliders(_event=None):
        model.heart_rate = s_hr.val
        model.R_systemic = s_bp.val
        model.stiffness = s_stiff.val

    s_hr.on_changed(update_sliders)
    s_bp.on_changed(update_sliders)
    s_stiff.on_changed(update_sliders)

    def animate(_frame):
        state = model.step(dt)

        draw_heart_section(ax_heart, state)

        if ecg_history_t and state["t_frac"] < ecg_history_t[-1]:
            ecg_history_t.clear()
            ecg_history_v.clear()
        ecg_history_t.append(state["t_frac"])
        ecg_history_v.append(state["ecg"])
        ecg_line.set_data(ecg_history_t, ecg_history_v)

        pv_history_v.append(state["V_LV"])
        pv_history_p.append(state["P_LV"])
        max_pts = int(2 * model.period / dt)
        if len(pv_history_v) > max_pts:
            del pv_history_v[: len(pv_history_v) - max_pts]
            del pv_history_p[: len(pv_history_p) - max_pts]
        pv_line.set_data(pv_history_v, pv_history_p)
        ax_pv.set_xlim(max(0, min(pv_history_v) - 10), max(pv_history_v) + 10)
        ax_pv.set_ylim(0, max(50, max(pv_history_p) + 20))

        time_history.append(state["t"])
        vol_history.append(state["V_LV"])
        aop_history.append(state["P_Ao"])
        window = 4.0
        while time_history and state["t"] - time_history[0] > window:
            time_history.pop(0)
            vol_history.pop(0)
            aop_history.pop(0)
        vol_line.set_data(time_history, vol_history)
        aop_line.set_data(time_history, aop_history)
        if time_history:
            ax_vol.set_xlim(time_history[0], max(time_history[0] + window, time_history[-1]))

        stroke_volume = max(pv_history_v[-max_pts:]) - min(pv_history_v[-max_pts:]) if pv_history_v else 0
        info_text.set_text(
            f"HR: {model.heart_rate:5.0f} bpm   "
            f"Mean Ao pressure: {state['P_Ao']:6.1f} mmHg   "
            f"LV volume: {state['V_LV']:6.1f} mL   "
            f"Stroke volume (approx): {stroke_volume:5.1f} mL"
        )
        return []

    ani = FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
    plt.show()
    return ani


if __name__ == "__main__":
    main()
