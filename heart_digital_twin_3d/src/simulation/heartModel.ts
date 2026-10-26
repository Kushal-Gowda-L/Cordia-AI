/**
 * CordiaAI — Digital Twin of a Heart: physiological simulation engine.
 *
 * This is the single source of truth ("HeartState" per the project spec)
 * that everything else — the 3D mesh, blood-flow particles, ECG, pressure
 * graph, and dashboard — reads from. Nothing in the visual layer invents
 * its own numbers; every visual is a function of the outputs computed here.
 *
 * MODEL DESIGN (see README.md in this folder for the full write-up):
 * Each ventricle is simulated with a real time-varying elastance model
 * (the standard Suga-Sagawa formulation used throughout cardiovascular
 * modelling), one-way valve flow resistors, and a Windkessel arterial
 * compartment for the systemic (LV) and pulmonary (RV) circuits. This is
 * the same family of equations validated in this project's standalone
 * Python prototype (digital_twin_heart_simulation.py), ported to
 * TypeScript and extended to a real right-heart simulation plus an
 * atrial-kick pressure model, rather than being independently re-derived.
 *
 * This is an EDUCATIONAL, SCIENTIFICALLY-INSPIRED approximation, not a
 * clinically validated model — every simplification is called out in
 * README.md.
 */

export interface HeartParams {
  heartRate: number; // BPM, 40-180
  baseStrokeVolume: number; // mL, 30-150 (baseline pump strength dial)
  systolicBP: number; // mmHg, 80-220 (target/baseline)
  diastolicBP: number; // mmHg, 40-140 (target/baseline)
  contractility: number; // 0-100 (low-high, inotropic state)
  preload: number; // 0-100 (low-high, venous return / filling pressure)
  afterload: number; // 0-100 (low-high, resistance the ventricle must eject against)
  oxygenSaturation: number; // %, 70-100
  vascularResistance: number; // 0-100 (low-high, systemic vascular resistance)
  aorticStenosis: number; // 0-1 severity, restricts aortic valve orifice
  mitralRegurgitation: number; // 0-1 severity, allows backward LV->LA leak during systole
}

export const DEFAULT_PARAMS: HeartParams = {
  heartRate: 72,
  baseStrokeVolume: 70,
  systolicBP: 120,
  diastolicBP: 80,
  contractility: 50,
  preload: 50,
  afterload: 50,
  oxygenSaturation: 98,
  vascularResistance: 50,
  aorticStenosis: 0,
  mitralRegurgitation: 0,
};

export const PARAM_RANGES: Record<
  keyof Omit<HeartParams, "aorticStenosis" | "mitralRegurgitation">,
  { min: number; max: number; unit: string; normalLow: number; normalHigh: number }
> = {
  heartRate: { min: 40, max: 180, unit: "BPM", normalLow: 60, normalHigh: 100 },
  baseStrokeVolume: { min: 30, max: 150, unit: "mL", normalLow: 60, normalHigh: 100 },
  systolicBP: { min: 80, max: 220, unit: "mmHg", normalLow: 90, normalHigh: 120 },
  diastolicBP: { min: 40, max: 140, unit: "mmHg", normalLow: 60, normalHigh: 80 },
  contractility: { min: 0, max: 100, unit: "%", normalLow: 40, normalHigh: 70 },
  preload: { min: 0, max: 100, unit: "%", normalLow: 40, normalHigh: 70 },
  afterload: { min: 0, max: 100, unit: "%", normalLow: 40, normalHigh: 70 },
  oxygenSaturation: { min: 70, max: 100, unit: "%", normalLow: 95, normalHigh: 100 },
  vascularResistance: { min: 0, max: 100, unit: "%", normalLow: 40, normalHigh: 70 },
};

// ---------------------------------------------------- Electrical timing map
// tFrac = 0 is defined as SA-node firing (start of the P wave) -- this is
// the standard ECG-based reference point for "start of one cardiac cycle."
// Every other electrical and mechanical event is timed relative to it, so
// the ECG, the conduction-system animation, and the chamber contractions
// all stay genuinely synchronized rather than merely sharing a period.
export const CONDUCTION_TIMING = {
  saNode: 0.0,
  atrialActivationEnd: 0.09,
  avNode: 0.1,
  bundleOfHis: 0.12,
  bundleBranches: 0.135,
  purkinjeFibers: 0.15,
  ventricularActivationEnd: 0.2,
};

// Electromechanical delay: the ventricle begins contracting slightly after
// the QRS complex starts (real physiology), not at tFrac = 0.
const VENTRICULAR_ACTIVATION_START = 0.17;
const ATRIAL_KICK_CENTER = 0.06;

// -------------------------------------------------------------- ECG synthesis
const ECG_WAVES = {
  P: { a: 0.2, b: 0.03, c: 0.02 },
  Q: { a: -0.1, b: 0.155, c: 0.006 },
  R: { a: 1.2, b: 0.17, c: 0.008 },
  S: { a: -0.3, b: 0.185, c: 0.008 },
  T: { a: 0.35, b: 0.42, c: 0.05 },
};

function gaussian(x: number, c: number): number {
  return Math.exp(-(x * x) / (2 * c * c));
}

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

export function ecgWaveform(tFrac: number): number {
  let v = 0;
  for (const w of Object.values(ECG_WAVES)) {
    const d = Math.min(Math.abs(tFrac - w.b), 1 - Math.abs(tFrac - w.b));
    v += w.a * gaussian(d, w.c);
  }
  return v;
}

// ------------------------------------------------------- Elastance activation
// Normalized 0-1 activation curve for one ventricle: flat at 0 until the
// QRS-triggered electromechanical delay, then rises through isovolumic
// contraction + ejection, falls through isovolumic relaxation, flat at 0
// through the rest of diastole. Matches Wiggers-diagram proportions and is
// driven off the same tFrac=0-at-P-wave clock as the ECG.
function elastanceShape(tFrac: number, systoleFrac: number, relaxFrac: number): number {
  const local = tFrac - VENTRICULAR_ACTIVATION_START;
  if (local < 0) return 0;
  if (local <= systoleFrac) {
    return 0.5 * (1 - Math.cos((Math.PI * local) / systoleFrac));
  } else if (local <= systoleFrac + relaxFrac) {
    return 0.5 * (1 + Math.cos((Math.PI * (local - systoleFrac)) / relaxFrac));
  }
  return 0;
}

// Atrial "kick" bump: a short pressure boost shortly after the P wave and
// just before the QRS complex, producing the real "a wave" seen in atrial
// pressure tracings and the extra late-diastolic filling boost of atrial
// systole -- timed to follow atrial depolarization, not the ventricle.
function atrialKick(tFrac: number, center = ATRIAL_KICK_CENTER, width = 0.03): number {
  const d = Math.min(Math.abs(tFrac - center), 1 - Math.abs(tFrac - center));
  return gaussian(d, width);
}

interface VentricleConstants {
  vDead: number;
  emaxBase: number;
  eminBase: number;
  rInflow: number;
  rOutflow: number;
  cArterial: number;
  systoleFrac: number;
  relaxFrac: number;
}

const LV_CONST: VentricleConstants = {
  vDead: 10,
  // End-systolic volume lands where E(t)*(V - V_dead) falls to aortic
  // pressure, so emaxBase sets how completely the ventricle empties.
  // Calibrated for a normal resting ejection fraction (~55-60%).
  emaxBase: 1.4,
  eminBase: 0.08, // diastolic stiffness; with P_LA this sets EDV
  rInflow: 0.01, // mitral
  rOutflow: 0.006, // aortic
  cArterial: 1.5,
  systoleFrac: 0.3,
  relaxFrac: 0.12,
};

const RV_CONST: VentricleConstants = {
  vDead: 12,
  emaxBase: 0.45, // right ventricle is a low-pressure pump
  eminBase: 0.045,
  rInflow: 0.009, // tricuspid
  rOutflow: 0.01, // pulmonic
  cArterial: 3.5, // pulmonary circuit is far more compliant
  systoleFrac: 0.3,
  relaxFrac: 0.12,
};

interface VentricleState {
  V: number;
  Pao: number; // arterial pressure downstream (aortic for LV, pulmonary for RV)
}

function stepVentricle(
  state: VentricleState,
  t: number,
  dt: number,
  period: number,
  c: VentricleConstants,
  Patrium: number,
  Emax: number,
  Emin: number,
  Rsystemic: number,
  stenosisFactor: number,
  regurgFraction: number
): { state: VentricleState; Plv: number; Qin: number; Qout: number; Qregurg: number } {
  const deriv = (V: number, Pao: number, tt: number) => {
    const tFrac = ((tt % period) + period) % period / period;
    const E = Emin + (Emax - Emin) * elastanceShape(tFrac, c.systoleFrac, c.relaxFrac);
    const Plv = clamp(E * Math.max(0, V - c.vDead), 0, 350);
    // Stenosis shrinks the valve opening; resistance across an orifice rises
    // roughly with the inverse square of its area, so severe stenosis (1)
    // gives a pressure drop across the valve of ~60 mmHg at peak ejection --
    // the range echo labs grade as severe -- and moderate (0.5-0.7) ~15-30.
    const rOut = c.rOutflow * (1 + 40 * stenosisFactor * stenosisFactor);
    const Qin = Math.max(0, (Patrium - Plv) / c.rInflow);
    const Qout = Math.max(0, (Plv - Pao) / rOut);
    // Backward leak through an incompetent mitral valve while the ventricle
    // is above atrial pressure. At severity 1 this returns ~50% of each
    // beat to the atrium (severe: regurgitant fraction >= 50%); 0.5 ~ 25-30%.
    const Qregurg = regurgFraction > 0 ? Math.max(0, Plv - Patrium) * regurgFraction * 2.5 : 0;
    const dV = Qin - Qout - Qregurg;
    const dPao = (Qout - Pao / Rsystemic) / c.cArterial;
    return { dV, dPao, Plv, Qin, Qout, Qregurg };
  };

  const k1 = deriv(state.V, state.Pao, t);
  const k2 = deriv(state.V + 0.5 * dt * k1.dV, state.Pao + 0.5 * dt * k1.dPao, t + 0.5 * dt);
  const k3 = deriv(state.V + 0.5 * dt * k2.dV, state.Pao + 0.5 * dt * k2.dPao, t + 0.5 * dt);
  const k4 = deriv(state.V + dt * k3.dV, state.Pao + dt * k3.dPao, t + dt);

  // Hard physiological clamps double as numerical-stability guards: the
  // valve-flow resistors are deliberately small (real orifices are low
  // resistance), which makes the explicit RK4 step stiff under extreme
  // slider combinations (e.g. severe aortic stenosis). Rather than
  // substepping the whole integrator, bound state to the outer edge of
  // plausible human physiology -- consistent with spec section 14's
  // "prevent impossible combinations where practical."
  const V = clamp(state.V + (dt / 6) * (k1.dV + 2 * k2.dV + 2 * k3.dV + k4.dV), c.vDead + 1, 400);
  const Pao = clamp(state.Pao + (dt / 6) * (k1.dPao + 2 * k2.dPao + 2 * k3.dPao + k4.dPao), 0, 300);

  const tFrac = ((t + dt) % period) / period;
  const E = Emin + (Emax - Emin) * elastanceShape(tFrac, c.systoleFrac, c.relaxFrac);
  const Plv = clamp(E * Math.max(0, V - c.vDead), 0, 350);

  return { state: { V, Pao }, Plv, Qin: k1.Qin, Qout: k1.Qout, Qregurg: k1.Qregurg };
}

// --------------------------------------------------------------- Full model
export type ValveName = "mitral" | "tricuspid" | "aortic" | "pulmonic";

export interface HeartOutputs {
  t: number;
  tFrac: number;
  period: number;
  ecg: number;
  // Chamber volumes (mL)
  vLV: number;
  vRV: number;
  vLA: number;
  vRA: number;
  // Chamber pressures (mmHg)
  pLV: number;
  pRV: number;
  pLA: number;
  pRA: number;
  pAo: number;
  pPA: number;
  // Valve open state (0-1, partially open supported for stenosis)
  valves: Record<ValveName, number>;
  // Flow rates (mL/s), used to drive particle speed
  flowSystemicIn: number; // vena cava -> RA
  flowPulmonaryOut: number; // RV -> PA
  flowPulmonaryIn: number; // pulmonary veins -> LA
  flowSystemicOut: number; // LV -> Aorta
  regurgFlow: number;
  // Derived clinical-style summary numbers
  cardiacOutput: number; // L/min
  ejectionFraction: number; // %
  edv: number; // mL
  esv: number; // mL
  meanArterialPressure: number; // mmHg
  cardiacWorkload: number; // rate-pressure product / 100
  coronaryPerfusionPressure: number; // mmHg
  effectiveStrokeVolume: number; // mL
}

export class HeartSimulation {
  private t = 0;
  private lv: VentricleState = { V: 120, Pao: 80 };
  private rv: VentricleState = { V: 130, Pao: 15 };
  // Rolling min/max trackers for the current beat, used to report
  // systolic/diastolic pressures and EDV/ESV from the live simulation.
  private beatMinV = 120;
  private beatMaxV = 170;
  private prevTFrac = 0;
  private lastBeatEDV = 170;
  private lastBeatESV = 120;
  // Volume leaked back through the mitral valve this beat / last beat (mL).
  private beatRegurg = 0;
  private lastBeatRegurg = 0;

  reset(params: HeartParams) {
    this.t = 0;
    this.lv = { V: 120, Pao: params.diastolicBP };
    this.rv = { V: 130, Pao: 15 };
    this.beatMinV = 120;
    this.beatRegurg = 0;
    this.lastBeatRegurg = 0;
    this.beatMaxV = 170;
    this.prevTFrac = 0;
  }

  step(dtRaw: number, params: HeartParams): HeartOutputs {
    // Cap dt so a dropped frame / tab-away doesn't blow up the integrator.
    const dt = Math.min(Math.max(dtRaw, 0), 0.02);
    const period = 60 / params.heartRate;

    const contractilityFactor = 0.5 + 1.5 * (params.contractility / 100); // 0.5x - 2x
    const preloadFactor = 0.6 + 0.8 * (params.preload / 100); // 0.6x - 1.4x -> scales P_atrium
    const targetMAP = params.diastolicBP + (params.systolicBP - params.diastolicBP) / 3;
    const afterloadFactor = 0.5 + 1.5 * (params.afterload / 100); // 0.5x - 2x

    // MAP = CO x TPR, rearranged to solve for the resistance that would
    // produce the requested baseline pressure at the current cardiac
    // output -- then layered with the vascular-resistance and afterload
    // sliders as physiological modifiers on top of that baseline.
    const approxCO = (params.heartRate * params.baseStrokeVolume) / 1000; // L/min
    const baseR = approxCO > 0.1 ? targetMAP / (approxCO * 16.67) : 1.2; // scaled to model units
    const resistanceFactor = 0.5 + 1.5 * (params.vascularResistance / 100);
    const Rsystemic = Math.max(0.3, Math.min(4.5, baseR * resistanceFactor * afterloadFactor * 0.5));
    const Rpulmonary = Math.max(0.08, Rsystemic * 0.12);

    const Emax = LV_CONST.emaxBase * contractilityFactor * (params.baseStrokeVolume / 70);
    const Emin = LV_CONST.eminBase;
    const EmaxRV = RV_CONST.emaxBase * contractilityFactor;
    const EminRV = RV_CONST.eminBase;

    const tFrac = ((this.t % period) + period) / period % 1;
    const kick = atrialKick(tFrac);
    // Filling ends when ventricular pressure rises to meet atrial
    // pressure, so these values (with Emin) set end-diastolic volume:
    // P_LA/Emin + V_dead. Calibrated so a default preload of 50 gives a
    // normal LA pressure near 10 mmHg and an EDV in the 120-170 mL range.
    // Mitral regurgitation loads the left atrium: its pressure rises (chronic
    // volume overload, which also fills the LV more -- a dilated LV) and a
    // systolic "v wave" builds while the jet pours in.
    const mr = params.mitralRegurgitation;
    const lvSystole = elastanceShape(tFrac, LV_CONST.systoleFrac, LV_CONST.relaxFrac);
    const pLA = 3 + 7 * preloadFactor + 4 * kick + mr * (1.5 + 6 * lvSystole);
    const pRA = 1.5 + 3.5 * preloadFactor + 2.5 * kick;

    const lvResult = stepVentricle(
      this.lv, this.t, dt, period, LV_CONST, pLA, Emax, Emin, Rsystemic,
      params.aorticStenosis, params.mitralRegurgitation
    );
    const rvResult = stepVentricle(
      this.rv, this.t, dt, period, RV_CONST, pRA, EmaxRV, EminRV, Rpulmonary, 0, 0
    );
    this.lv = lvResult.state;
    this.rv = rvResult.state;
    this.t += dt;

    const newTFrac = (this.t % period) / period;
    // Track per-beat min/max volume to report EDV/ESV from the live sim.
    if (newTFrac < this.prevTFrac) {
      this.lastBeatEDV = this.beatMaxV;
      this.lastBeatESV = this.beatMinV;
      this.lastBeatRegurg = this.beatRegurg;
      this.beatRegurg = 0;
      this.beatMinV = this.lv.V;
      this.beatMaxV = this.lv.V;
    } else {
      this.beatMinV = Math.min(this.beatMinV, this.lv.V);
      this.beatMaxV = Math.max(this.beatMaxV, this.lv.V);
    }
    this.prevTFrac = newTFrac;

    const edv = this.lastBeatEDV;
    const esv = this.lastBeatESV;
    this.beatRegurg += lvResult.Qregurg * dt;
    // Forward stroke volume: what reaches the aorta, i.e. the total emptied
    // minus what leaked back into the atrium. Cardiac output uses this.
    const effectiveStrokeVolume = Math.max(5, edv - esv - this.lastBeatRegurg);
    const cardiacOutput = (params.heartRate * effectiveStrokeVolume) / 1000;
    // Ejection fraction counts all the blood the ventricle empties, as an echo
    // measures it -- so it can look normal in mitral regurgitation even as
    // forward output falls.
    const ejectionFraction = edv > 0 ? (Math.max(5, edv - esv) / edv) * 100 : 0;
    const meanArterialPressure = params.diastolicBP + (params.systolicBP - params.diastolicBP) / 3;
    const cardiacWorkload = (params.heartRate * params.systolicBP) / 100; // rate-pressure product / 100
    const lvedp = 5 + 15 * (params.preload / 100);
    const coronaryPerfusionPressure = Math.max(0, params.diastolicBP - lvedp);

    const mitralOpen = lvResult.Qin > 0.5 ? 1 : 0;
    const aorticOpen = lvResult.Qout > 0.5 ? Math.max(0.15, 1 - params.aorticStenosis * 0.7) : 0;
    const tricuspidOpen = rvResult.Qin > 0.5 ? 1 : 0;
    const pulmonicOpen = rvResult.Qout > 0.5 ? 1 : 0;

    return {
      t: this.t,
      tFrac: newTFrac,
      period,
      ecg: ecgWaveform(newTFrac),
      vLV: this.lv.V,
      vRV: this.rv.V,
      vLA: 40 + 20 * preloadFactor + 15 * kick + mr * (18 + 30 * lvSystole),
      vRA: 35 + 15 * preloadFactor + 12 * kick,
      pLV: lvResult.Plv,
      pRV: rvResult.Plv,
      pLA,
      pRA,
      pAo: this.lv.Pao,
      pPA: this.rv.Pao,
      valves: { mitral: mitralOpen, tricuspid: tricuspidOpen, aortic: aorticOpen, pulmonic: pulmonicOpen },
      flowSystemicIn: rvResult.Qin,
      flowPulmonaryOut: rvResult.Qout,
      flowPulmonaryIn: lvResult.Qin,
      flowSystemicOut: lvResult.Qout,
      regurgFlow: lvResult.Qregurg,
      cardiacOutput,
      ejectionFraction: Math.min(90, ejectionFraction),
      edv,
      esv,
      meanArterialPressure,
      cardiacWorkload,
      coronaryPerfusionPressure,
      effectiveStrokeVolume,
    };
  }
}
