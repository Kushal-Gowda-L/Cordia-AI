"""How the model's five-class output is read and shown, in two steps.

The model gives five probabilities (No disease, Mild, Moderate, Severe,
Critical). Picking the largest single one to answer "does this patient have
heart disease?" can contradict the total: 45% "No disease" beats each disease
level on its own, yet the disease levels together make 55%. So the reading is:

  Step 1 -- Is heart disease likely?  The total chance of disease,
            1 - P(No disease), with the model's 50% decision point.
  Step 2 -- If disease is present, how severe?  The four disease levels
            rescaled to add up to 100% among disease cases, i.e. the
            conditional probability P(level | disease) = P(level) / P(disease).

The 35% / 65% zone limits are this application's display convention for
flagging results close to the 50% decision point. They are NOT clinical
categories and come from no guideline.
"""

DECISION_POINT = 50.0  # the model's disease / no-disease cut-off, in percent
UNLIKELY_BELOW = 35.0
LIKELY_FROM = 65.0


def zone(disease_percent: float) -> dict:
    """Step 1 on its own -- needs only the total chance of disease."""
    predicts_disease = disease_percent >= DECISION_POINT
    if disease_percent < UNLIKELY_BELOW:
        band, verdict = "unlikely", "Heart disease unlikely"
    elif disease_percent < LIKELY_FROM:
        band = "borderline"
        verdict = "Borderline — leans towards " + ("heart disease" if predicts_disease else "no heart disease")
    else:
        band, verdict = "likely", "Heart disease likely"
    return {"band": band, "verdict": verdict, "predicts_disease": predicts_disease}


def read_risk(probs: list[float], classes: list[str]) -> dict:
    """Both steps, from the model's class probabilities (classes[0] is "No disease")."""
    disease = 1.0 - float(probs[0])
    reading = {
        "disease_percent": round(disease * 100, 1),
        "no_disease_percent": round(float(probs[0]) * 100, 1),
        "decision_point": DECISION_POINT,
        "zones": {"unlikely_below": UNLIKELY_BELOW, "likely_from": LIKELY_FROM},
        **zone(disease * 100),
    }
    if disease > 1e-6:
        conditional = {classes[i]: float(probs[i]) / disease for i in range(1, len(classes))}
        top = max(conditional, key=conditional.get)
        reading["severity_if_disease"] = {k: round(v * 100, 1) for k, v in conditional.items()}
        reading["likely_severity_if_disease"] = top
        reading["likely_severity_percent"] = round(conditional[top] * 100, 1)
    else:
        reading["severity_if_disease"] = {c: 0.0 for c in classes[1:]}
        reading["likely_severity_if_disease"] = None
        reading["likely_severity_percent"] = None
    return reading
