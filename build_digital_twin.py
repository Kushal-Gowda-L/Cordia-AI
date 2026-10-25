"""Longitudinal patient digital twin (methodology step 9 area / README intro).

Honest scope note, read before trusting the trajectory curves this prints:
the *real* Framingham Heart Study is longitudinal (the same patients
examined every ~2 years across decades), which is exactly what a genuine
digital twin needs -- but that multi-exam data is NIH/BioLINCC controlled
access (a formal application, days-to-weeks turnaround), not something
obtainable in this session. What's downloaded here
(`datasets/framingham/framingham.csv`, 4,240 patients) is the public
*cross-sectional* teaching subset: one baseline exam per patient plus
whether they developed CHD within the next 10 years. There is no
per-patient repeated-measurement trajectory in this file at all.

So this script does NOT reconstruct anyone's real trajectory. It:
1. Trains a risk model (age, sex, cholesterol, blood pressure, BMI,
   smoking, diabetes, hypertension -> 10-year CHD probability) on the
   cross-sectional data -- a legitimate baseline risk model, comparable to
   the actual clinical Framingham Risk Score.
2. Fits *population-level* linear drift rates for the modifiable risk
   factors (total cholesterol, systolic/diastolic BP, BMI) against age,
   from the same cross-sectional data -- i.e. "how much higher is this
   factor, on average, in an X-year-older member of this same population,"
   not "how much does any specific person's factor actually change."
3. Projects a synthetic future trajectory for one patient by repeatedly
   aging them forward and nudging their modifiable factors along that
   population drift, re-scoring risk at each simulated year.

This is a real, published technique (population-drift microsimulation),
used exactly because true longitudinal per-patient data is rarely
available -- but it is a simulated projection, not a measured one, and the
paper should describe it as such rather than as "tracking the patient over
time."
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "framingham" / "framingham.csv"
CHECKPOINT_DIR = ROOT / "checkpoints"

FEATURE_COLS = ["age", "male", "totChol", "sysBP", "diaBP", "BMI", "currentSmoker", "diabetes", "prevalentHyp"]
DRIFT_COLS = ["totChol", "sysBP", "diaBP", "BMI"]  # factors that plausibly change with age; sex/smoking/diabetes held fixed
TARGET = "TenYearCHD"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURE_COLS + [TARGET]].copy()
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())
    return df


def fit_risk_model(df: pd.DataFrame) -> tuple[Pipeline, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLS], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    model = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def fit_drift_rates(df: pd.DataFrame) -> dict:
    """Population-level 'per year of age' slope for each modifiable factor,
    from a simple linear fit against age on the cross-sectional cohort."""
    rates = {}
    for col in DRIFT_COLS:
        reg = LinearRegression().fit(df[["age"]], df[col])
        rates[col] = float(reg.coef_[0])
    return rates


def simulate_trajectory(model: Pipeline, drift_rates: dict, patient: dict, years: int = 10) -> pd.DataFrame:
    rows = []
    state = dict(patient)
    for year in range(years + 1):
        x = pd.DataFrame([{col: state[col] for col in FEATURE_COLS}])
        risk = model.predict_proba(x)[0, 1]
        rows.append({"year": year, "age": state["age"], "risk_pct": round(risk * 100, 1), **{c: round(state[c], 1) for c in DRIFT_COLS}})
        state["age"] += 1
        for col in DRIFT_COLS:
            state[col] += drift_rates[col]
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    print(f"Framingham cross-sectional cohort: {len(df)} patients, {df[TARGET].mean()*100:.1f}% developed CHD within 10 years")

    model, auc = fit_risk_model(df)
    print(f"Risk model (logistic regression, class-balanced): held-out AUC = {auc:.4f}")

    drift_rates = fit_drift_rates(df)
    print("\nPopulation drift rates (per year of age, fit on this cohort):")
    for col, rate in drift_rates.items():
        print(f"  {col}: {rate:+.3f} / year")

    example_patients = {
        "35yo non-smoker, healthy vitals": {
            "age": 35, "male": 0, "totChol": 190, "sysBP": 115, "diaBP": 75, "BMI": 23,
            "currentSmoker": 0, "diabetes": 0, "prevalentHyp": 0,
        },
        "55yo smoker, hypertensive, elevated cholesterol": {
            "age": 55, "male": 1, "totChol": 260, "sysBP": 150, "diaBP": 95, "BMI": 29,
            "currentSmoker": 1, "diabetes": 0, "prevalentHyp": 1,
        },
    }

    for name, patient in example_patients.items():
        print(f"\n{'=' * 70}\nSimulated 10-year risk trajectory — {name}\n{'=' * 70}")
        traj = simulate_trajectory(model, drift_rates, patient, years=10)
        print(traj.to_string(index=False))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "drift_rates": drift_rates, "feature_cols": FEATURE_COLS}, CHECKPOINT_DIR / "digital_twin.joblib")
    print("\nSaved: checkpoints/digital_twin.joblib")
    print(
        "\nReminder: trajectories above are a population-drift SIMULATION from "
        "cross-sectional data, not a measured individual trajectory. See this "
        "script's module docstring / README.md's Digital Twin section."
    )


if __name__ == "__main__":
    main()
