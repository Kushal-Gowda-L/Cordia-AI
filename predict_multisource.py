"""Score one patient with the multi-source heart-risk model and explain it.

    python predict_multisource.py --example
    python predict_multisource.py --profile patient.json
    python predict_multisource.py --profile patient.json --what-if '{"smoker": 0, "systolic_bp": 125}'
    python predict_multisource.py --profile patient.json --json     # full result as JSON

A profile is a JSON object of the fields in src/multisource/profile.py; every
field is optional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.multisource.fusion import DEFAULT_DIR, HeartRiskModel
from src.multisource.report import plain_language_report

EXAMPLE = {
    "age": 58, "sex": "male", "height_cm": 175, "weight_kg": 92,
    "systolic_bp": 148, "diastolic_bp": 92, "bp_meds": 0, "total_cholesterol": 245, "glucose": 104,
    "diabetes": 0, "smoker": 1, "cigs_per_day": 15, "heavy_alcohol": 0, "physically_active": 0,
    "general_health": "fair", "physical_health_days": 4, "mental_health_days": 2, "difficulty_walking": 0,
    "stroke_history": 0, "sleep_hours": 6, "resting_heart_rate": 82,
    "chest_pain_type": "asymptomatic", "max_heart_rate": 132, "exercise_angina": 1, "st_depression": 1.8,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--example", action="store_true", help="use a built-in example patient")
    ap.add_argument("--what-if", help="JSON object of fields to change")
    ap.add_argument("--checkpoints", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    if not (args.profile or args.example):
        ap.error("give --profile FILE or --example")
    profile = EXAMPLE if args.example else json.loads(args.profile.read_text())
    model = HeartRiskModel.load(args.checkpoints)

    if args.what_if:
        out = model.what_if(profile, json.loads(args.what_if), explain=True)
        if args.json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        print("BEFORE\n" + plain_language_report(out["before"]))
        print("\nAFTER\n" + plain_language_report(out["after"]))
        if out["risk_change"] is not None:
            print(f"\nChange in risk index: {100 * out['risk_change']:+.1f} percentage points")
        return 0

    result = model.predict(profile)
    print(json.dumps(result, indent=2, default=str) if args.json else plain_language_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
