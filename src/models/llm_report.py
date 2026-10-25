"""Offline LLM clinical report generation (methodology Section 6.4, step 10).

Runs entirely locally via Ollama — no patient data ever leaves the machine,
matching the methodology's offline requirement. This builds one carefully
structured prompt from the same real prediction/SHAP/DiCE data already used
elsewhere in this app and calls the local model once, rather than the
methodology's full LangGraph multi-agent pipeline (DataAgent -> ModelAgent
-> XAIAgent -> ReportAgent): the actual deliverable — an offline,
privacy-preserving, plain-English narration of a real prediction — doesn't
need four separate agent hops to reach a good result, and skipping
LangGraph keeps one fewer heavy dependency in a pipeline that already has
several. If a genuine multi-agent decomposition becomes useful later (e.g.
each stage needing its own retry/validation logic), this is the natural
place to add it.

Model substitution, stated plainly: `llama3.2:3b` (~2GB) is used instead of
the methodology's Llama-3 8B (~4.7GB+) — a smaller, faster download and
faster local inference on a laptop CPU, while still genuinely running
locally offline. Swap MODEL below to a larger model if better report
quality is worth the extra download/inference time.
"""
import json
import os
import urllib.error
import urllib.request

# Overridable so the Docker Compose setup (docker-compose.yml) can point
# this at the separate "ollama" service container instead of localhost.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
MODEL = "llama3.2:3b"


def build_prompt(report: dict) -> str:
    findings = "\n".join(f"- {f}" for f in report.get("findings", []))
    shap = "\n".join(
        f"- {f['feature']} = {f['value']} ({'raises' if f['impact'] > 0 else 'lowers'} the chance of heart disease by about {abs(f['impact']) * 100:.0f} percentage points)"
        for f in report.get("shap_factors", [])
    ) or "- not available"
    dice = "\n".join(
        f"- Changing {', '.join(f'{k} to {v}' for k, v in cf['changes'].items())} could move risk toward '{cf['target_class']}'"
        for cf in report.get("dice_counterfactuals", [])
    ) or "- no counterfactual scenario found"
    probs = ", ".join(f"{k}: {v}%" for k, v in report.get("class_probabilities", {}).items())
    reading = report.get("reading", {})

    return f"""You are writing a short, plain-English clinical summary for a patient's cardiovascular risk report. Be clear, calm, and factual — no alarming language, no medical jargon without a brief explanation, and never invent numbers not given below.

Patient: age {report.get('age')}
Chance of heart disease (total of all disease levels): {report.get('risk_percent')}%
Model's reading: {reading.get('verdict', '')} (the model's decision point is 50%; 35-65% is shown as borderline)
If heart disease is present, the most likely severity: {reading.get('likely_severity_if_disease')} ({reading.get('likely_severity_percent')}% among disease cases)
Raw five-outcome breakdown: {probs}
Do not describe the patient as having "no disease" just because "No disease" is the largest single outcome; use the total chance of heart disease and the reading above.

Clinical findings from their input values:
{findings or '- none outside common reference ranges'}

Top contributing factors (SHAP):
{shap}

What could change the outlook (DiCE counterfactuals):
{dice}

Write a 3-4 sentence summary covering: (1) the overall risk level in plain words, (2) the one or two biggest contributing factors, (3) one concrete, encouraging next step from the counterfactual data above if one exists. Do not use headers or bullet points, write flowing sentences."""


def generate_report(report: dict, timeout: int = 45) -> dict:
    prompt = build_prompt(report)
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return {"text": data.get("response", "").strip(), "available": True}
    except urllib.error.URLError as e:
        return {
            "text": "Offline AI summary unavailable: Ollama isn't reachable at localhost:11434. "
            "Run `ollama serve` and `ollama pull llama3.2:3b` first.",
            "available": False,
            "error": str(e),
        }
    except Exception as e:
        return {"text": f"Offline AI summary failed: {e}", "available": False, "error": str(e)}
