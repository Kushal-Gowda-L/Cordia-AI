"""Plain-language summary of a HeartRiskModel prediction.

Written from templates, so it runs anywhere with no language model, server or
download (unlike src/models/llm_report.py, which needs Ollama).
"""
from __future__ import annotations


def _pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def plain_language_report(result: dict, top_factors: int = 5) -> str:
    lines: list[str] = []
    if result.get("risk") is None:
        return result.get("message", "No prediction could be made.")
    lines.append(
        f"Overall heart-risk index: {_pct(result['risk'])}, which is {result['band']} "
        f"(higher than about {result['percentile']:.0f}% of people in the reference datasets)."
    )
    lines.append("This index combines several models trained on different public datasets; each one's own "
                 "result is listed below. It supports, and does not replace, a clinician's assessment.")

    factors = result.get("factors") or []
    up = [f for f in factors if f["log_odds"] > 0][:top_factors]
    down = [f for f in factors if f["log_odds"] < 0][:top_factors]
    if up:
        lines.append("")
        lines.append("What raises the risk most:")
        lines += [f"  - {f['label']}" + (f" ({f['value']})" if f["value"] is not None else "") for f in up]
    if down:
        lines.append("")
        lines.append("What lowers it:")
        lines += [f"  - {f['label']}" + (f" ({f['value']})" if f["value"] is not None else "") for f in down]

    lines.append("")
    lines.append("By dataset:")
    for s in result["sources"]:
        if "probability" not in s:
            continue
        status = f"weight {_pct(s['share'])}" if s.get("used") else f"not used: {s['reason']}"
        lines.append(f"  - {s['title']}: {_pct(s['probability'])} chance of {s['outcome'].lower()} "
                     f"({s['percentile']:.0f}th percentile; {status})")

    if result.get("suggested_measurements"):
        lines.append("")
        lines.append("Adding these would make the estimate more reliable: "
                     + ", ".join(m["label"].lower() for m in result["suggested_measurements"]) + ".")
    return "\n".join(lines)
