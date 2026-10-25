"""Keep measured numbers deterministic; ask the model for cited interpretation."""
import math
import re
from decimal import Decimal, InvalidOperation

def observation_schema(measurements):
    observation = {"type": "object", "properties": {
        "evidence": {"type": "array", "items": {"type": "string", "enum": list(measurements)}, "minItems": 1, "maxItems": 4},
        "text": {"type": "string", "maxLength": 600}},
        "required": ["evidence", "text"], "additionalProperties": False}
    return {"type": "object", "properties": {"observations": {"type": "array", "items": observation, "minItems": 1, "maxItems": 4}},
            "required": ["observations"], "additionalProperties": False}

def validate_observations(raw, measurements):
    """Accept source-key aliases and rounded source numbers, never novel numbers."""
    accepted = []
    if not isinstance(raw, list): return accepted
    for item in raw[:4]:
        if not isinstance(item, dict): continue
        keys, text = item.get("evidence"), item.get("text")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys): continue
        keys = [k.removeprefix("measurements.") for k in keys]
        if not all(k in measurements for k in keys) or not isinstance(text, str) or not text.strip() or len(text) > 600: continue
        # No safety/stability standard or dynamic study is present in this evidence contract.
        unsupported = ("stable operation", "normal voltage", "standards compliant", "guaranteed safe operation",
                       "significant", "very fast", "quickly", "excellent", "negligible")
        if any(claim in text.casefold() for claim in unsupported): continue
        values = [Decimal(str(measurements[k]["value"])) for k in keys]
        valid = True
        for token in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
            try:
                number = Decimal(token)
                if len(token) > 50 or abs(number.as_tuple().exponent) > 100:
                    valid = False
                    break
                tolerance = Decimal(0) if '.' not in token and 'e' not in token.lower() else Decimal(10) ** number.as_tuple().exponent / 2
                if not any(abs(number - value) <= tolerance + Decimal('1e-14') for value in values):
                    valid = False
                    break
            except InvalidOperation:
                valid = False
        if valid: accepted.append({"evidence": keys, "text": text})
    return accepted

LABELS = {
    "n_bus": ("Bus count", "count"), "n_gen": ("Generator count", "count"), "n_branch": ("Branch count", "count"),
    "vm_min_pu": ("Minimum voltage magnitude", "p.u."), "vm_max_pu": ("Maximum voltage magnitude", "p.u."),
    "max_p_mismatch_mw": ("Maximum active-power mismatch", "MW"), "max_q_mismatch_mvar": ("Maximum reactive-power mismatch", "Mvar"),
    "p_branch_loss_mw": ("Total branch active loss", "MW"), "p_load_mw": ("Total active load", "MW"),
    "warm_ac_median_seconds": ("Median warm AC solve time", "s"),
    "max_loading_percent": ("Maximum branch loading", "%"),
}

def compact(facts):
    measurements = {}
    sources = []
    limits = []
    if "runs" in facts:
        for index, run in enumerate(facts["runs"]):
            job = run["job"]
            sources.append({"id": job["id"], "kind": job["kind"], "status": job["status"],
                            "task": job.get("task"), "case": job.get("case"),
                            "data_path": job.get("data_path"), "parameters": job.get("parameters", {})})
            for metric in run.get("metrics", []):
                if metric["points"]:
                    key = f"run{index}.{metric['name']}"
                    measurements[key] = {"label": metric["name"] + " (last recorded value)", "value": metric["points"][-1]["value"],
                                         "unit": "original metric unit", "source": job["id"] + "/mlruns/**/metrics/" + metric["name"]}
            for artifact in run.get("evidence", []):
                summary = artifact["summary"]
                if isinstance(summary, dict) and "numeric_summary" in summary:
                    for col, stats in list(summary["numeric_summary"].items())[:25]:
                        key = f"run{index}.{artifact['file']}.{col}.mean"
                        measurements[key] = {"label": col + " (mean of scanned rows)", "value": stats["mean"], "unit": "original column unit",
                                             "source": job["id"] + "/" + artifact["file"]}
                if isinstance(summary, dict) and "vm_rmse_vs_saved_datakit" in summary:
                    measurements[f"run{index}.vm_rmse"] = {"label": "Vm RMSE versus saved DataKit scenario",
                        "value": summary["vm_rmse_vs_saved_datakit"], "unit": "p.u.", "source": job["id"]+"/"+artifact["file"]}
                if isinstance(summary, dict) and "converged" in summary:
                    sources[-1]["converged"] = summary["converged"]
                    sources[-1]["settings"] = summary.get("settings", {})
                    for key, (title, unit) in {**LABELS, "seconds": ("Per-run time (including import and conversion)", "s")}.items():
                        value = summary.get(key)
                        if isinstance(value, (float, int)) and math.isfinite(value):
                            measurements[f"run{index}.{key}"] = {"label": title, "value": value, "unit": unit,
                                "source": job["id"]+"/"+artifact["file"]}
            if job["status"] != "completed":
                limits.append(job["id"] + " did not complete; no training-quality inference is supported.")
            if not run.get("metrics"):
                limits.append(job["id"] + " has no recorded training metrics.")
        limits.append("Cross-run comparisons require matching tasks, splits, and scenarios. A CSV column mean is not automatically a model error metric.")
    else:
        for label, summary in (("primary", facts.get("primary")), ("comparison", facts.get("comparison"))):
            if not summary: continue
            sources.append({"id": label, "case": summary.get("case"), "kind": facts.get("provenance"),
                            "ac_converged": summary.get("ac_converged"), "source": "test/results/"+str(summary.get("case"))})
            for key, (title, unit) in LABELS.items():
                value = summary.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    measurements[f"{label}.{key}"] = {"label": title, "value": value, "unit": unit,
                                                      "source": str(summary.get("case")) + "/summary.json or bus_results.csv"}
            validation = summary.get("independent_validation", {})
            if validation.get("max_vm_error_pu") is not None:
                measurements[f"{label}.validation_vm"] = {"label": "Maximum Vm difference versus independent PYPOWER",
                    "value": validation["max_vm_error_pu"], "unit": "p.u.", "source": str(summary.get("case")) + "/bus_results.csv"}
        limits.append("These saved DataKit numerical power-flow results are not GENCO neural-network training or inference performance.")
        if facts.get("comparison") and facts["comparison"].get("case") != facts["primary"].get("case"):
            limits.append("Different grid cases permit descriptive comparison only, not solver accuracy or speed rankings.")
    return {"measurements": measurements, "sources": sources, "limitations": limits}

def render(payload, observations):
    def safe(text):
        return str(text).replace("|", "/").replace("\n", " ")
    lines = ["# GENCO experiment report", "", "## Result sources", ""]
    for src in payload["sources"]:
        parts = [str(src.get("id", "result")), str(src.get("kind", "result"))]
        for key in ("case", "task", "status"):
            if src.get(key):
                parts.append(str(src[key]))
        if src.get("source"):
            parts.append("source: " + str(src["source"]))
        lines.append("- " + safe(" · ".join(parts)))
    lines += ["", "## Verified measurements", "", "| Evidence key | Metric | Value | Unit |", "|---|---|---:|---|"]
    for key, measurement in payload["measurements"].items():
        lines.append(f"| {safe(key)} | {safe(measurement['label'])} | {measurement['value']:.8g} | {safe(measurement['unit'])} |")
    if not payload["measurements"]:
        lines.append("| — | No numeric measurements available | — | — |")
    lines += ["", "## Model interpretation", ""]
    for obs in observations:
        lines.append("- " + obs["text"] + " (evidence: " + ", ".join(obs["evidence"]) + ")")
    if not observations: lines.append("The model returned no interpretation with verifiable citations; measurements still come from actual result files.")
    lines += ["", "## Comparison scope and limitations", ""]
    lines += ["- " + limit for limit in payload["limitations"]]
    lines.append("- Static power-flow values and voltage range cannot alone prove dynamic stability or operational compliance. Expert review remains necessary.")
    return "\n".join(lines) + "\n"
