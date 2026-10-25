"""Bounded summaries of real result artifacts for reports."""
import csv
import json
import math
from pathlib import Path

def summarize_csv(path):
    stats = {}
    count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            count += 1
            if count > 100000:
                break
            for key in columns[:80]:
                try: value = float(row[key])
                except (ValueError, TypeError, KeyError): continue
                if not math.isfinite(value): continue
                s = stats.setdefault(key, {"count": 0, "mean": 0., "min": value, "max": value})
                s["count"] += 1
                s["mean"] += (value-s["mean"])/s["count"]
                s["min"] = min(value, s["min"])
                s["max"] = max(value, s["max"])
    return {"rows_scanned": min(count, 100000), "truncated": count > 100000, "columns": columns, "numeric_summary": stats}

def evidence(folder):
    result = []
    used = 0
    for path in sorted(folder.rglob("*")):
        if len(result) >= 20 or used > 30000:
            break
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            continue
        if path.suffix not in (".json", ".csv") or path.name in ("job.json",):
            continue
        if path.stat().st_size > 50_000_000:
            continue
        try:
            value = summarize_csv(path) if path.suffix == ".csv" else json.loads(path.read_text()) if path.stat().st_size < 20000 else {"omitted": "JSON too large"}
            size = len(json.dumps(value))
            if used + size > 35000:
                value = {"omitted": "report evidence budget exceeded"}
            result.append({"file": str(path.relative_to(folder)), "summary": value})
            used += len(json.dumps(value))
        except (OSError, ValueError, UnicodeError):
            continue
    return result

def curves(folder):
    results = []
    for path in folder.glob("mlruns/**/metrics/*"):
        if not path.is_file() or len(results) >= 30:
            continue
        points = []
        for line in path.read_text(errors="replace").splitlines()[-1000:]:
            cells = line.split()
            try:
                value = float(cells[1])
                if math.isfinite(value):
                    points.append({"step": int(cells[2]), "value": value})
            except (ValueError, IndexError):
                continue
        if points:
            results.append({"name": path.name, "points": points})
    return results
