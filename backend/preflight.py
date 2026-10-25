"""Inspect the selected training interpreter without importing native torch libraries."""
import json
import subprocess

PROBE = '''import importlib.util, importlib.metadata, json
names = ['torch', 'torch_geometric', 'torch_scatter', 'torch_sparse', 'lightning', 'mlflow']
missing = [n for n in names if importlib.util.find_spec(n) is None]
try: version = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError: version = None
print(json.dumps({'missing': missing, 'torch_version': version}))
'''


def inspect_python(executable):
    try:
        result = subprocess.run([executable, "-c", PROBE], capture_output=True, text=True, timeout=15)
        if result.returncode:
            return {"ready": False, "issues": ["Training Python probe failed; verify GENCO_PYTHON points to a working interpreter."]}
        data = json.loads(result.stdout)
        issues = ["Missing training dependencies: " + ", ".join(data["missing"])] if data["missing"] else []
        version = data.get("torch_version")
        if version:
            pair = tuple(int(x) for x in version.split("+")[0].split(".")[:2])
            if not (pair >= (2, 10) and pair < (2, 13)):
                issues.append(f"Current torch {version} does not satisfy the local GraphKit requirement >=2.10,<2.13.")
        return {**data, "ready": not issues, "issues": issues}
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        return {"ready": False, "issues": ["Cannot inspect training Python; check GENCO_PYTHON and its environment."]}
