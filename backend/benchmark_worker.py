"""Run an actual AC solver on the supplied MATPOWER case in a child process."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

def load_case(path):
    # The local inputs are numeric MATPOWER v2 tables, not arbitrary MATLAB code.
    import numpy as np
    text = Path(path).read_text()
    text = re.sub(r"%[^\n]*", "", text)
    base = re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)", text)
    if not base:
        raise ValueError("Only numeric MATPOWER v2 cases are supported")
    case = {"version": "2", "baseMVA": float(base.group(1))}
    for name in ("bus", "gen", "branch"):
        match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
        if not match:
            raise ValueError(f"Missing numeric {name} table")
        rows = [[float(x) for x in row.replace(",", " ").split()] for row in match.group(1).split(";") if row.strip()]
        case[name] = np.asarray(rows, dtype=float)
    return case

def solve(case_path, engine, out):
    import numpy as np
    case = load_case(case_path)
    start = time.perf_counter()
    if engine == "pandapower":
        import pandapower as pp
        from pandapower.converter.pypower.from_ppc import from_ppc
        net = from_ppc(case, f_hz=60, validate_conversion=False)
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="flat",
                 enforce_q_lims=False, tolerance_mva=1e-8, max_iteration=80, numba=False)
        if not net.converged:
            raise RuntimeError("pandapower did not converge")
        bus_ids = case["bus"][:, 0].astype(int)
        ordered = net.res_bus.loc[bus_ids]
        buses = [{"bus": int(bus), "vm": float(vm), "va": float(va)} for bus, vm, va in
                 zip(bus_ids, ordered.vm_pu.to_numpy(), ordered.va_degree.to_numpy())]
        version = pp.__version__
    elif engine == "matpower":
        octave = os.getenv("OCTAVE_BIN") or shutil.which("octave")
        root = os.getenv("MATPOWER_ROOT")
        if not octave or not root or not Path(root).is_dir():
            raise RuntimeError("MATPOWER requires octave and MATPOWER_ROOT pointing to an installed MATPOWER directory")
        def quote(value): return str(value).replace("'", "''")
        script = Path(out).parent / "benchmark.m"
        csv_path = Path(out).parent / "matpower_bus.csv"
        script.write_text(
            f"addpath(genpath('{quote(Path(root).resolve())}'));\n"
            f"mpc=loadcase('{quote(Path(case_path).resolve())}');\n"
            "opt=mpoption('verbose',0,'out.all',0,'pf.tol',1e-10,'pf.nr.max_it',80,'pf.enforce_q_lims',0);\n"
            "[r,ok]=runpf(mpc,opt); if ~ok, error('MATPOWER did not converge'); end;\n"
            f"dlmwrite('{quote(csv_path)}',r.bus(:,[1 8 9]),'precision',16);\n")
        subprocess.run([octave, "--quiet", "--no-gui", str(script)], check=True, timeout=180)
        rows = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        buses = [{"bus": int(r[0]), "vm": float(r[1]), "va": float(r[2])} for r in rows]
        version = "local MATPOWER via Octave"
    else:
        raise ValueError("Unsupported engine")
    result = {"engine": engine, "version": version, "case_path": str(case_path), "seconds": time.perf_counter()-start,
              "converged": True, "buses": buses, "settings": {"algorithm": "Newton AC", "enforce_q_lims": False,
              "load_scale": 1, "voltage_setpoints": "generator VG", "timing": "includes conversion and startup; not inference-only"}}
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False))
    summary = {"engine": engine, "n_bus": len(buses), "converged": True,
               "vm_min_pu": min(b["vm"] for b in buses), "vm_max_pu": max(b["vm"] for b in buses),
               "seconds": result["seconds"], "settings": result["settings"]}
    Path(out).with_name("benchmark_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"{engine}: converged, {len(buses)} buses", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--engine", choices=["pandapower", "matpower"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    solve(args.case, args.engine, args.out)
