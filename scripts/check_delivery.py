"""Read-only preflight for a local GENCO Studio checkout.

Run from the repository root with: python -m scripts.check_delivery
"""
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env.local", override=False)
from backend import service


def main():
    problems = []
    print(f"OpenGridFM source: {service.SOURCE}")
    if not service.GRAPHKIT.is_dir():
        problems.append("GraphKit not found. Set GENCO_ROOT to the OpenGridFM directory.")
    print(f"GraphKit configurations: {len(service.CONFIGS)}")
    if not service.CONFIGS:
        problems.append("No YAML configurations found under gridfm-graphkit/examples/config.")
    print(f"Saved AC cases: {', '.join(service.CASES) or 'none'}")
    if not service.CASES:
        print("Warning: saved DataKit AC comparisons are unavailable without test/results CSVs.")

    built = Path(service.ROOT, "dist", "index.html").is_file()
    print(f"Web build: {'present' if built else 'missing'}")
    if not built:
        problems.append("Run npm run build before delivery.")

    if service.CONFIGS:
        name = "hello_world_train_case14.yaml" if "hello_world_train_case14.yaml" in service.CONFIGS else next(iter(service.CONFIGS))
        try:
            result = service.training_preflight(service.Training(config=name))
            print(f"Training preflight ({name}): {'ready' if result['ready'] else 'not ready'}")
            for issue in result["issues"]:
                print(f"  - {issue}")
        except Exception as exc:
            print(f"Training preflight could not complete: {exc}")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 2
    print("Core local checkout is ready. Training and optional providers have separate prerequisites.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
