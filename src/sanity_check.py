import importlib
import json
from py_compile import compile as py_compile

from src.config import FIG_RUNS_DIR, METRICS_DIR, MODEL_COMPARISON_PATH, PROJECT_ROOT
from src.data_access import check_official_data_availability, load_processed_datasets


def check_imports():
    for module_name in [
        "src.data_access",
        "src.make_dataset",
        "src.train",
        "src.evaluate",
        "src.predict",
        "app.loaders",
    ]:
        importlib.import_module(module_name)


def check_official_data():
    check_official_data_availability()
    train, test = load_processed_datasets(source="remote", refresh=False, prefer_local_processed=False)
    if train.empty or test.empty:
        raise RuntimeError("Official data source is reachable, but the processed tables are empty.")


def check_artifacts():
    required_files = [
        METRICS_DIR / "model_info.json",
        METRICS_DIR / "best_run.json",
        METRICS_DIR / "validation_metrics.json",
        MODEL_COMPARISON_PATH,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required project artifact is missing: {path}")

    best_run = json.loads((METRICS_DIR / "best_run.json").read_text(encoding="utf-8"))
    run_id = str(best_run["run_id"]).strip()
    for path in [
        FIG_RUNS_DIR / run_id / "top_features.json",
        METRICS_DIR / "runs" / f"val_predictions_{run_id}.csv",
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Best-run artifact is missing: {path}")


def check_app_entrypoint():
    for path in [PROJECT_ROOT / "streamlit_app.py", PROJECT_ROOT / "app" / "streamlit_app.py"]:
        if not path.exists():
            raise FileNotFoundError(f"Streamlit entrypoint not found: {path}")
        py_compile(str(path), doraise=True)


def main():
    checks = [
        ("Imports", check_imports),
        ("Official Data", check_official_data),
        ("Artifacts", check_artifacts),
        ("App Entrypoint", check_app_entrypoint),
    ]

    failures: list[str] = []
    for label, fn in checks:
        try:
            fn()
            print(f"[OK] {label}")
        except Exception as exc:
            failures.append(f"[FAIL] {label}: {exc}")

    if failures:
        print("\n".join(failures))
        raise SystemExit(1)

    print("Sanity check passed. Repository is ready for public demo validation.")


if __name__ == "__main__":
    main()
