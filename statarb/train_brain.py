# statarb/train_brain.py
"""
Trains and saves the BrainEngine P_win classifier.

This is the missing second half of the retrain workflow:
    1. statarb/dataset_builder_v2.py --engine equities   (builds the CSV)
    2. statarb/train_brain.py --engine equities          (this script: fits + saves)

dataset_builder_v2.py only ever writes data/brain_events_<engine>.csv — nothing
else in the repo calls BrainEngine.fit()/save_model(). Without this script,
main.py's `if os.path.exists(_MODEL_PATH): brain_engine.load_model(...)` check
always fails and the bot silently runs on the heuristic P_win fallback forever,
no matter how many times the dataset builder is re-run.

Usage:
    python statarb/train_brain.py --engine equities
    python statarb/train_brain.py --engine equities --data data/brain_events_equities.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# See statarb/dataset_builder_v2.py for why this is needed: running this file
# directly (`python statarb/train_brain.py`) puts statarb/ itself on sys.path,
# not the repo root, so `from statarb import ...` can't find the package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd  # noqa: E402

from statarb import config as cfg  # noqa: E402
from statarb.brain_engine import BrainEngine  # noqa: E402


def _model_path_for(engine: str) -> str:
    engines = getattr(cfg, "ENGINES", None)
    if isinstance(engines, dict) and engine in engines and "model_path" in engines[engine]:
        return str(engines[engine]["model_path"])
    # Fallback so this still works against an older/incomplete config.py.
    return f"models/brain_model_{engine}.joblib"


def train(engine: str, data_path: str | None, test_frac: float) -> None:
    csv_path = Path(data_path) if data_path else Path("data") / f"brain_events_{engine}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"Training data not found at {csv_path}.\n"
            f"Run this first:  python statarb/dataset_builder_v2.py --engine {engine}"
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty — nothing to train on.")

    date_col = "signal_dt" if "signal_dt" in df.columns else None
    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)

    # Chronological holdout: the most recent `test_frac` of events are held out
    # as an unbiased final test set. fit() already does walk-forward CV
    # internally for probability calibration — this is a SEPARATE, later split
    # so evaluate() reports performance the model has genuinely never seen,
    # rather than folds it was calibrated on.
    n_test = max(1, int(len(df) * test_frac))
    train_df, test_df = df.iloc[:-n_test], df.iloc[-n_test:]
    print(f"[train_brain] {len(df):,} events -> {len(train_df):,} train / "
          f"{len(test_df):,} held-out test (most recent {test_frac:.0%})")

    if len(train_df) < 50:
        print(f"[train_brain] WARNING: only {len(train_df)} training rows. "
              "The model will be unreliable — widen DATASET_START or lower "
              "ENTRY_Z in statarb/config.py to generate more events.")

    model_type = str(getattr(cfg, "BRAIN_MODEL_TYPE", "hgb"))
    engine_obj = BrainEngine(model_type=model_type)
    print(f"[train_brain] Fitting {model_type} model on {len(train_df):,} rows…")
    engine_obj.fit(train_df)

    print("[train_brain] Evaluating on held-out (never-seen) test set…")
    metrics = engine_obj.evaluate(test_df)
    print(f"[train_brain] Held-out AUC={metrics['auc']:.3f}  "
          f"Brier={metrics['brier']:.4f}  "
          f"mean_pred_prob={metrics['mean_prob']:.3f}  "
          f"actual_win_rate={metrics['win_rate']:.3f}  "
          f"n={int(metrics['n'])}")
    if metrics["auc"] < 0.55:
        print("[train_brain] NOTE: AUC is close to 0.5 (no better than chance). "
              "This can be normal with limited history — treat P_win from this "
              "model with caution until more data accumulates, and re-check "
              "after paper-trading builds up real trade_history.csv rows.")

    out_path = Path(_model_path_for(engine))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine_obj.save_model(str(out_path))
    print(f"[train_brain] Saved trained model -> {out_path}")
    print("[train_brain] Done. main.py will pick this up automatically on next start "
          "(it loads the model at cfg.ENGINES[engine]['model_path'] if the file exists).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="equities",
                   help="Engine name (must match a cfg.ENGINES key). Default: equities "
                        "(the engine main.py actually loads at runtime).")
    p.add_argument("--data", default=None,
                   help="Override path to the training CSV "
                        "(default: data/brain_events_<engine>.csv)")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="Fraction of the most recent events held out for final "
                        "evaluation (default: 0.2)")
    args = p.parse_args()
    train(args.engine, args.data, args.test_frac)


if __name__ == "__main__":
    main()
