"""MTL notebook: param load, config print, jsonl record, patience logic."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# configs/mtl/example.yaml 과 동기화
TRAIN_PARAM_KEYS = (
    "run_name",
    "epochs",
    "batch_size",
    "lr",
    "age_weight",
    "variant",
    "weight_decay",
    "dropout",
    "patience",
)

RUN_META_KEYS = (
    "train_csv",
    "val_csv",
    "data_root",
    "output_weights",
    "yaml_config",
)


def load_train_params(cfg: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    d = defaults or {}
    raw = {
        "run_name": cfg.get("run_name", d.get("run_name", "mtl_nb_001")),
        "epochs": cfg.get("epochs", d.get("epochs", 15)),
        "batch_size": cfg.get("batch_size", d.get("batch_size", 32)),
        "lr": cfg.get("lr", d.get("lr", 1e-3)),
        "age_weight": cfg.get("age_weight", d.get("age_weight", 0.01)),
        "variant": cfg.get("variant", d.get("variant", "large")),
        "weight_decay": cfg.get("weight_decay", d.get("weight_decay", 1e-4)),
        "dropout": cfg.get("dropout", d.get("dropout", 0.2)),
        "patience": cfg.get("patience", d.get("patience", 4)),
    }
    return {
        "run_name": str(raw["run_name"]),
        "epochs": int(raw["epochs"]),
        "batch_size": int(raw["batch_size"]),
        "lr": float(raw["lr"]),
        "age_weight": float(raw["age_weight"]),
        "variant": str(raw["variant"]),
        "weight_decay": float(raw["weight_decay"]),
        "dropout": float(raw["dropout"]),
        "patience": int(raw["patience"]),
    }


def build_run_meta(
    *,
    train_csv: Path,
    val_csv: Path | None,
    data_root: Path,
    output_weights: Path,
    yaml_config: Path,
) -> dict[str, str]:
    return {
        "train_csv": str(train_csv.resolve()),
        "val_csv": str(val_csv.resolve()) if val_csv else "",
        "data_root": str(data_root.resolve()),
        "output_weights": str(output_weights.resolve()),
        "yaml_config": str(yaml_config.resolve()),
    }


def build_full_settings(
    params: dict[str, Any],
    meta: dict[str, str],
    *,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """노트북/로그용 전체 설정 (순서 고정)."""
    ordered: dict[str, Any] = {}
    for key in TRAIN_PARAM_KEYS:
        ordered[key] = params.get(key)
    for key in RUN_META_KEYS:
        ordered[key] = meta.get(key)
    if extras:
        for key, value in extras.items():
            ordered[key] = value
    return ordered


def print_train_config(
    params: dict[str, Any],
    meta: dict[str, str],
    *,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import pprint

    full = build_full_settings(params, meta, extras=extras)
    print("=== TRAIN_PARAMS ===")
    for key in TRAIN_PARAM_KEYS:
        print(f"  {key}: {full.get(key, '<missing>')}")
    print("=== DATA PATHS ===")
    for key in RUN_META_KEYS:
        print(f"  {key}: {full.get(key, '<missing>')}")
    if extras:
        print("=== MODEL / DATA (config) ===")
        for key, value in extras.items():
            print(f"  {key}: {value}")
    missing = [k for k in TRAIN_PARAM_KEYS if k not in params]
    if missing:
        print("  WARNING missing TRAIN_PARAM_KEYS:", missing)
    print("\n=== ALL SETTINGS (full) ===")
    pprint.pp(full, width=120, sort_dicts=False)
    return full


def params_for_log(params: dict[str, Any], meta: dict[str, str]) -> dict[str, Any]:
    """experiments/mtl_runs.jsonl 의 params 필드 (학습+경로)."""
    return {**params, **meta}


def make_log_record(
    params: dict[str, Any],
    meta: dict[str, str],
    metrics: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "params": params_for_log(params, meta),
        "metrics": metrics,
        "notes": notes,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_experiments_jsonl(path: Path):
    import pandas as pd

    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    flat = []
    for r in rows:
        row = {"run_id": r["run_id"], "notes": r.get("notes", "")}
        row.update({f"p_{k}": v for k, v in r.get("params", {}).items()})
        row.update({f"m_{k}": v for k, v in r.get("metrics", {}).items()})
        flat.append(row)
    return pd.DataFrame(flat)


def patience_stop_epoch(val_losses: list[float], patience: int) -> tuple[int | None, int]:
    """
    학습 셀과 동일한 조기 종료 규칙.
    Returns (stopped_at_epoch or None if never triggered, best_epoch_index_1based).
    """
    if patience < 1:
        raise ValueError("patience must be >= 1")
    best_val = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    stopped_at: int | None = None

    for epoch, val_loss in enumerate(val_losses, start=1):
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            stopped_at = epoch
            break

    return stopped_at, best_epoch
