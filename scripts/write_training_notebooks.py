"""notebooks/yolo, mtl, slm 학습 ipynb 생성."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_SRC = Path(r"C:\Users\moonjintae\projects\face_attr_mtl\notebooks\yolo_train_and_tune.ipynb")

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}


def _cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


def _save(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


ROOT_SNIPPET = '''
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path.cwd().resolve()
for _ in range(5):
    if (ROOT / "config.py").is_file():
        break
    if ROOT.parent == ROOT:
        break
    ROOT = ROOT.parent
else:
    raise FileNotFoundError("emotion-bot 루트(config.py)를 찾지 못했습니다.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

def log_run(log_path: Path, params: dict, metrics: dict, notes: str = "") -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "params": params,
        "metrics": metrics,
        "notes": notes,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")
    return record

def load_log(log_path: Path) -> pd.DataFrame:
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    flat = []
    for r in rows:
        row = {"run_id": r["run_id"], "notes": r.get("notes", "")}
        row.update({f"p_{k}": v for k, v in r.get("params", {}).items()})
        row.update({f"m_{k}": v for k, v in r.get("metrics", {}).items()})
        flat.append(row)
    return pd.DataFrame(flat)
'''


def write_yolo() -> Path:
    dst = ROOT / "notebooks" / "yolo" / "yolo_train_and_tune.ipynb"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if YOLO_SRC.is_file():
        shutil.copy2(YOLO_SRC, dst)
    else:
        raise FileNotFoundError(YOLO_SRC)
    return dst


def write_mtl() -> Path:
    path = ROOT / "notebooks" / "mtl" / "mtl_train_and_tune.ipynb"
    setup = ROOT_SNIPPET + '''
EXPERIMENTS_LOG = ROOT / "experiments" / "mtl_runs.jsonl"
print("ROOT:", ROOT)
print("log:", EXPERIMENTS_LOG)
'''
    train = '''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from models.mtl_model import FaceAttributeMTL
from models.preprocess import IMAGENET_MEAN, IMAGENET_STD

class FaceAttributeDataset(Dataset):
    def __init__(self, csv_path, root, input_size=224):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / str(row["image_path"])).convert("RGB")
        return self.transform(img), int(row["expression_id"]), float(row["age"])

YAML_CFG = ROOT / "configs" / "mtl" / "example.yaml"
cfg = yaml.safe_load(YAML_CFG.read_text(encoding="utf-8")) if YAML_CFG.is_file() else {}
TRAIN_PARAMS = {
    "run_name": cfg.get("run_name", "mtl_nb_001"),
    "epochs": cfg.get("epochs", 15),
    "batch_size": cfg.get("batch_size", 32),
    "lr": cfg.get("lr", 1e-3),
    "age_weight": cfg.get("age_weight", 0.01),
    "variant": cfg.get("variant", config.MOBILENET_VARIANT),
}
TRAIN_CSV = Path(cfg.get("train_csv", "data/mtl/train.csv"))
VAL_CSV = Path(cfg.get("val_csv", "data/mtl/val.csv")) if cfg.get("val_csv") else None
DATA_ROOT = Path(cfg.get("data_root", "data/mtl"))
OUTPUT = config.MTL_WEIGHTS
TRAIN_PARAMS
'''
    run_train = '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not TRAIN_CSV.is_file():
    raise FileNotFoundError(f"train CSV 없음: {TRAIN_CSV}")

model = FaceAttributeMTL(
    num_expressions=config.NUM_EXPRESSIONS,
    variant=TRAIN_PARAMS["variant"],
    pretrained_backbone=True,
).to(device)
train_loader = DataLoader(
    FaceAttributeDataset(TRAIN_CSV, DATA_ROOT, config.INPUT_SIZE),
    batch_size=TRAIN_PARAMS["batch_size"],
    shuffle=True,
    num_workers=0,
)
val_loader = None
if VAL_CSV and VAL_CSV.is_file():
    val_loader = DataLoader(
        FaceAttributeDataset(VAL_CSV, DATA_ROOT, config.INPUT_SIZE),
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0,
    )
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_PARAMS["lr"])
ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
history = {"train_loss": [], "val_loss": []}
best_val = float("inf")

for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
    model.train()
    total = 0.0
    for images, expr_ids, ages in train_loader:
        images, expr_ids = images.to(device), expr_ids.to(device)
        ages = ages.to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        expr_logits, pred_ages = model(images)
        loss = ce(expr_logits, expr_ids) + TRAIN_PARAMS["age_weight"] * mse(pred_ages, ages)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
    train_loss = total / len(train_loader.dataset)
    history["train_loss"].append(train_loss)
    msg = f"epoch {epoch} train_loss={train_loss:.4f}"
    val_loss = None
    if val_loader:
        model.eval()
        vtotal = 0.0
        with torch.inference_mode():
            for images, expr_ids, ages in val_loader:
                images, expr_ids = images.to(device), expr_ids.to(device)
                ages = ages.to(device, dtype=torch.float32)
                expr_logits, pred_ages = model(images)
                vloss = ce(expr_logits, expr_ids) + TRAIN_PARAMS["age_weight"] * mse(pred_ages, ages)
                vtotal += float(vloss.item()) * images.size(0)
        val_loss = vtotal / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        msg += f" val_loss={val_loss:.4f}"
        if val_loss < best_val:
            best_val = val_loss
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT)
            msg += " [saved]"
    else:
        torch.save(model.state_dict(), OUTPUT)
    print(msg)

metrics = {"final_train_loss": history["train_loss"][-1], "best_val_loss": best_val if val_loader else None}
record = log_run(EXPERIMENTS_LOG, TRAIN_PARAMS, metrics, notes="mtl notebook train")
print("weights:", OUTPUT)
record
'''
    compare = '''
df = load_log(EXPERIMENTS_LOG)
display(df.sort_values("run_id") if not df.empty else "로그 없음")
if not df.empty and "m_final_train_loss" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(x="run_id", y=[c for c in df.columns if c.startswith("m_")], kind="bar", ax=ax, rot=45)
    plt.tight_layout()
    plt.show()
'''
    nb = {
        "cells": [
            _cell_md(
                "# MobileNetV3 MTL (표정·나이) — 학습 & 튜닝\\n\\n"
                "- CNN 백본 공유 Multi-Task Learning\\n"
                "- 로그: `experiments/mtl_runs.jsonl`\\n"
                "- CSV: `image_path,expression_id,age`\\n"
                "- 가중치: `weights/face_mtl_mobilenetv3.pt`"
            ),
            _cell_code(setup),
            _cell_md("## 1. 파라미터"),
            _cell_code(train),
            _cell_md("## 2. 데이터 경로 확인"),
            _cell_code(
                "print('train_csv:', TRAIN_CSV, TRAIN_CSV.is_file())\\n"
                "print('val_csv:', VAL_CSV, VAL_CSV.is_file() if VAL_CSV else None)\\n"
                "print('data_root:', DATA_ROOT, DATA_ROOT.is_dir())"
            ),
            _cell_md("## 3. 학습"),
            _cell_code(run_train),
            _cell_md("## 4. run 비교"),
            _cell_code(compare),
        ],
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _save(nb, path)
    return path


def write_slm() -> Path:
    path = ROOT / "notebooks" / "slm" / "slm_train_and_tune.ipynb"
    setup = ROOT_SNIPPET + '''
EXPERIMENTS_LOG = ROOT / "experiments" / "slm_runs.jsonl"
print("ROOT:", ROOT)
print("log:", EXPERIMENTS_LOG)
'''
    cfg_cell = '''
YAML_CFG = ROOT / "configs" / "slm" / "example.yaml"
cfg = yaml.safe_load(YAML_CFG.read_text(encoding="utf-8")) if YAML_CFG.is_file() else {}
TRAIN_PARAMS = {
    "run_name": cfg.get("run_name", "slm_nb_001"),
    "model_name": cfg.get("model_name", "meta-llama/Llama-3.2-1B-Instruct"),
    "epochs": cfg.get("epochs", 3),
    "batch_size": cfg.get("batch_size", 4),
    "lr": cfg.get("lr", 2e-5),
    "max_seq_len": cfg.get("max_seq_len", 512),
}
DATA_PATH = Path(cfg.get("data_path", "data/slm/train.jsonl"))
TRAIN_PARAMS
'''
    train_placeholder = '''
# SLM 학습 코드는 사용하는 프레임워크(HuggingFace TRL, unsloth 등)에 맞게 채우세요.
RUN_TRAINING = False

if not DATA_PATH.is_file():
    print(f"데이터 없음: {DATA_PATH}")
    print("예: JSONL 한 줄당 {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}")
elif RUN_TRAINING:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install transformers datasets accelerate")

    ds = load_dataset("json", data_files=str(DATA_PATH), split="train")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_PARAMS["model_name"])
    model = AutoModelForCausalLM.from_pretrained(TRAIN_PARAMS["model_name"])

    # TODO: tokenize ds, Trainer(...).train()
    metrics = {"status": "completed"}
    log_run(EXPERIMENTS_LOG, TRAIN_PARAMS, metrics, notes="slm placeholder")
else:
    print("RUN_TRAINING=False — 파라미터만 확인했습니다.")
'''
    compare = '''
df = load_log(EXPERIMENTS_LOG)
display(df if not df.empty else "아직 slm_runs.jsonl 기록 없음")
'''
    nb = {
        "cells": [
            _cell_md(
                "# SLM 학습 & 튜닝 (emotion-bot)\\n\\n"
                "대화/감정 분석용 SLM fine-tuning 템플릿입니다.\\n"
                "- 로그: `experiments/slm_runs.jsonl`\\n"
                "- 설정: `configs/slm/example.yaml`\\n"
                "- `RUN_TRAINING=True` 후 학습 셀을 프로젝트에 맞게 완성하세요."
            ),
            _cell_code(setup),
            _cell_md("## 1. 파라미터"),
            _cell_code(cfg_cell),
            _cell_md("## 2. 학습 (템플릿)"),
            _cell_code(train_placeholder),
            _cell_md("## 3. run 비교"),
            _cell_code(compare),
        ],
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _save(nb, path)
    return path


if __name__ == "__main__":
    paths = [write_yolo(), write_mtl(), write_slm()]
    for p in paths:
        print(p, p.stat().st_size)
