# 학습용 Notebooks (emotion-bot)

| 모델 | ipynb 경로 | 실험 로그 |
|------|------------|-----------|
| **YOLO** (얼굴 검출) | [yolo/yolo_train_and_tune.ipynb](yolo/yolo_train_and_tune.ipynb) | `experiments/yolo_runs.jsonl` |
| **CNN / MTL** (표정·나이) | [mtl/mtl_train_and_tune.ipynb](mtl/mtl_train_and_tune.ipynb) | `experiments/mtl_runs.jsonl` |
| **SLM** | [slm/slm_train_and_tune.ipynb](slm/slm_train_and_tune.ipynb) | `experiments/slm_runs.jsonl` |

## 실행

```powershell
cd "C:\Users\moonjintae\projects\emotion bot(cursor ver)"
pip install -r requirements-notebook.txt
jupyter notebook notebooks/yolo/yolo_train_and_tune.ipynb
```

SLM 학습 시 추가: `pip install transformers datasets accelerate`

## 재생성

ipynb가 없을 때:

```powershell
python scripts/write_training_notebooks.py
```
