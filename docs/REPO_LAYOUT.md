# emotion-bot 저장소 구조

GitHub: https://github.com/Moon-j-t/emotion-bot  
로컬: `C:\Users\moonjintae\projects\emotion bot(cursor ver)`

## 한 repo 안에서 역할 분리

```
emotion-bot/
├── notebooks/          # 모델 학습·튜닝 (ipynb) ← 이번에 추가하는 영역
│   ├── yolo/
│   ├── mtl/            # CNN/MTL (예정)
│   └── slm/            # SLM (예정)
├── train/              # 재현 가능한 학습 스크립트
├── models/             # 추론용 모델 코드
├── inference/          # 봇/데모 실행
├── experiments/        # jsonl 실험 로그 (Git)
├── weights/            # .pt (Git 제외)
└── ...
```

## `face_attr_mtl` 폴더는?

`C:\Users\moonjintae\projects\face_attr_mtl` 는 동일 코드의 **사본·실험용**으로 보입니다.  
**GitHub에는 emotion-bot 하나만** 두고, Cursor도 `emotion bot(cursor ver)` 를 여는 것을 권장합니다.

노트북이 `face_attr_mtl`에만 있다면:

```powershell
Copy-Item -LiteralPath "C:\Users\moonjintae\projects\face_attr_mtl\notebooks\yolo_train_and_tune.ipynb" `
  -Destination "C:\Users\moonjintae\projects\emotion bot(cursor ver)\notebooks\yolo\yolo_train_and_tune.ipynb"
```

## 산출물 흐름

1. `notebooks/` 에서 파라미터 탐색 → `experiments/*.jsonl`
2. `train/*.py` 로 본 학습 → `weights/*.pt`
3. `inference/` · 봇 코드에서 `weights/` 로드
