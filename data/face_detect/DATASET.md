# YOLO 얼굴 데이터 연동

## 원본 위치

`C:\Users\moonjintae\datasets\Human Face dataset`

```
Human Face dataset/
├── data.yaml
├── train/images    train/labels
└── valid/images    valid/labels
```

## 연동 (이미지 복사 없음)

```powershell
cd "C:\Users\moonjintae\projects\emotion bot(cursor ver)"
python scripts/link_human_face_dataset.py
```

→ `data/face_detect/data.yaml` 과 원본 `data.yaml` 이 위 경로를 가리킵니다.

## Gitignore

| 경로 | Git |
|------|-----|
| `data/face_detect/images/`, `labels/` | ignore |
| `data/face_detect/data.yaml` | 커밋 (경로만) |
| 원본 데이터셋 폴더 | 프로젝트 밖 |
