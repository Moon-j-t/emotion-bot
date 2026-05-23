# YOLO 얼굴 데이터 연동

## Human Face dataset (Roboflow)

**로컬 원본** (Git 제외, 그대로 유지):

`C:\Users\moonjintae\datasets\Human Face dataset`

```
Human Face dataset/
├── data.yaml          # Roboflow 원본 (참고용)
├── train/images, train/labels
└── valid/images, valid/labels
```

**프로젝트 학습 설정** (`data/face_detect/data.yaml`):

- `link_human_face_dataset.py`가 위 경로를 가리키도록 생성·갱신
- 이미지 복사 없음 → 디스크 절약

## 연동 명령

```powershell
cd "C:\Users\moonjintae\projects\emotion bot(cursor ver)"
python scripts/link_human_face_dataset.py
```

## Gitignore

| 경로 | Git |
|------|-----|
| `data/face_detect/images/`, `labels/` | ignore (프로젝트 내 복사본용) |
| `data/face_detect/data.yaml` | 커밋 (경로만 기록) |
| 원본 `C:\Users\moonjintae\datasets\...` | 프로젝트 밖 → Git 무관 |
