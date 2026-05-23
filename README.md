# emotion-bot

GitHub: https://github.com/Moon-j-t/emotion-bot  
로컬: `C:\Users\moonjintae\projects\emotion bot(cursor ver)`

표정·나이 등 감정 관련 **봇/추론 파이프라인**과, YOLO·CNN(MTL)·SLM 등 **모델 학습(notebook)** 을 **이 저장소 하나**에서 관리합니다.

**직접 학습한 YOLO**로 얼굴을 검출하고, **MobileNetV3 MTL**로 표정·나이를 추정합니다.  
연속 프레임에서는 Temporal Smoothing으로 예측을 안정화합니다.

> `C:\Users\moonjintae\projects\face_attr_mtl` 은 동일 작업의 복사본일 수 있습니다. Git push는 **이 폴더(emotion-bot)** 기준으로 하세요. 구조 설명: [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md)

## 파이프라인 개요

```
[1] YOLO 학습 (train/train_yolo_face.py)
         ↓ weights/face_yolo.pt
[2] MTL 학습 (train/train_mtl.py) — 얼굴 crop 기준
         ↓ weights/face_mtl_mobilenetv3.pt
[3] 추론 (inference/run_webcam.py)
```

| 단계 | 설명 |
|------|------|
| YOLO | 사용자 데이터로 **얼굴 bbox 검출기** 직접 학습 |
| MTL | 검출된 얼굴 crop → 표정 + 나이 (공유 백본 1회 forward) |
| Smoothing | track별 N프레임 평균 |

## 설치

```bash
cd "C:\Users\moonjintae\projects\emotion bot(cursor ver)"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Notebook (YOLO·MTL·SLM 학습·튜닝)

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/yolo/yolo_train_and_tune.ipynb
```

| 역할 | 위치 |
|------|------|
| 학습·파라미터 비교 | `notebooks/` |
| 실험 이력 (Git) | `experiments/*.jsonl` |
| 채택 설정 요약 | 아래 **실험 결과 요약** |

| Notebook | 경로 |
|----------|------|
| YOLO | `notebooks/yolo/yolo_train_and_tune.ipynb` |
| CNN/MTL | `notebooks/mtl/mtl_train_and_tune.ipynb` |
| SLM | `notebooks/slm/slm_train_and_tune.ipynb` |

## 실험 결과 요약

<!-- notebooks 마지막 셀 출력을 채택 run 확정 후 여기에 붙여 넣기 -->

| 항목 | 값 |
|------|-----|
| run_id | (미정) |
| epochs | - |
| batch | - |
| mAP50 | - |

전체 이력: [experiments/yolo_runs.jsonl](experiments/yolo_runs.jsonl)

## 1. YOLO 얼굴 검출 — 데이터 준비

### Human Face dataset (연동됨)

로컬 Roboflow 데이터:

`C:\Users\moonjintae\datasets\Human Face dataset`

```powershell
python scripts/link_human_face_dataset.py
```

→ `data/face_detect/data.yaml`이 위 경로를 가리킵니다 (이미지 복사 없음).  
자세히: [data/face_detect/DATASET.md](data/face_detect/DATASET.md)

### 방법 A: CSV (픽셀 bbox)

`data/face_detect/annotations.example.csv` 참고:

```csv
image_path,xmin,ymin,xmax,ymax
img001.jpg,120,80,220,200
```

```bash
python scripts/prepare_yolo_dataset.py ^
  --csv path\to\annotations.csv ^
  --images-dir path\to\raw_images ^
  --val-ratio 0.2
```

### 방법 B: 이미 YOLO 형식

```
my_dataset/
  images/
  labels/   # class x_center y_center width height (0~1)
```

```bash
python scripts/prepare_yolo_dataset.py --source-yolo path\to\my_dataset --val-ratio 0.2
```

출력: `data/face_detect/` (`images/train`, `labels/train`, `data.yaml`)

## 2. YOLO 얼굴 검출 — 학습

**Notebook (탐색):** `notebooks/yolo/yolo_train_and_tune.ipynb`  
**Script (본 학습):**

```bash
python train/train_yolo_face.py
```

- 베이스 아키텍처: `config.YOLO_BASE_ARCH` (`yolov8n.pt`, COCO 초기화만 사용)
- **얼굴 탐지 성능은 사용자 라벨 데이터로 학습**
- 완료 시 `weights/face_yolo.pt` 자동 저장

주요 옵션:

```bash
python train/train_yolo_face.py --epochs 150 --batch 8 --imgsz 640 --resume
```

## 3. MTL (표정·나이) 학습

얼굴이 잘린 이미지 + CSV:

```csv
image_path,expression_id,age
faces/001.jpg,4,32.0
```

```bash
python train/train_mtl.py --train-csv data/train.csv --val-csv data/val.csv --data-root data/
```

## 4. 추론

**YOLO 학습이 먼저 필요합니다.** `weights/face_yolo.pt` 없으면 명확한 오류 메시지가 표시됩니다.

```bash
python inference/run_webcam.py
python inference/run_image.py path\to\face.jpg -o out.jpg
```

```bash
python inference/run_webcam.py --yolo weights\face_yolo.pt --mtl weights\face_mtl_mobilenetv3.pt
```

## 프로젝트 구조

```
emotion-bot/
├── notebooks/                 # YOLO / MTL(CNN) / SLM 학습 ipynb
│   └── yolo/
├── experiments/               # yolo_runs.jsonl 등 (Git)
├── configs/yolo/
├── train/                     # 재현용 학습 스크립트
├── models/ · inference/       # 봇·추론
├── data/face_detect/
└── weights/                   # .pt (Git 제외)
```

## 설정 (`config.py`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `YOLO_MODEL` | `weights/face_yolo.pt` | 추론용 (학습 후 생성) |
| `YOLO_BASE_ARCH` | `yolov8n.pt` | 학습 시작 체크포인트 |
| `YOLO_DATA_YAML` | `data/face_detect/data.yaml` | 학습 데이터 |
| `YOLO_TRAIN_EPOCHS` | 100 | 학습 epoch |

## 참고

- 사전 배포된 `yolov8n-face.pt` 등 **외부 얼굴 모델 다운로드는 사용하지 않습니다.**
- MTL 가중치 없이도 YOLO만 학습하면 검출·박스 시각화는 가능합니다 (표정/나이 헤드는 랜덤 초기화).
