# YOLO 얼굴 검출 데이터셋

## 디렉터리 구조 (학습 준비 후)

```
face_detect/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/    # 이미지와 동일 파일명, .txt
    └── val/
```

## 라벨 형식 (YOLO)

각 `.txt` 파일은 한 줄당 한 얼굴:

```
0 x_center y_center width height
```

좌표는 **0~1 정규화**(이미지 너비·높이 기준). 클래스 ID는 항상 `0` (face).

## 준비 방법

CSV(픽셀 bbox)에서 변환:

```bash
python scripts/prepare_yolo_dataset.py --csv annotations.csv --images-dir raw_images --val-ratio 0.2
```

이미 YOLO 형식 폴더가 있으면 train/val 분할만:

```bash
python scripts/prepare_yolo_dataset.py --source-yolo path/to/yolo_dataset --val-ratio 0.2
```

CSV 컬럼 예 (`annotations.csv`):

```csv
image_path,xmin,ymin,xmax,ymax
img001.jpg,120,80,220,200
```
