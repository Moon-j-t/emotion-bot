# MTL 학습 데이터 (표정 + 나이)

## 생성

```powershell
python scripts/prepare_mtl_from_roboflow_emotion.py ^
  --source "C:\Users\moonjintae\Downloads\emotion detection.v1i.yolov8"
```

## 원본 클래스 → expression_id

| Roboflow id | name | expression_id | config 라벨 |
|-------------|------|---------------|-------------|
| 0 | anger | 0 | angry |
| 1 | content | 6 | neutral |
| 2 | disgust | 1 | disgust |
| 3 | fear | 2 | fear |
| 4 | happy | 3 | happy |
| 5 | neutral | 6 | neutral |
| 6 | sad | 4 | sad |
| 7 | surprise | 5 | surprise |

## CSV 형식

`image_path,expression_id,age` — `data_root`는 `mtl/`

나이: InsightFace `buffalo_l` (genderage 모듈) 추정값.
