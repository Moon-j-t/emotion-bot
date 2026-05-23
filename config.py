"""프로젝트 전역 설정."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data" / "face_detect"

# Roboflow Human Face (로컬). scripts/link_human_face_dataset.py 로 data.yaml 연동
HUMAN_FACE_DATASET = Path(r"C:\Users\moonjintae\datasets\Human Face dataset")

# --- YOLO 얼굴 검출 (직접 학습) ---
# 추론 시 사용: train/train_yolo_face.py 학습 후 생성되는 가중치
YOLO_MODEL = WEIGHTS_DIR / "face_yolo.pt"

# 학습 시 아키텍처 초기화용 베이스(일반 COCO 사전학습). 얼굴 탐지 능력은 아래 데이터로 학습.
YOLO_BASE_ARCH = "yolov8n.pt"
YOLO_DATA_YAML = DATA_DIR / "data.yaml"
YOLO_CLASS_NAME = "face"

YOLO_CONF = 0.45
YOLO_IOU = 0.5
YOLO_IMGSZ = 640

# YOLO 학습 하이퍼파라미터
YOLO_TRAIN_EPOCHS = 100
YOLO_TRAIN_BATCH = 16
YOLO_TRAIN_PATIENCE = 20
YOLO_TRAIN_PROJECT = PROJECT_ROOT / "runs" / "detect"
YOLO_TRAIN_NAME = "face_train"

# --- MobileNetV3 MTL ---
MTL_WEIGHTS = WEIGHTS_DIR / "face_mtl_mobilenetv3.pt"
MOBILENET_VARIANT = "large"  # "large" | "small"
INPUT_SIZE = 224
NUM_EXPRESSIONS = 7
AGE_MIN = 1.0
AGE_MAX = 100.0

EXPRESSION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# Temporal smoothing
SMOOTH_WINDOW = 8
TRACK_IOU_THRESHOLD = 0.35
TRACK_MAX_MISSED = 10

# 디바이스: "cuda" | "cpu" | None(자동)
DEVICE = None
