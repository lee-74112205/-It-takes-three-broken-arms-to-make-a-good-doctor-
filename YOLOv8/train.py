from sympy import monic
from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n.pt")
    model.train(
        data="coco_custom.yaml",
        epochs=300,
        batch=4,
        patience=50,
        name="bone_fracture_yolo_v2"
    )


if __name__ == "__main__":
    train_model()
