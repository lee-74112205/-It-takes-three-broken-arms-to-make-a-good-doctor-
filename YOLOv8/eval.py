import os
from ultralytics import YOLO

def run_validation():
    model = YOLO("runs/detect/bone_fracture_yolo_v26/weights/best.pt")
    metrics = model.val(
        data="coco_custom.yaml",
        imgsz=640,
        batch=16,
        name="bf_v24_val2"
    )
    return metrics

if __name__ == "__main__":
    metrics = run_validation()

    for class_name, precision in zip(metrics.names.values(), metrics.box.p):
        print(f"{class_name:<12} Precision: {precision:.3f}")

    # 直接访问属性，不要加 ()
    print(f"\nMean Precision (mp): {metrics.box.mp:.3f}")
