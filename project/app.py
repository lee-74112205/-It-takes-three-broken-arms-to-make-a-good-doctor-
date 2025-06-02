import io
import os
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, send_file, abort
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from ultralytics import YOLO

app = Flask(__name__)

# ----------------------------------------------------------------------------
# 0. 定義「索引 → 英文代號」的對照列表（共 6 類）
# ----------------------------------------------------------------------------
#   ID  英文代號
#    0  Normal
#    1  SCF_type1
#    2  SCF_type2
#    3  SCF_type3
#    4  LCF
#    5  MCF
CLASS_NAMES = [
    "Normal",
    "SCF_type1",
    "SCF_type2",
    "SCF_type3",
    "LCF",
    "MCF"
]

# ----------------------------------------------------------------------------
# 1. 指定「標注 JSON」的資料夾
# ----------------------------------------------------------------------------
ANNOT_FOLDER = "標示"  # 這個資料夾裡放你 LabelMe 匯出的 JSON

# ----------------------------------------------------------------------------
# 2. 全域載入模型 (啟動時只做一次)
# ----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用的 device：{device}")

# 2.1 載入 Faster R-CNN（6 類，保留所有候選框）
try:
    faster_model = fasterrcnn_resnet50_fpn(
        weights=False,
        num_classes=len(CLASS_NAMES),  # =6
        box_score_thresh=0.0,          # 讓我們自行決定要畫哪個框
        box_nms_thresh=0.5
    )
    checkpoint = torch.load('models/best_mode_normal.pth', map_location=device)
    faster_model.load_state_dict(checkpoint)
    faster_model.to(device)
    faster_model.eval()
    print("[INFO] Faster R-CNN 權重載入成功")
except Exception as e:
    print(f"[ERROR] 載入 Faster R-CNN 權重失敗：{e}")
    raise

# 2.2 載入 YOLOv8（自行訓練的 6 類模型）
try:
    yolo_model = YOLO('models/yolov28.pt')
    print("[INFO] YOLOv8 權重載入成功")
except Exception as e:
    print(f"[ERROR] 載入 YOLOv8 權重失敗：{e}")
    raise

# 只做 ToTensor，不做 Normalize
to_tensor = ToTensor()


# ----------------------------------------------------------------------------
# 3. 輔助函式：畫出真實標注框 (GT)，顯示純英文代號
# ----------------------------------------------------------------------------
def draw_ground_truth(image_pil: Image.Image, json_path: str, scale: float):
    """
    從 LabelMe JSON 載入所有 rectangle 標註，依照 scale 縮放後，
    把每個方框畫成綠色，並以 CLASS_NAMES 的英文代號標註。
    """
    if not os.path.isfile(json_path):
        return image_pil  # 若找不到對應 JSON，就直接回傳原圖
    
    # 讀取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 讀取 JSON 標注失敗: {e}")
        return image_pil

    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape.get("points", [])
        if len(pts) != 2:
            continue

        label_str = shape.get("label", "")
        # 把 label_str 轉成 int，再從 CLASS_NAMES 拿英文代號
        try:
            label_idx = int(label_str)
            if 0 <= label_idx < len(CLASS_NAMES):
                label_name = CLASS_NAMES[label_idx]
            else:
                label_name = label_str
        except:
            label_name = label_str

        # 取得原始座標，並乘以 scale
        (x1, y1), (x2, y2) = pts
        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)

        # 畫綠色實線矩形框
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)

        # 在框左上角畫綠底文字區，並顯示英文代號
        text = f"GT:{label_name}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill='green')
        draw.text((x1, y1 - th), text, fill='white', font=font)

    return image_pil


# ----------------------------------------------------------------------------
# 4. 輔助函式：畫出 Faster R-CNN 分數最高的預測框 (藍色)
# ----------------------------------------------------------------------------
def draw_faster_best_box(image_pil: Image.Image, outputs):
    """
    將 Faster R-CNN 的分數最高框畫到 image_pil 上，顯示 Pred:<英文代號>:<置信度>。
    """
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    boxes = outputs[0]['boxes'].detach().cpu().numpy()   # [N,4]
    scores = outputs[0]['scores'].detach().cpu().numpy() # [N]
    labels = outputs[0]['labels'].detach().cpu().numpy() # [N]

    if len(boxes) == 0:
        print("[DEBUG][Faster] 沒有任何預測框")
        return image_pil

    max_idx = int(np.argmax(scores))
    max_score = float(scores[max_idx])
    max_box = boxes[max_idx]
    max_label_idx = int(labels[max_idx])

    if 0 <= max_label_idx < len(CLASS_NAMES):
        label_name = CLASS_NAMES[max_label_idx]
    else:
        label_name = str(max_label_idx)

    print(f"[DEBUG][Faster] 最佳預測 idx={max_idx}, label='{label_name}', score={max_score:.4f}")

    x1, y1, x2, y2 = map(int, max_box)
    draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)

    text = f"Pred:{label_name}:{max_score:.2f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - th, x1 + tw, y1], fill='blue')
    draw.text((x1, y1 - th), text, fill='white', font=font)

    return image_pil


# ----------------------------------------------------------------------------
# 5. 輔助函式：畫出 YOLOv8 分數最高的預測框 (紅色)
# ----------------------------------------------------------------------------
def draw_yolo_best_box(image_pil: Image.Image, results):
    """
    將 YOLOv8 的分數最高框畫到 image_pil 上，顯示 Pred:<英文代號>:<置信度>。
    """
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    boxes = results[0].boxes.xyxy.cpu().numpy()   # [N,4]
    scores = results[0].boxes.conf.cpu().numpy()  # [N]
    classes = results[0].boxes.cls.cpu().numpy()  # [N]

    if len(boxes) == 0:
        print("[DEBUG][YOLO] 沒有任何預測框")
        return image_pil

    max_idx = int(np.argmax(scores))
    max_score = float(scores[max_idx])
    max_box = boxes[max_idx]
    max_cls_idx = int(classes[max_idx])

    if 0 <= max_cls_idx < len(CLASS_NAMES):
        label_name = CLASS_NAMES[max_cls_idx]
    else:
        label_name = str(max_cls_idx)

    print(f"[DEBUG][YOLO] 最佳預測 idx={max_idx}, label='{label_name}', score={max_score:.4f}")

    x1, y1, x2, y2 = map(int, max_box)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    text = f"Pred:{label_name}:{max_score:.2f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - th, x1 + tw, y1], fill='red')
    draw.text((x1, y1 - th), text, fill='white', font=font)

    return image_pil


# ----------------------------------------------------------------------------
# 6. /predict 路由：接收前端送來的影像檔，並畫出 GT (綠色英文字) + Pred (藍/紅)
# ----------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return abort(400, '沒有找到上傳的檔案')
        file = request.files['file']
        if file.filename == '':
            return abort(400, '沒有選擇檔案')
        chosen_model = request.form.get('model', 'faster')
        print(f"[INFO] 使用者選擇模型：{chosen_model}")

        # 1) 把上傳的二進位讀成 OpenCV BGR
        img_stream = file.stream.read()
        np_img = np.frombuffer(img_stream, np.uint8)
        cv2_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)  # BGR
        if cv2_img is None:
            print("[ERROR] cv2.imdecode 無法解析上傳影像")
            return abort(400, '影像解碼失敗')

        # 2) 先記住原始尺寸
        orig_h, orig_w = cv2_img.shape[:2]

        # 3) 若影像過大，就先 resize (最大邊 800)，並計算縮放比例
        max_side = max(orig_h, orig_w)
        if max_side > 800:
            scale = 800 / max_side
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            cv2_img = cv2.resize(cv2_img, (new_w, new_h))
            print(f"[DEBUG] 影像被縮放到 {(new_w, new_h)}")
        else:
            scale = 1.0  # 沒有縮放就用 1.0

        # 4) 轉成灰階，複製三通道，然後轉成 PIL Image
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])
        pil_img = Image.fromarray(gray_3ch)

        # 5) 在縮放後的 PIL Image 上先畫「真實標注框」(綠色英文字)
        base_img = pil_img.copy()
        json_name = os.path.splitext(file.filename)[0] + ".json"
        json_path = os.path.join(ANNOT_FOLDER, json_name)
        base_img = draw_ground_truth(base_img, json_path, scale)

        # 6) 根據使用者選擇的模型推論，並把「預測框」畫在同一張圖上
        if chosen_model == 'faster':
            img_tensor = to_tensor(pil_img).to(device)
            with torch.no_grad():
                outputs = faster_model([img_tensor])
            result_img = draw_faster_best_box(base_img, outputs)
        else:
            results = yolo_model(pil_img, conf=0.10)
            result_img = draw_yolo_best_box(base_img, results)

        # 7) 將結果存到 buffer，回傳給前端
        buf = io.BytesIO()
        result_img.save(buf, format='JPEG')
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')

    except Exception as e:
        print(f"[ERROR][predict] 發生例外：{e}")
        import traceback; traceback.print_exc()
        return abort(500, '伺服器內部錯誤')


# ----------------------------------------------------------------------------
# 8. 啟動 Flask
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
