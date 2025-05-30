import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
import numpy as np
import random
from PIL import Image
import time

# ----- 設定隨機種子 -----
def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

# ----- 定義 Dataset -----
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.dataset = CocoDetection(img_dir, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, targets = self.dataset[idx]
        img = F.to_tensor(img)

        boxes = []
        labels = []
        for obj in targets:
            boxes.append(obj["bbox"])
            labels.append(obj["category_id"])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.dataset)

# ----- 凍結 & 解凍 Backbone -----
def freeze_backbone(model):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    print("✅ Backbone frozen.")

def unfreeze_backbone(model):
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True
    print("🔓 Backbone unfrozen.")

# ----- 驗證損失計算 -----
@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    # 暂时切换到 train 模式以计算 loss
    prev_mode = model.training
    model.train()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targs)
            total_loss += sum(loss_dict.values()).item()
            count += 1
    # 恢复模式
    if not prev_mode:
        model.eval()
    val_loss = total_loss / count if count > 0 else 0.0
    return val_loss


# ----- 訓練主要流程 -----
def main():
    # --- 參數設定 ---
    root_dir = 'split_dataset/train/images'
    ann_file = 'split_dataset/train/annotations_pretty.json'
    val_root = 'split_dataset/val/images'
    val_ann = 'split_dataset/val/annotations.json'
    num_classes = 6  # 含背景
    batch_size = 4
    num_epochs = 30
    unfreeze_epoch = 10
    patience = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 載入資料 ---
    train_dataset = CocoDataset(root_dir, ann_file)
    val_dataset = CocoDataset(val_root, val_ann)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # --- 建立模型 ---
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # --- 凍結 backbone ---
    freeze_backbone(model)

    # --- Optimizer（初期只訓練分類頭）---
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- 開始訓練 ---
    best_val_loss = float('inf')
    wait = 0
    for epoch in range(1, num_epochs + 1):
        if epoch == unfreeze_epoch:
            unfreeze_backbone(model)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=0.002, momentum=0.9, weight_decay=0.001
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        model.train()
        total_train_loss = 0.0
        for imgs, tgts in train_loader:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        val_loss = evaluate_loss(model, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.cuda.empty_cache()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⛔ Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    print("✅ 訓練完成")

if __name__ == "__main__":
    main()
