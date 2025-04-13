from ultralytics import YOLO
import torch
# 初始化YOLOv8
model = YOLO('yolov8s.yaml')

# 加载预训练骨干
model.model.backbone.load_state_dict(
    torch.load('yolov8s_backbone_simclr.pth'))

# 微调检测任务
model.train(data='defect.yaml', epochs=100)