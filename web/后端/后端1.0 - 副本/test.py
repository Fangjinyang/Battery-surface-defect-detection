from ultralytics import YOLO
import cv2
import os

# 创建outputs文件夹（如果不存在）
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载YOLOv8模型
yolo = YOLO("yolov8n.pt")

# 对图像进行推理
result = yolo("ultralytics/assets/bus.jpg")

# 打印原始检测结果
print("Detection Results:", result)

# 提取检测框信息
# result[0] 是第一张图片的检测结果
boxes = result[0].boxes  # 获取检测到的边界框
labels = result[0].names  # 获取类别名称
img = result[0].orig_img  # 获取原图

# 如果你想打印每个检测到的物体的信息
for box in boxes:
    # 获取边界框的坐标（x1, y1, x2, y2）
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    # 获取目标类别的 ID 和名称
    class_id = int(box.cls)
    class_name = labels[class_id]  # 根据类 ID 获取类别名称
    
    # 获取目标的置信度并转换为float
    confidence = box.conf.item()  # 使用 .item() 转换为普通的浮点数
    
    # 打印每个检测到的目标的信息
    print(f"Detected {class_name} with confidence {confidence:.2f}")
    print(f"Bounding box: [{x1}, {y1}, {x2}, {y2}]")

    # 可选：绘制边界框并标注
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, f"{class_name} {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 保存带检测框的图像到outputs文件夹
output_path = os.path.join(OUTPUT_FOLDER, 'output_with_detections.jpg')
cv2.imwrite(output_path, img)
print(f"Saved image with detections to {output_path}")
