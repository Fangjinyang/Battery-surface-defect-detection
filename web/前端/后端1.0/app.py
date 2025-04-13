from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载 YOLOv8 目标检测模型
yolo_model = YOLO("yolov8-LSK.yaml")
yolo_model.train(data='yolo-bvn.yaml', epochs=50, imgsz=640, batch=32)

@app.route('/api/submit-images', methods=['POST'])
def upload_and_detect():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('images')

    if len(files) == 0:
        return jsonify({'error': 'No files uploaded'}), 400
    
    uploaded_files = []
    detection_results = []

    for file in files:
        try:
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # 获取文件名并确保安全
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Saving file to {filepath}")

            file.save(filepath)
            # 执行目标检测
            results = yolo_model(filepath)

            # 解析检测结果
            detected_objects = []
            for result in results:
                boxes = result.boxes  # 获取所有检测到的目标
                labels = result.names  # 获取类别名称
                img = result.orig_img  # 获取原始图像

                for box in boxes:
                    # 获取边界框的坐标（x1, y1, x2, y2）
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 获取目标类别的 ID 和名称
                    class_id = int(box.cls)
                    class_name = labels[class_id]

                    # 获取目标的置信度并转换为 float
                    confidence = box.conf.item()

                    # 打印每个检测到的目标的信息
                    detected_objects.append({
                        'class': class_name,  # 目标类别
                        'confidence': confidence,  # 置信度
                        'bbox': [float(coord) for coord in box.xyxy[0]]  # 边界框 [x1, y1, x2, y2]
                    })

                    # 可选：绘制边界框并标注
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_name} {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 保存带检测框的图片到outputs文件夹
                output_filepath = os.path.join(OUTPUT_FOLDER, filename)
                cv2.imwrite(output_filepath, img)  # 保存图像

            uploaded_files.append(filename)
            detection_results.append({
                'filename': filename,
                'detections': detected_objects,
                'output_url': f'/api/get-image/{filename}'  # 返回带检测框的图像的URL
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
    return jsonify({'message': 'Files processed successfully', 'results': detection_results}), 200

OUTPUT_FOLDER = 'outputs/'  # 假设输出文件夹为 outputs/

@app.route('/api/download-image/<filename>', methods=['GET'])
def download_image(filename):
    # 确保文件存在，并使用 send_from_directory 安全提供文件
    try:
        # 安全地从指定文件夹返回文件
        return send_from_directory(OUTPUT_FOLDER, filename)
    except FileNotFoundError:
        # 如果文件不存在，返回 404 错误
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)

