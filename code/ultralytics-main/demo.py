from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8-SwinTransformer.yaml')
    model.train(data='yolo-bvn.yaml', epochs=50, imgsz=640, batch=16)

