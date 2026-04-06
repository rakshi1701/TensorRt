import cv2
from ultralytics import YOLO
from tensorrt_pipeline import tensorrt_model
cap = cv2.VideoCapture('rtsp://admin:abcd@1234@192.168.1.2')

# model = YOLO('yolov8n.engine')
model = YOLO('yolov8s.pt')

if model.model_path.endswith(".engine"):
    print("TensorRT model loaded")
else:
    print("Not a TensorRT model")
    model = tensorrt_model(model)

while 1:
    _, frame = cap.read()

    results = model(frame)
    result = results[0].plot()

    cv2.imshow('detection', result)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break

cap.release()
