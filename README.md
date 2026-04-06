# YOLOv8 TensorRT Demo

Lightweight demo for running YOLOv8 object detection with optional TensorRT engine acceleration.

Project layout
- `detection_tensorrt.py` — main demo script that reads an RTSP stream and displays detections.
- `tensorrt_pipeline.py` / `tensorrt_pipe_line.py` / `tensorrt_pipeline.ipynb` — helper code and notebook for building/using TensorRT models.
- `yolov8n.pt`, `yolov8n.onnx`, `yolov8n.engine` — example model files (PyTorch, ONNX, and TensorRT engine).
- `data/` — sample images.

Requirements
- Linux with NVIDIA GPU and appropriate drivers.
- Python 3.8+ (project includes a virtualenv at `tensorrt/` in this workspace).
- Recommended Python packages: ultralytics, opencv-python, torch (matching your CUDA), onnx, tensorrt (if using engine).

Quick setup
1. Activate the provided virtual environment:

   source tensorrt/bin/activate

2. Install or verify Python packages (adjust CUDA-specific torch/tensorrt packages as needed):

   pip install ultralytics opencv-python torch torchvision onnx

3. If you plan to use TensorRT engine (`*.engine`), install/configure TensorRT and the matching Python bindings for your platform.

Running the demo
- Edit `detection_tensorrt.py` to point `cv2.VideoCapture(...)` to your RTSP stream or change to a local video/file.
- The script will attempt to load the Ultralytics model path. If it ends with `.engine`, it will treat it as a TensorRT engine; otherwise it will load a PyTorch `.pt` model and call the pipeline helper to convert/use TensorRT when available.

Notes and troubleshooting
- If using TensorRT, make sure the engine was built for your GPU/driver/CUDA version. Incompatibilities cause load/runtime failures.
- For ONNX conversion and engine building, consult the files in the repo and the `tensorrt_pipeline.ipynb` notebook.
- If OpenCV cannot open the RTSP stream, test with a local video file or `cv2.VideoCapture(0)` for webcam.

License
- This repository contains demo code — adapt and use at your own risk. No warranty provided.
