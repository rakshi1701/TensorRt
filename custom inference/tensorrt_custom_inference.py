import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Auto-initializes CUDA context
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
 
 
# ─────────────────────────────────────────────
# COCO class names (replace with your own if custom-trained)
# ─────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
 
 
# ─────────────────────────────────────────────
# Detection result dataclass
# ─────────────────────────────────────────────
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
 
    def __repr__(self):
        return (
            f"Detection(class='{self.class_name}' (id={self.class_id}), "
            f"conf={self.confidence:.2f}, "
            f"xyxy=[{self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}])"
        )
 
 
# ─────────────────────────────────────────────
# TensorRT Engine Loader
# ─────────────────────────────────────────────
class TRTEngine:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
 
    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
 
    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
 
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
 
            # Handle dynamic batch dimension
            shape = tuple(abs(s) for s in shape)
            size = int(np.prod(shape))
 
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
 
            tensor_info = {
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype,
            }
 
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(tensor_info)
            else:
                outputs.append(tensor_info)
 
        return inputs, outputs, bindings, stream
 
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], input_data.ravel())
 
        # Host → Device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
 
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
 
        # Device → Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
 
        self.stream.synchronize()
 
        return [out["host"].reshape(out["shape"]) for out in self.outputs]
 
 
# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess(image: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize → normalize → NCHW format
    Returns: preprocessed tensor, scale, (pad_w, pad_h)
    """
    h, w = image.shape[:2]
    target_w, target_h = input_size
 
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
 
    resized = cv2.resize(image, (new_w, new_h))
 
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
 
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
 
    # BGR → RGB, normalize to [0, 1], NCHW
    img = padded[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # CHW → NCHW
    img = np.ascontiguousarray(img)
 
    return img, scale, (pad_w, pad_h)
 
 
# ─────────────────────────────────────────────
# Postprocessing
# ─────────────────────────────────────────────
def postprocess(
    output: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: List[str] = COCO_CLASSES,
) -> List[Detection]:
    """
    Decode YOLOv8 output → list of Detection objects.
    YOLOv8 output shape: [1, 4 + num_classes, num_anchors]
    """
    pad_w, pad_h = pad
    predictions = output[0]  # Remove batch dim → [4+nc, num_anchors]
 
    # Transpose to [num_anchors, 4+nc]
    predictions = predictions.T
 
    boxes = predictions[:, :4]       # cx, cy, w, h
    scores = predictions[:, 4:]      # class scores
 
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
 
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
 
    if len(boxes) == 0:
        return []
 
    # cx, cy, w, h → x1, y1, x2, y2 (in padded/scaled space)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
 
    # Remove letterbox padding and rescale to original image coords
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale
 
    # NMS using OpenCV
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(
        nms_boxes, confidences.tolist(), conf_threshold, iou_threshold
    )
 
    detections = []
    for idx in indices:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        cid = int(class_ids[i])
        detections.append(Detection(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            confidence=float(confidences[i]),
            class_id=cid,
            class_name=class_names[cid] if cid < len(class_names) else f"class_{cid}",
        ))
 
    return detections
 
 
# ─────────────────────────────────────────────
# Draw detections on frame
# ─────────────────────────────────────────────
def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        label = f"{det.class_name} {det.confidence:.2f}"
 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return image
 
 
# ─────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────
def run_inference(
    engine_path: str,
    source,                          # image path (str) or frame (np.ndarray)
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: Tuple[int, int] = (640, 640),
    class_names: List[str] = COCO_CLASSES,
    save_output: bool = False,
    output_path: str = "output.jpg",
) -> List[Detection]:
 
    # Load engine once (you can reuse this across frames)
    engine = TRTEngine(engine_path)
 
    # Load image
    if isinstance(source, str):
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Image not found: {source}")
    else:
        image = source  # already a numpy frame
 
    # Preprocess
    tensor, scale, pad = preprocess(image, input_size)
 
    # Infer
    outputs = engine.infer(tensor)
 
    # Postprocess → detections
    detections = postprocess(
        outputs[0], scale, pad,
        conf_threshold, iou_threshold, class_names
    )
 
    # Print results
    print(f"\n{'─'*50}")
    print(f"Detections found: {len(detections)}")
    print(f"{'─'*50}")
    for det in detections:
        print(f"  Class     : {det.class_name} (id={det.class_id})")
        print(f"  Confidence: {det.confidence:.4f}")
        print(f"  BBox xyxy : [{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}]")
        print()
 
    # Optionally save annotated output
    if save_output:
        annotated = draw_detections(image.copy(), detections)
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image → {output_path}")
 
    return detections
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ENGINE_PATH = "model.engine"       # ← your .engine file
    IMAGE_PATH  = "test.jpg"           # ← your test image
 
    # Optional: replace with your custom class list
    # CLASS_NAMES = ["cat", "dog", "car"]
 
    detections = run_inference(
        engine_path=ENGINE_PATH,
        source=IMAGE_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45,
        input_size=(640, 640),
        class_names=COCO_CLASSES,     # ← swap with CLASS_NAMES if custom
        save_output=True,
        output_path="result.jpg",
    )
 
    # Access detection fields directly
    for det in detections:
        print(det.x1, det.y1, det.x2, det.y2)
        print(det.class_id, det.class_name, det.confidence)