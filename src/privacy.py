"""Anonymize dash-cam frames before anything is shown, saved or exported.

Uses a COCO-pretrained YOLO detector to find people and vehicles and blurs them,
so faces and number plates never reach the annotated video or evidence snapshots.
"""
import cv2

PRIVACY_MODEL = "yolo11n.pt"  # COCO-pretrained; auto-downloaded by Ultralytics on first use
PRIVACY_CLASSES = [0, 1, 2, 3, 5, 7]  # COCO: person, bicycle, car, motorcycle, bus, truck


def blur_boxes(frame, boxes):
    """Return a copy of `frame` with every [x1, y1, x2, y2] box heavily blurred."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        roi = out[y1:y2, x1:x2]
        if roi.size:
            k = max(3, (min(roi.shape[:2]) // 3) | 1)  # odd kernel, scales with box size
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def anonymize(frame, detector, device="cpu", conf=0.25):
    """Detect people/vehicles with `detector` (an Ultralytics YOLO model) and blur them."""
    res = detector.predict(frame, classes=PRIVACY_CLASSES, conf=conf, device=device, verbose=False)[0]
    return blur_boxes(frame, res.boxes.xyxy.tolist())
