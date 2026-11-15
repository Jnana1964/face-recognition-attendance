"""Screen / phone detection (anti-spoofing against photos and video calls on a device).

Uses YOLOX-S (COCO, 80 classes) from the OpenCV model zoo. If a recognised face lies
inside a detected cell phone, laptop or TV, the face is being shown on a screen -> spoof.

Decoding follows the official opencv_zoo YOLOX demo.
"""
import os

import cv2
import numpy as np

from common import MODELS_DIR

SCREEN_MODEL = os.path.join(MODELS_DIR, "object_detection_yolox_2022nov.onnx")
SCREEN_CLASSES = {62: "tv", 63: "laptop", 67: "cell phone"}
INPUT = 640


class ScreenDetector:
    def __init__(self, conf=0.30, nms=0.5):
        if not os.path.exists(SCREEN_MODEL):
            raise FileNotFoundError(SCREEN_MODEL)
        self.net = cv2.dnn.readNet(SCREEN_MODEL)
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for s in (8, 16, 32):
            n = INPUT // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
            strides.append(np.full((1, n * n, 1), s))
        self.grids = np.concatenate(grids, 1)
        self.strides = np.concatenate(strides, 1)

    def detect(self, frame, classes=SCREEN_CLASSES):
        """Return list of (x, y, w, h, score, class_name) in frame coordinates."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ratio = min(INPUT / rgb.shape[0], INPUT / rgb.shape[1])
        padded = np.full((INPUT, INPUT, 3), 114.0, dtype=np.float32)
        h, w = int(rgb.shape[0] * ratio), int(rgb.shape[1] * ratio)
        padded[:h, :w] = cv2.resize(rgb, (w, h)).astype(np.float32)
        self.net.setInput(np.transpose(padded, (2, 0, 1))[np.newaxis])
        dets = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0].copy()

        dets[:, :2] = (dets[:, :2] + self.grids[0]) * self.strides[0]
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.strides[0]
        boxes = np.stack([dets[:, 0] - dets[:, 2] / 2, dets[:, 1] - dets[:, 3] / 2,
                          dets[:, 2], dets[:, 3]], 1)
        scores = dets[:, 4:5] * dets[:, 5:]
        cls = scores.argmax(1)
        best = scores.max(1)
        mask = np.isin(cls, list(classes)) & (best > self.conf) if classes else best > self.conf
        if not mask.any():
            return []
        b, s, c = boxes[mask], best[mask], cls[mask]
        keep = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), self.conf, self.nms)
        out = []
        for i in np.array(keep).flatten():
            x, y, bw, bh = b[i] / ratio
            name = classes.get(int(c[i]), str(int(c[i]))) if classes else int(c[i])
            out.append((float(x), float(y), float(bw), float(bh), float(s[i]), name))
        return out


def face_on_screen(face, screens, margin=0.15):
    """True if the face centre lies inside any detected screen box (expanded by margin)."""
    fx, fy, fw, fh = face[:4]
    cx, cy = fx + fw / 2, fy + fh / 2
    for x, y, w, h, _, _ in screens:
        mx, my = w * margin, h * margin
        if x - mx <= cx <= x + w + mx and y - my <= cy <= y + h + my:
            return True
    return False
