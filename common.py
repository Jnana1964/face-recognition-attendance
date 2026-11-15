"""Shared code: paths, the face engine (detect + recognise), student database."""
import csv
import os
import sys

import time

import cv2
import numpy as np

# Hide OpenCV's harmless "Targets are not supported by the new graph engine" warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
DET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

DATA_DIR = os.path.join(BASE, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")
ATTENDANCE_DIR = os.path.join(BASE, "attendance")

# Cosine similarity needed to accept a match. SFace authors suggest 0.363;
# 0.45 is stricter = fewer wrong names, slightly more "Unknown".
MATCH_THRESHOLD = 0.45


CAMERA_HELP = """
Could not get video from the webcam. The camera is probably BUSY in another program.
  1. Close any browser tab using the camera (the Snapshot tab in this app, Meet, etc.)
  2. Close Zoom / Teams / the Windows Camera app
  3. Windows Settings > Privacy & security > Camera > allow desktop apps
  4. Still failing? Try another camera:  --camera 1
"""


def open_camera(index=0):
    """Open the webcam robustly. On Windows try DirectShow first, then Media Foundation.
    Returns an opened cv2.VideoCapture that has delivered a real frame, or exits with help."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(30):  # cameras often need a moment to warm up
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
    sys.exit(CAMERA_HELP)


CAMERA_LOCK = os.path.join(DATA_DIR, ".camera_in_use")


def hold_camera_lock(owner):
    """Mark the webcam as busy (refresh every second while the window is open)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CAMERA_LOCK, "w", encoding="utf-8") as f:
        f.write(owner)


def release_camera_lock():
    try:
        os.remove(CAMERA_LOCK)
    except OSError:
        pass


def camera_in_use():
    """Name of the window using the webcam, or None. A lock not refreshed for 5 s is stale."""
    try:
        if time.time() - os.path.getmtime(CAMERA_LOCK) < 5:
            with open(CAMERA_LOCK, encoding="utf-8") as f:
                return f.read().strip() or "another camera window"
    except OSError:
        pass
    return None


def read_frame(cap, max_failures=30):
    """Read a frame, tolerating brief glitches. Returns None if the camera has really stopped."""
    for _ in range(max_failures):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.03)
    print(CAMERA_HELP)
    return None


class FaceEngine:
    """YuNet face detector + SFace face recogniser (both built into OpenCV)."""

    def __init__(self, det_score=0.8):
        for p in (DET_MODEL, REC_MODEL):
            if not os.path.exists(p):
                sys.exit(f"Model missing: {p}\nRun:  python download_models.py")
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), det_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")

    def detect(self, frame):
        """Return array of faces, each row: x, y, w, h, 5 landmarks (10 values), score."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def embed(self, frame, face):
        """Align the face using its landmarks and return a unit-length 128-d embedding."""
        aligned = self.recognizer.alignCrop(frame, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)


def load_students():
    """Return (ids, names, matrix) where matrix[i] is the mean embedding of student i."""
    ids, names, embs = [], [], []
    if not os.path.exists(STUDENTS_CSV):
        return ids, names, np.empty((0, 128), dtype=np.float32)
    with open(STUDENTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = os.path.join(EMB_DIR, f"{row['id']}.npy")
            if not os.path.exists(path):
                continue
            samples = np.load(path)
            mean = samples.mean(axis=0)
            embs.append(mean / np.linalg.norm(mean))
            ids.append(row["id"])
            names.append(row["name"])
    matrix = np.vstack(embs) if embs else np.empty((0, 128), dtype=np.float32)
    return ids, names, matrix


def save_student(student_id, name, samples):
    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(os.path.join(EMB_DIR, f"{student_id}.npy"), np.asarray(samples, dtype=np.float32))
    rows = []
    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["id"] != student_id]
    rows.append({"id": student_id, "name": name})
    with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader()
        w.writerows(rows)


def match(embedding, ids, names, matrix, threshold=MATCH_THRESHOLD):
    """Return (id, name, score); id/name are None when no one is close enough."""
    if len(ids) == 0:
        return None, None, 0.0
    scores = matrix @ embedding
    i = int(np.argmax(scores))
    score = float(scores[i])
    if score < threshold:
        return None, None, score
    return ids[i], names[i], score