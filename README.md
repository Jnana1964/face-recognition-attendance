# Real-Time Face Recognition Based Attendance System Using Deep Facial Embeddings

Computer Vision course project. A webcam recognises registered students and writes their names into the day's attendance sheet. It also has anti-spoofing checks, so a photo or a video call on a phone can't be used for proxy attendance.

## Quick start (Windows)

1. Install Python 3.10 or newer from python.org and **tick "Add python.exe to PATH"**.
2. Double-click **`setup.bat`**. This runs once and takes a few minutes. It ends with a system check, and every line should say `[OK]`.
3. Double-click **`run.bat`**. The app opens in your browser at http://localhost:8501. Keep the black window open; close it to stop the app.

If anything breaks, run `venv\Scripts\python.exe check_system.py`. It prints exactly what is wrong and how to fix it.

## Using the app

1. **Register Student**: type the register number and name, open the "Webcam auto-capture (30)" tab, and click Start. A camera window opens. Turn your head slightly until it reaches 30/30.
2. **Take Attendance → Live webcam**: face the camera. Once you're recognised (5/5), turn your head left and right. You're then marked PRESENT. Press `q` to close the window.
3. **Dashboard, Students, Records & Reports**: today's status, attendance %, and the Excel register download.

## The computer vision pipeline

| Step | Method | Purpose |
|---|---|---|
| 1. Face detection | YuNet CNN | Face boxes plus 5 landmarks (eyes, nose, mouth corners) |
| 2. Alignment | Similarity transform from landmarks | A standard 112×112 face crop |
| 3. Feature extraction | SFace CNN (margin-based loss) | A 128-d embedding per face |
| 4. Matching | Cosine similarity, threshold 0.45 | Identify the student, or "Unknown" |
| 5. Temporal confirmation | 5 consecutive frames | Suppress one-frame false matches |
| 6. Liveness | Head yaw = (nose_x − eyes_mid_x) / eye_distance, must pass ±0.22 on both sides within 10 s | Blocks printed or on-screen photos |
| 7. Screen detection | YOLOX object detector (COCO): cell phone, laptop, tv | Blocks a face shown on a phone, including a live video call |
| 8. Distance rule | Face width ≤ 40% of the frame | Stops a phone being held so close that its edges are hidden |

All models are pretrained (OpenCV model zoo). No training step and no GPU are needed.

## Files

| File | Purpose |
|---|---|
| `setup.bat` / `run.bat` | One-click install / start (Windows) |
| `app.py` | Streamlit web interface |
| `common.py` | Face engine (detect, align, embed), student database, matching |
| `register.py` | Enrol a student from the webcam or from photos |
| `attendance.py` | Live recognition with liveness and screen checks; group-photo mode |
| `liveness.py` | Head-turn liveness check |
| `screen_detect.py` | YOLOX phone/screen detector |
| `report.py` | Excel register (P/A per day, totals, %) |
| `download_models.py` | Re-downloads and size-verifies the models if a file is ever damaged |
| `check_system.py` | Self-test |
| `models/` | The 3 ONNX models (included) |
| `data/`, `attendance/` | Created when you use the app: embeddings and the daily CSV sheets |

## Command line (optional)

```
venv\Scripts\activate
python register.py --id RA2311003012009 --name "Jnana Keerthana"
python attendance.py                  # live, with all protections
python attendance.py --no-liveness    # protections off, for comparison
python attendance.py --image class.jpg
python report.py
```

## Known limitations

- A large monitor with its edges out of view can still fool an ordinary RGB webcam. Fully stopping this needs depth/IR sensors (like Face ID).
- Accuracy drops in poor or uneven lighting, with masks, and at strong side angles.
- Snapshot and photo-upload modes can't prove liveness, so they only preview recognition while liveness is ON.
- Data is stored in CSV files. A real deployment would use a database with access control.
