# Real-Time Face Recognition Based Attendance System Using Deep Facial Embeddings

A webcam-based attendance system that recognises registered students in real time and records attendance automatically. It also checks liveness, so a printed photo or a face shown on a phone (including a live video call) can't be used for proxy attendance.

> Computer Vision course project · B.Tech CSE, SRM Institute of Science and Technology, Kattankulathur
> **Team:** Bollu Jnana Keerthana (RA2311003012009) · Shamil S (RA2311003012051)

---

## Features

- **Real-time face recognition** from a laptop webcam, CPU only, with no GPU needed
- **Anti-spoofing**
  - Head-turn liveness check: a flat photo can't change its head angle
  - Phone / laptop / TV detection: a face shown on a screen is rejected, and nobody is marked while a phone is visible
  - Distance rule: faces too close to the camera are asked to move back
- **5-frame confirmation**, so a single noisy frame can never mark the wrong person
- **Web dashboard** (Streamlit): today's attendance, absentees, a trend chart and attendance %
- **Student management**: register by webcam (30 samples) or photos, with register-number validation
- **Reports**: one CSV per day, plus an Excel register showing P/A, totals and percentage
- **Privacy**: only numeric face embeddings are stored, never face photos

## How it works

| Step | Method | Purpose |
|---|---|---|
| 1. Face detection | YuNet CNN | Face box plus 5 landmarks (eyes, nose, mouth corners) |
| 2. Alignment | Similarity transform from the landmarks | Standard 112×112 face crop |
| 3. Feature extraction | SFace CNN | 128-dimensional face embedding |
| 4. Matching | Cosine similarity, threshold 0.45 | Identify the student, otherwise "Unknown" |
| 5. Confirmation | 5 consecutive frames | Suppress false matches |
| 6. Liveness | Head yaw = (nose_x − eyes_mid_x) / eye_distance, must pass ±0.22 both ways | Reject printed or on-screen photos |
| 7. Screen detection | YOLOX object detector (COCO): cell phone, laptop, TV | Reject faces on screens, including video calls |

All models are pretrained and come from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo).

## Tech stack

Python · OpenCV (DNN: YuNet, SFace, YOLOX) · NumPy · Streamlit · pandas · openpyxl

## Setup (Windows)

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/) and tick **"Add python.exe to PATH"**.
2. Clone the repository:
   ```
   git clone https://github.com/Jnana1964/face-recognition-attendance.git
   cd face-recognition-attendance
   ```
3. Double-click **`setup.bat`**. It creates a virtual environment, installs the libraries, downloads the models (about 75 MB) and runs a system check.
4. Double-click **`run.bat`**. The app opens at `http://localhost:8501`.

**Manual setup:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python check_system.py
streamlit run app.py
```

## Usage

1. **Register Student**: enter the register number (`RA` + 13 digits) and name, then click **Start auto-capture**. Look at the camera and slowly turn your head until it reaches 30/30.
2. **Take Attendance → Start live attendance**: look at the camera. Once you're recognised, turn your head left and right. The window shows **"NAME marked PRESENT"**. Press `q` to finish.
3. **Dashboard / Students / Records & Reports**: view attendance, attendance %, day-wise records, and download the Excel register.

> Only one camera window can run at a time. Close the attendance window (`q`) before registering a new student.

**Command line:**
```
python register.py --id RA2311003012009 --name "Jnana Keerthana"
python attendance.py                  # live, all protections on
python attendance.py --no-liveness    # protections off, for comparison
python attendance.py --image class.jpg
python report.py
```

## Project structure

```
├── app.py              # Streamlit web interface
├── common.py           # Face engine (detect, align, embed), matching, camera handling
├── register.py         # Webcam / photo enrolment
├── attendance.py       # Live attendance with liveness + screen checks
├── liveness.py         # Head-turn liveness check
├── screen_detect.py    # YOLOX phone/screen detector
├── report.py           # Excel attendance register
├── download_models.py  # Downloads and size-verifies the 3 ONNX models
├── check_system.py     # Self-test
├── setup.bat / run.bat # One-click setup and launch (Windows)
└── .streamlit/         # Theme configuration
```

`models/`, `data/` (face embeddings) and `attendance/` (records) are created locally and are **not** in this repository.

## Limitations

- A large monitor with its edges out of view can still fool an ordinary RGB webcam. Fully closing this gap needs depth or IR sensors (like Face ID).
- Accuracy drops in poor or uneven lighting, with masks, and at strong side angles.
- Data is stored in CSV files. A real deployment would use a database with access control.

## References

1. W. Wu, H. Peng, S. Yu, "YuNet: A Tiny Millisecond-level Face Detector," *Machine Intelligence Research*, 2023.
2. Y. Zhong et al., "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition," *IEEE Trans. Image Processing*, 2021.
3. Z. Ge et al., "YOLOX: Exceeding YOLO Series in 2021," arXiv:2107.08430, 2021.
4. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV, 2014.
