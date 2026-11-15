"""Self-test: run this if anything goes wrong. It tells you exactly what is broken.

    python check_system.py
"""
import os
import sys

OK, BAD = "  [OK]  ", "  [FAIL]"
problems = 0


def report(good, msg, fix=""):
    global problems
    print((OK if good else BAD), msg)
    if not good:
        problems += 1
        if fix:
            print("         FIX:", fix)


print("\n=== FaceAttend system check ===\n")
v = sys.version_info
report(v >= (3, 9), f"Python {v.major}.{v.minor}.{v.micro}", "Install Python 3.10 or newer from python.org")
report(sys.prefix != sys.base_prefix, "Running inside the venv",
       "Use run.bat / setup.bat, or run: venv\\Scripts\\activate")

try:
    import cv2
    import numpy  # noqa: F401
    import openpyxl  # noqa: F401
    import pandas  # noqa: F401
    import streamlit  # noqa: F401
    report(True, f"Libraries installed (OpenCV {cv2.__version__}, Streamlit {streamlit.__version__})")
except ImportError as e:
    report(False, f"Missing library: {e.name}", "Run setup.bat again (or: pip install -r requirements.txt)")
    sys.exit(1)

report(hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"),
       "OpenCV has the face modules",
       "pip uninstall -y opencv-python opencv-contrib-python && pip install -r requirements.txt")

from common import DET_MODEL, REC_MODEL  # noqa: E402
from screen_detect import SCREEN_MODEL  # noqa: E402

SIZES = {DET_MODEL: 232_589, REC_MODEL: 38_696_353, SCREEN_MODEL: 35_858_002}
for path, size in SIZES.items():
    name = os.path.basename(path)
    actual = os.path.getsize(path) if os.path.exists(path) else 0
    report(actual == size, f"Model file {name} ({actual:,} / {size:,} bytes)",
           "Run: python download_models.py")

try:
    cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320))
    cv2.FaceRecognizerSF.create(REC_MODEL, "")
    cv2.dnn.readNet(SCREEN_MODEL)
    report(True, "All 3 models load correctly")
except Exception as e:
    report(False, f"Model failed to load: {str(e)[:120]}", "Run: python download_models.py")

from common import open_camera  # noqa: E402
try:
    cap = open_camera(0)
    ok, frame = cap.read()
    cap.release()
except SystemExit:
    ok, frame = False, None
report(ok, "Webcam works" + (f" ({frame.shape[1]}x{frame.shape[0]})" if ok else ""),
       "Camera busy: close browser tabs using the camera, Zoom/Teams/Camera app; then Settings > Privacy > Camera > allow desktop apps")

print("\nRESULT:", "everything is ready. Start with run.bat" if problems == 0
      else f"{problems} problem(s) - follow the FIX lines above.")