"""Download the pretrained ONNX models from the official OpenCV model zoo.

Each file is checked against its exact expected size, so a broken or half-finished
download is detected and downloaded again automatically.
"""
import os
import sys
import urllib.request

from common import DET_MODEL, MODELS_DIR, REC_MODEL
from screen_detect import SCREEN_MODEL

BASE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
FILES = {
    DET_MODEL: (BASE_URL + "face_detection_yunet/face_detection_yunet_2023mar.onnx", 232_589),
    REC_MODEL: (BASE_URL + "face_recognition_sface/face_recognition_sface_2021dec.onnx", 38_696_353),
    SCREEN_MODEL: (BASE_URL + "object_detection_yolox/object_detection_yolox_2022nov.onnx", 35_858_002),
}


def progress(blocks, block_size, total):
    if total > 0:
        pct = min(100, blocks * block_size * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%  ({blocks * block_size // 1_048_576} / {total // 1_048_576} MB)")
        sys.stdout.flush()


os.makedirs(MODELS_DIR, exist_ok=True)
ok = True
for path, (url, size) in FILES.items():
    name = os.path.basename(path)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print("OK (already downloaded):", name)
        continue
    if os.path.exists(path):
        print(f"Broken/incomplete file ({os.path.getsize(path):,} of {size:,} bytes): {name} - downloading again")
        os.remove(path)
    print("Downloading", name)
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, progress)
        print()
    except Exception as e:
        print(f"\n  FAILED: {e}\n  Check your internet and run this script again.")
        ok = False
        continue
    if os.path.getsize(tmp) != size:
        print(f"  Size mismatch ({os.path.getsize(tmp):,} bytes) - run this script again.")
        os.remove(tmp)
        ok = False
        continue
    os.replace(tmp, path)
    print("  saved and verified", name)

print("All models ready." if ok else "Some downloads failed - run: python download_models.py")
