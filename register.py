"""Register a student: capture face samples from the webcam (or a folder of photos).

    python register.py --id RA2311003012009 --name "Jnana Keerthana"
    python register.py --id RA231100301XXXX --name "Friend" --images photos/friend
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from common import (FaceEngine, hold_camera_lock, load_students, match, open_camera, read_frame,
                    release_camera_lock, save_student)


def from_images(engine, folder):
    samples = []
    files = sorted(glob.glob(os.path.join(folder, "*")))
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        faces = engine.detect(img)
        if len(faces) != 1:
            print(f"  skip {os.path.basename(path)}: found {len(faces)} faces (need exactly 1)")
            continue
        samples.append(engine.embed(img, faces[0]))
        print(f"  ok   {os.path.basename(path)}")
    return samples


def from_webcam(engine, n_samples, camera):
    cap = open_camera(camera)
    hold_camera_lock("the registration window")
    samples, last, last_lock, frame = [], 0.0, 0.0, None
    print("Look at the camera. Slowly turn your head a little left/right/up/down. Press q to quit.", flush=True)
    try:
        while len(samples) < n_samples:
            frame = read_frame(cap)
            if frame is None:
                break
            if time.time() - last_lock > 1:
                hold_camera_lock("the registration window")
                last_lock = time.time()
            faces = engine.detect(frame)
            msg, color = "Look at the camera and slowly turn your head a little", (255, 255, 255)
            if len(faces) == 1:
                x, y, w, h = faces[0][:4].astype(int)
                big_enough = w > 100
                if big_enough and time.time() - last > 0.25:  # ~4 samples/sec, varied poses
                    samples.append(engine.embed(frame, faces[0]))
                    last = time.time()
                color = (0, 200, 0) if big_enough else (0, 165, 255)
                if not big_enough:
                    msg = "Come closer to the camera"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            elif len(faces) > 1:
                msg, color = "Only ONE person in front of the camera", (0, 0, 255)
            else:
                msg, color = "No face visible - face the camera", (0, 0, 255)
            H, W = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (W, 50), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 34), (10 + int((W - 20) * len(samples) / n_samples), 44), (0, 200, 0), -1)
            cv2.putText(frame, f"Registering: {len(samples)}/{n_samples}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, H - 40), (W, H), (20, 20, 20), -1)
            cv2.putText(frame, msg, (10, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Register", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if frame is not None and len(samples) >= n_samples:
            H, W = frame.shape[:2]
            cv2.rectangle(frame, (0, H // 2 - 35), (W, H // 2 + 35), (0, 120, 0), -1)
            cv2.putText(frame, "DONE - all samples captured", (20, H // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Register", frame)
            cv2.waitKey(1500)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        release_camera_lock()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Register number, e.g. RA2311003012009")
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", help="Folder of photos instead of webcam")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    engine = FaceEngine()
    samples = from_images(engine, args.images) if args.images else from_webcam(engine, args.samples, args.camera)
    if len(samples) < 3:
        raise SystemExit(f"Only {len(samples)} good samples - not saved. Try again with better light.")

    # Warn if this face already belongs to someone else (catches duplicate registration)
    ids, names, matrix = load_students()
    mean = np.mean(samples, axis=0)
    mean /= np.linalg.norm(mean)
    other_id, other_name, score = match(mean, ids, names, matrix, threshold=0.6)
    if other_id and other_id != args.id:
        print(f"WARNING: this face looks like {other_name} ({other_id}), similarity {score:.2f}")

    save_student(args.id, args.name, samples)
    print(f"Registered {args.name} ({args.id}) with {len(samples)} samples.", flush=True)


if __name__ == "__main__":
    main()