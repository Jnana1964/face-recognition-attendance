"""Take attendance: recognise faces and write names into today's attendance sheet.

    python attendance.py                       # live webcam with liveness + phone/screen checks
    python attendance.py --no-liveness         # recognition only (photos CAN pass - for comparison)
    python attendance.py --image class.jpg     # a group photo of the class (teacher use)
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from common import (ATTENDANCE_DIR, MATCH_THRESHOLD, FaceEngine, hold_camera_lock, load_students,
                    match, open_camera, read_frame, release_camera_lock)
from liveness import LivenessChallenge, head_yaw

try:
    from screen_detect import ScreenDetector, face_on_screen
except Exception:  # pragma: no cover
    ScreenDetector = None

CONFIRM_FRAMES = 5     # same person must be recognised in 5 consecutive frames
MAX_FACE_RATIO = 0.40  # face wider than 40% of the frame -> "move back" (stops close-up screens)
SCREEN_EVERY = 2       # run the phone/screen detector every N frames
SCREEN_CONF = 0.25     # detector confidence for phones/laptops/TVs
PHONE_HOLD = 2.0       # seconds after a phone was last seen during which nobody can be marked
BANNER_SECONDS = 4.0   # how long the green "marked PRESENT" banner stays on screen

GREEN, YELLOW, ORANGE, RED, WHITE = (0, 200, 0), (0, 200, 255), (0, 140, 255), (0, 0, 255), (255, 255, 255)


def today_sheet():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, datetime.now().strftime("%Y-%m-%d") + ".csv")


def marked_today():
    """{student_id: time marked} for today."""
    path = today_sheet()
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row.get("time", "") for row in csv.DictReader(f)}


def already_marked():
    return set(marked_today())


def mark(student_id, name, score):
    path = today_sheet()
    new = not os.path.exists(path)
    now = datetime.now().strftime("%H:%M:%S")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "name", "time", "confidence"])
        w.writerow([student_id, name, now, f"{score:.2f}"])
    print(f"[PRESENT] {name} ({student_id})  score={score:.2f}", flush=True)
    return now


def draw(frame, face, label, color):
    x, y, w, h = face[:4].astype(int)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def bar(frame, text, color, top):
    """Full-width text bar at the top or bottom of the window."""
    h, w = frame.shape[:2]
    y0 = 0 if top else h - 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    scale = 0.75
    while scale > 0.4 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 24:
        scale -= 0.05  # shrink long messages so they fit a 640-px webcam window
    cv2.putText(frame, text, (12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def run_image(engine, path, threshold, students):
    ids, names, matrix = students
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read {path}")
    marked = already_marked()
    faces = engine.detect(img)
    print(f"Found {len(faces)} faces")
    for face in faces:
        sid, name, score = match(engine.embed(img, face), ids, names, matrix, threshold)
        if sid:
            if sid not in marked:
                mark(sid, name, score)
                marked.add(sid)
            draw(img, face, f"{name} {score:.2f}", GREEN)
        else:
            draw(img, face, f"Unknown {score:.2f}", RED)
    out = os.path.splitext(path)[0] + "_result.jpg"
    cv2.imwrite(out, img)
    print("Annotated image saved:", out)


def centre(face):
    x, y, w, h = face[:4]
    return np.array([x + w / 2, y + h / 2]), w


def run_webcam(engine, camera, threshold, students, liveness=True):
    ids, names, matrix = students
    registered = set(ids)
    screen_det = None
    if liveness and ScreenDetector is not None:
        try:
            screen_det = ScreenDetector(conf=SCREEN_CONF)
        except FileNotFoundError:
            print("Screen-detection model missing - run download_models.py. Continuing without it.")

    cap = open_camera(camera)
    hold_camera_lock("the live attendance window")
    marked = {k: v for k, v in marked_today().items() if k in registered}
    streak, pending = {}, {}
    screens, phone_seen_at, frame_no, last_lock = [], 0.0, 0, 0.0
    banner = None  # (text, until)
    print("Attendance running" + (" with liveness + phone/screen checks" if liveness else " WITHOUT liveness")
          + ". Press q to stop.", flush=True)
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            now = time.time()
            frame_no += 1
            if now - last_lock > 1:
                hold_camera_lock("the live attendance window")
                last_lock = now

            # ---- phone / screen detection ----
            if screen_det and frame_no % SCREEN_EVERY == 1:
                screens = screen_det.detect(frame)
                if any(s[5] == "cell phone" for s in screens):
                    phone_seen_at = now
            for x, y, w, h, sc, cname in screens:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
                cv2.putText(frame, f"{cname} {sc:.2f}", (int(x), int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            phone_alert = liveness and now - phone_seen_at < PHONE_HOLD
            if phone_alert:
                pending.clear()
                streak.clear()

            instruction, instr_color = "Look at the camera", WHITE
            seen = set()
            for face in engine.detect(frame):
                if liveness and face_on_screen(face, screens):
                    draw(frame, face, "SPOOF: face on a screen", RED)
                    instruction, instr_color = "Face on a phone/screen is NOT accepted", RED
                    continue
                if liveness and face[2] / frame.shape[1] > MAX_FACE_RATIO:
                    draw(frame, face, "Too close - move back", ORANGE)
                    instruction, instr_color = "Move back a little", ORANGE
                    continue
                sid, name, score = match(engine.embed(frame, face), ids, names, matrix, threshold)
                pos, width = centre(face)
                if sid is None:  # keep tracking a person mid head-turn by box position
                    for psid, p in pending.items():
                        if np.linalg.norm(p["pos"] - pos) < width * 0.6:
                            sid, name, score = psid, p["name"], p["score"]
                            break
                if sid is None:
                    draw(frame, face, "Unknown", RED)
                    instruction, instr_color = "Face not recognised - not registered?", RED
                    continue
                seen.add(sid)

                if phone_alert:
                    draw(frame, face, f"{name}: phone detected", RED)
                    continue
                if sid in marked:
                    # Do NOT say PRESENT here - this face has not passed liveness right now.
                    draw(frame, face, f"{name}: already marked at {marked[sid][:5]}", GREEN)
                    instruction, instr_color = f"{name} is already marked - next student, please", GREEN
                    continue

                streak[sid] = streak.get(sid, 0) + 1
                if streak[sid] < CONFIRM_FRAMES:
                    draw(frame, face, f"{name} ({streak[sid]}/{CONFIRM_FRAMES})", YELLOW)
                    instruction, instr_color = f"Hold still - recognising {name}", YELLOW
                    continue

                if liveness:
                    p = pending.setdefault(sid, {"challenge": LivenessChallenge(), "name": name, "score": score})
                    p["pos"] = pos
                    p["challenge"].update(head_yaw(face))
                    if not p["challenge"].passed:
                        draw(frame, face, f"{name}: turn head", ORANGE)
                        side = ("now the OTHER way" if (p["challenge"].turned_one_side or
                                                       p["challenge"].turned_other_side) else "LEFT and RIGHT")
                        instruction, instr_color = f"{name}: turn your head {side}", ORANGE
                        continue
                    pending.pop(sid)
                    score = p["score"]

                t = mark(sid, name, score)
                marked[sid] = t
                draw(frame, face, f"{name} - PRESENT", GREEN)
                banner = (f"{name} marked PRESENT at {t[:5]}", now + BANNER_SECONDS)

            for sid in list(streak):
                if sid not in seen:
                    streak[sid] = 0
            for sid in list(pending):
                if sid not in seen:
                    pending.pop(sid)  # left the frame - challenge restarts

            # ---- top and bottom bars ----
            if banner and now < banner[1]:
                bar(frame, "OK  " + banner[0], (80, 255, 80), top=True)
            else:
                bar(frame, f"Present today: {len(marked)}/{len(ids)}   |   press q to finish", WHITE, top=True)
            if phone_alert:
                bar(frame, "PHONE DETECTED - put it away. Nobody is marked while it is visible.", RED, top=False)
            else:
                bar(frame, instruction, instr_color, top=False)

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        release_camera_lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Group photo instead of webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--no-liveness", action="store_true", help="Disable anti-spoofing")
    args = ap.parse_args()

    students = load_students()
    if not students[0]:
        raise SystemExit("No students registered. Run register.py first.")
    engine = FaceEngine()
    if args.image:
        run_image(engine, args.image, args.threshold, students)
    else:
        run_webcam(engine, args.camera, args.threshold, students, liveness=not args.no_liveness)
    print("Sheet:", today_sheet())


if __name__ == "__main__":
    main()