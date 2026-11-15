"""Web interface for the Face Recognition Attendance System.

    streamlit run app.py
"""
import os
import subprocess
import sys
from datetime import date, datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance import already_marked, draw, mark, marked_today, today_sheet
from common import (ATTENDANCE_DIR, BASE, EMB_DIR, MATCH_THRESHOLD, STUDENTS_CSV,
                    FaceEngine, camera_in_use, load_students, match, save_student)
from report import build_report, load_register

st.set_page_config(page_title="Face Attendance", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --base:#0D1117; --surface:#161B22; --surface2:#1C2128; --border:#30363D; --border-soft:#21262D;
  --fg:#E6EDF3; --muted:#8B949E; --accent:#14B8A6; --accent-600:#2DD4BF;
  --gold:#14B8A6; --gold-bright:#2DD4BF; --success:#22c55e; --danger:#ef4444;
}
html, body, .stApp {font-family:'Inter', system-ui, sans-serif;}
[data-testid="stIconMaterial"], span[class*="material-symbols"] {font-family:'Material Symbols Rounded' !important;}
.stApp {background: var(--base);}
#MainMenu, footer, [data-testid="stDecoration"] {visibility:hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3, h4, [data-testid="stHeading"] *, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-family:'Inter', system-ui, sans-serif !important; font-weight:600 !important; letter-spacing:-0.01em; color:var(--fg);}
::-webkit-scrollbar {width:8px; height:8px;} ::-webkit-scrollbar-track {background:var(--base);}
::-webkit-scrollbar-thumb {background:#2a2a2a; border-radius:8px;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {background: var(--base); border-right: 1px solid var(--border);}
section[data-testid="stSidebar"] * {color: var(--muted);}
section[data-testid="stSidebar"] hr {border-color: var(--border); margin: 18px 0;}
.brand {display:flex; align-items:center; gap:12px; margin: 6px 0 2px 0;}
.brand-logo {width:42px; height:42px; border-radius:12px; background:rgba(20,184,166,0.10);
  border:1px solid rgba(20,184,166,0.40); display:flex; align-items:center; justify-content:center; font-size:22px;}
section[data-testid="stSidebar"] .brand-title {font-family:'Inter', system-ui, sans-serif; font-size:1.25rem; letter-spacing:-0.01em;
  font-weight:600; line-height:1; color: var(--fg) !important;}
section[data-testid="stSidebar"] .brand-title span {color: var(--gold-bright) !important;}
section[data-testid="stSidebar"] .brand-sub {font-size:0.68rem; letter-spacing:0.22em; text-transform:uppercase;
  color: var(--gold) !important; font-weight:600; margin-top:4px;}
/* nav: radio styled as PrepWise nav links */
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 6px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: 10px 14px; border-radius: 14px; width: 100%;
  border: 1px solid transparent; transition: background-color .15s, border-color .15s;}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {font-size: 0.92rem; font-weight: 500;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: var(--surface2);}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {color: #fff;}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {background: rgba(20,184,166,0.10);
  border-color: rgba(20,184,166,0.40);}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p {color: var(--accent) !important; font-weight:600;}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {display: none;}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color: var(--fg) !important; font-weight:500;}

/* ---------- hero ---------- */
.hero {background: radial-gradient(circle at 100% 0%, rgba(20,184,166,0.14), transparent 45%),
  radial-gradient(circle at 0% 100%, rgba(56,139,253,0.05), transparent 40%), var(--surface);
  border: 1px solid var(--border); border-radius: 20px; padding: 28px 32px; margin-bottom: 26px;}
.hero .eyebrow {font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--gold); font-weight: 600;}
.hero .hero-title {font-family:'Inter', system-ui, sans-serif !important; color: var(--fg); font-size: 2.3rem;
  font-weight: 700; line-height: 1.15; margin: 6px 0 0 0; letter-spacing: -0.02em;}
.hero .hero-title em {font-style: normal; color: var(--gold-bright); font-family: inherit !important;}
.hero p {color: var(--muted); margin: 8px 0 0 0; font-size: 0.98rem;}

/* ---------- cards / metrics ---------- */
div[data-testid="stMetric"] {background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
  padding: 18px 22px; transition: transform .15s, border-color .15s;}
div[data-testid="stMetric"]:hover {transform: scale(1.02); border-color: rgba(20,184,166,0.40);}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {font-size: 0.72rem !important; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted) !important; font-weight: 600;}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {font-family:'Inter', system-ui, sans-serif !important;
  font-size: 2.2rem !important; font-weight: 700 !important; color: var(--fg) !important; line-height: 1.2; letter-spacing: -0.02em;}
div[data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 20px; overflow: hidden;}
div[data-testid="stExpander"] details {border: 1px solid var(--border) !important; border-radius: 14px; background: var(--surface);}
div[data-testid="stAlert"] {border-radius: 14px;}

/* ---------- buttons & inputs ---------- */
.stButton button, .stDownloadButton button {border-radius: 14px; font-weight: 600; padding: 0.55rem 1.3rem;
  transition: background-color .15s, border-color .15s;}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"],
[data-testid="stBaseButton-primary"] {background: var(--accent); border: 1px solid var(--accent); color: #0D1117 !important;}
[data-testid="stBaseButton-primary"] p {color: #0D1117 !important; font-weight: 600;}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {background: var(--accent-600); border-color: var(--accent-600);}
.stButton button[kind="secondary"], [data-testid="stBaseButton-secondary"] {background: transparent;
  border: 1px solid var(--accent); color: var(--accent);}
.stButton button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {background: rgba(20,184,166,0.10);
  color: var(--accent); border-color: var(--accent);}
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {background: var(--surface2) !important;
  border: 1px solid var(--border) !important; border-radius: 14px !important;}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(20,184,166,0.15);}
div[data-baseweb="input"] input {background: transparent !important;}
button[data-baseweb="tab"] p {font-weight: 500;}
[data-testid="stFileUploaderDropzone"] {background: var(--surface); border: 1px dashed var(--border); border-radius: 20px;}

/* ---------- status text / pills ---------- */
.present {color: var(--success); font-weight:600;}
.unknown {color: var(--danger); font-weight:600;}
.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.75rem; font-weight:600;}
.pill-on {background: rgba(34,197,94,0.10); color: var(--success);}
.pill-off {background: rgba(239,68,68,0.10); color: var(--danger);}
.page-fade {animation: fadein 200ms ease-out;}
@keyframes fadein {from {opacity:0;} to {opacity:1;}}
</style>""", unsafe_allow_html=True)


def falling_stars(count=45):
    """Gold stars that fall from the top of the screen once, then fade away (pure CSS, no clicks blocked)."""
    import random
    stars = []
    for _ in range(count):
        left = random.uniform(0, 100)
        size = random.uniform(14, 34)
        delay = random.uniform(0, 1.6)
        dur = random.uniform(2.2, 3.6)
        spin = random.choice([-1, 1]) * random.randint(180, 540)
        color = random.choice(["#14B8A6", "#2DD4BF", "#5EEAD4", "#E6EDF3", "#FCD34D"])
        stars.append(f"<span class='star' style='left:{left:.1f}vw;font-size:{size:.0f}px;color:{color};"
                     f"animation-duration:{dur:.2f}s;animation-delay:{delay:.2f}s;--spin:{spin}deg'>★</span>")
    st.markdown("""<style>
.starfall {position:fixed; inset:0; pointer-events:none; z-index:999999; overflow:hidden;}
.starfall .star {position:absolute; top:-50px; opacity:0; text-shadow:0 0 12px rgba(45,212,191,0.8);
    animation-name:fall; animation-timing-function:cubic-bezier(.3,.1,.6,1); animation-fill-mode:forwards;}
@keyframes fall {0% {transform:translateY(0) rotate(0deg); opacity:0;}
                 10% {opacity:1;} 80% {opacity:1;}
                 100% {transform:translateY(105vh) rotate(var(--spin)); opacity:0;}}
</style><div class='starfall'>""" + "".join(stars) + "</div>", unsafe_allow_html=True)


EYEBROWS = {"Dashboard": "Overview", "Take Attendance": "Live recognition", "Register Student": "Enrolment",
            "Students": "Directory", "Records & Reports": "History", "How it works": "Computer vision pipeline"}


def hero(title, subtitle):
    """PrepWise-style surface card with a Luxora serif headline; the last word is set in gold."""
    words = title.split(" ")
    head = (" ".join(words[:-1]) + " <em>" + words[-1] + "</em>") if len(words) > 1 else f"<em>{title}</em>"
    st.markdown(f"<div class='hero page-fade'><div class='eyebrow'>FaceAttend · {EYEBROWS.get(title, '')}</div>"
                f"<div class='hero-title'>{head}</div><p>{subtitle}</p></div>", unsafe_allow_html=True)


def models_ready():
    from common import DET_MODEL, REC_MODEL
    missing = [os.path.basename(p) for p in (DET_MODEL, REC_MODEL) if not os.path.exists(p)]
    if missing:
        st.error("Face models are missing: " + ", ".join(missing) + "\n\nStop the app (Ctrl + C in the "
                 "terminal), run `python download_models.py`, then `streamlit run app.py` again.")
        st.stop()


@st.cache_resource
def get_engine():
    models_ready()
    return FaceEngine()


def to_bgr(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def recognise(img, threshold, do_mark=True):
    """Recognise every face in img, mark attendance, return (annotated_rgb, results)."""
    engine = get_engine()
    ids, names, matrix = load_students()
    marked = already_marked()
    results = []
    out = img.copy()
    for face in engine.detect(img):
        sid, name, score = match(engine.embed(img, face), ids, names, matrix, threshold)
        if sid is None:
            draw(out, face, f"Unknown {score:.2f}", (0, 0, 255))
            results.append(("Unknown", "-", score, "Not recognised"))
            continue
        if sid in marked:
            status = "Already marked"
        elif do_mark:
            mark(sid, name, score)
            marked.add(sid)
            status = "Marked present"
        else:
            status = "Recognised"
        draw(out, face, f"{name} {score:.2f}", (0, 200, 0))
        results.append((name, sid, score, status))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), results


def run_camera_tool(cmd, waiting_msg):
    """Run register.py / attendance.py in its own camera window, wait for it, return its output.
    Returns (None, False) if another camera window from this app is still open."""
    owner = camera_in_use()
    if owner:
        st.error(f"🚫 The webcam is being used by **{owner}**. One webcam works with only one window at a "
                 "time: click that window and press **q** to close it, then press this button again.")
        return None, False
    with st.spinner(waiting_msg):
        res = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    log = "\n".join(l for l in out.splitlines() if l.strip() and "WARN" not in l and "graph engine" not in l)
    busy = any(k in out for k in ("BUSY", "Could not", "can't grab", "Could not open webcam"))
    return log, busy


def delete_attendance(student_id=None, day=None):
    """Delete attendance rows: a whole day, one student everywhere, or both. Returns rows removed."""
    import glob
    removed = 0
    files = [os.path.join(ATTENDANCE_DIR, f"{day}.csv")] if day else glob.glob(os.path.join(ATTENDANCE_DIR, "*.csv"))
    for path in files:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str)
        keep = df[df["id"] != student_id] if student_id else df.iloc[0:0]
        removed += len(df) - len(keep)
        if len(keep):
            keep.to_csv(path, index=False)
        else:
            os.remove(path)
    return removed


CAMERA_BUSY_MSG = ("**The webcam could not be used.** Another program is holding it. Close any browser tab "
                   "that shows a camera icon in the address bar, close Zoom/Teams/Camera app and any other "
                   "camera window from this app, then press the button again.")


def show_results(annotated, results):
    c1, c2 = st.columns([3, 2])
    c1.image(annotated, width="stretch")
    with c2:
        if not results:
            st.warning("No face detected. Face the camera with good lighting.")
        for name, sid, score, status in results:
            css = "unknown" if name == "Unknown" else "present"
            st.markdown(f"<span class='{css}'>{name}</span> &nbsp; `{sid}` &nbsp; "
                        f"similarity **{score:.2f}** — {status}", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
st.sidebar.markdown("""<div class='brand'><div class='brand-logo'>🎓</div>
<div><div class='brand-title'>Face<span>Attend</span></div><div class='brand-sub'>Vision attendance</div></div></div>""",
                    unsafe_allow_html=True)
st.sidebar.markdown("---")
PAGES = {"📊  Dashboard": "Dashboard", "📸  Take Attendance": "Take Attendance",
         "➕  Register Student": "Register Student", "👥  Students": "Students",
         "📁  Records & Reports": "Records & Reports", "🧠  How it works": "How it works"}
page = PAGES[st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")]
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Match threshold", 0.30, 0.70, MATCH_THRESHOLD, 0.01,
                              help="Higher = stricter (fewer wrong names, more 'Unknown').")
liveness = st.sidebar.toggle("Liveness check (anti-spoofing)", value=True,
                             help="Live mode requires a head turn, so a photo of a student cannot mark them. "
                                  "Snapshot and photo-upload modes only preview recognition while this is on.")
st.sidebar.markdown("---")
st.sidebar.caption("YuNet · SFace · head-turn liveness · YOLOX screen detection")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    hero("Dashboard", datetime.now().strftime("%A, %d %B %Y") + " · today's attendance at a glance")
    ids, names, _ = load_students()
    present = already_marked() & set(ids)          # only students who are still registered
    students, days = load_register()
    total = len(ids)
    pct = (len(present) / total * 100) if total else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered students", total)
    m2.metric("Present today", len(present))
    m3.metric("Absent today", max(total - len(present), 0))
    m4.metric("Attendance today", f"{pct:.0f}%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Today's attendance")
        today = pd.read_csv(today_sheet(), dtype=str) if os.path.exists(today_sheet()) else None
        if today is not None:
            today = today[today["id"].isin(ids)]
        if today is not None and len(today):
            st.dataframe(today, width="stretch", hide_index=True)
        else:
            st.info("No registered student has been marked yet today. Go to **Take Attendance**.")
    with right:
        st.subheader("Absent today")
        absent = [f"{n} ({i})" for i, n in zip(ids, names) if i not in present]
        st.write("\n".join(f"- {a}" for a in absent) if absent else "Everyone is present 🎉")

    if days:
        st.subheader("Attendance trend")
        trend = pd.DataFrame({"Present": [len(v & set(ids)) for v in days.values()]}, index=list(days))
        st.bar_chart(trend, color="#14B8A6")

# ---------------- Take attendance ----------------
elif page == "Take Attendance":
    hero("Take Attendance", "Recognise registered students and mark them present")
    if not load_students()[0]:
        st.error("No students registered yet. Register at least one student first.")
        st.stop()
    pill = ("<span class='pill pill-on'>Liveness ON</span>" if liveness
            else "<span class='pill pill-off'>Liveness OFF — photos can mark attendance</span>")
    st.markdown(pill, unsafe_allow_html=True)
    tab3, tab2 = st.tabs(["🎥 Live webcam (recommended)", "🖼️ Group photo"])

    with tab3:
        st.write("Opens a live video window. Each student is recognised over 5 frames"
                 + (", then must **turn their head left and right** to prove they are a real person. "
                    "Faces on a phone/laptop screen are rejected, and **nobody is marked while a phone is "
                    "visible**." if liveness else ".")
                 + " The window itself shows a green **'NAME marked PRESENT'** banner and step-by-step "
                   "instructions. Press **q** in that window when everyone is done.")
        if st.button("Start live attendance", type="primary"):
            cmd = [sys.executable, os.path.join(BASE, "attendance.py"), "--threshold", str(threshold)]
            if not liveness:
                cmd.append("--no-liveness")
            log, busy = run_camera_tool(cmd, "The live attendance window is open (check the taskbar if you "
                                             "can't see it). Press q in that window to finish…")
            if log is None:
                st.stop()
            marked = [l.replace("[PRESENT]", "").strip() for l in log.splitlines() if l.startswith("[PRESENT]")]
            if marked:
                st.success("✅ Marked present: " + ", ".join(marked))
            elif busy:
                st.error(CAMERA_BUSY_MSG)
            else:
                st.info("Window closed. No new students were marked this time.")
            with st.expander("Technical log"):
                st.code(log or "(no output)")

    with tab2:
        st.write("Upload a photo of the whole class. All recognised students are "
                 + ("previewed." if liveness else "marked at once."))
        if liveness:
            st.info("Liveness is ON, so this mode only **previews** recognition — a still photo cannot prove "
                    "the person is really there. Switch liveness off in the sidebar to mark from a "
                    "teacher-taken class photo.")
        up = st.file_uploader("Class photo", type=["jpg", "jpeg", "png"])
        if up is not None:
            show_results(*recognise(to_bgr(up), threshold, do_mark=not liveness))

# ---------------- Register ----------------
elif page == "Register Student":
    hero("Register Student", "Enrol a new face into the system")
    c1, c2 = st.columns(2)
    sid = c1.text_input("Register number", placeholder="e.g. RA2311003012009").strip().upper().replace(" ", "")
    name = " ".join(c2.text_input("Full name", placeholder="e.g. Jnana Keerthana").split())

    # Validate before anything is captured: SRM register numbers are RA + 13 digits.
    import re
    problems = []
    if sid and not re.fullmatch(r"RA\d{13}", sid):
        digits = len(re.sub(r"\D", "", sid))
        problems.append(f"Register number **{sid}** is not valid. It must be **RA followed by 13 digits** "
                        f"(e.g. RA2311003012009). You typed {digits} digit(s).")
    if name and re.search(r"\d", name):
        problems.append("The full name contains numbers. Did you swap the name and register number?")
    if name and len(name) < 2:
        problems.append("Name is too short.")
    for p in problems:
        st.error(p)
    if problems:
        sid, name = "", ""   # disables the capture/save buttons until fixed

    engine = get_engine()
    t3, t2 = st.tabs(["🎥 Webcam auto-capture (recommended)", "🖼️ Upload photos"])

    with t3:
        st.write("Opens a webcam window that captures 30 samples automatically. Sit at arm's length, "
                 "face a light, and slowly turn your head a little until it reaches 30/30.")
        if not (sid and name):
            st.caption("Type the register number and full name above first.")
        if st.button("Start auto-capture", type="primary", disabled=not (sid and name)):
            ids_now = load_students()[0]
            if sid in ids_now:
                st.warning(f"{sid} is already registered — this capture will REPLACE their face data.")
            log, busy = run_camera_tool(
                [sys.executable, os.path.join(BASE, "register.py"), "--id", sid, "--name", name],
                "The capture window is open (check the taskbar if you can't see it). "
                "Look at the camera until it reaches 30/30…")
            if log is None:
                st.stop()
            if sid in load_students()[0] and "Registered" in log:
                st.success(f"✅ Registered {name} ({sid}). Check the Students page.")
                falling_stars()
            elif busy:
                st.error(CAMERA_BUSY_MSG)
            else:
                st.error("❌ Not saved. The window closed before 30/30 (or too few clear face samples). "
                         "Try again with better light, facing the camera.")
            with st.expander("Technical log"):
                st.code(log or "(no output)")

    with t2:
        st.write("Or upload **at least 5** clear photos of the student (one face per photo).")
        if "samples" not in st.session_state:
            st.session_state.samples = []

        def add_sample(img, label):
            faces = engine.detect(img)
            if len(faces) != 1:
                st.warning(f"{label}: found {len(faces)} faces — need exactly one.")
                return
            st.session_state.samples.append(engine.embed(img, faces[0]))

        files = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files and st.button("Add uploaded photos"):
            for f in files:
                add_sample(to_bgr(f), f.name)
        n = len(st.session_state.samples)
        st.progress(min(n / 5, 1.0), text=f"{n} photo sample(s) collected")
        b1, b2 = st.columns([1, 5])
        if b1.button("Save student", disabled=not (sid and name and n >= 5)):
            save_student(sid, name, st.session_state.samples)
            st.session_state.samples = []
            st.success(f"✅ Registered {name} ({sid}).")
            falling_stars()
        if b2.button("Clear samples"):
            st.session_state.samples = []
            st.rerun()

# ---------------- Students ----------------
elif page == "Students":
    hero("Students", "Everyone enrolled, with their attendance percentage")
    students, days = load_register()
    if not students:
        st.info("No students registered yet.")
    else:
        rows = []
        for s in students:
            p = sum(s["id"] in v for v in days.values())
            rows.append({"Reg. No": s["id"], "Name": s["name"], "Days present": p,
                         "Total days": len(days),
                         "Attendance %": round(p / len(days) * 100, 1) if days else 0.0})
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"Attendance %": st.column_config.ProgressColumn(
                         "Attendance %", min_value=0, max_value=100, format="%.1f%%")})
        low = df[(df["Attendance %"] < 75) & (df["Total days"] > 0)]
        if len(low):
            st.warning("Below 75%: " + ", ".join(low["Name"]))
        st.markdown("---")
        target = st.selectbox("Remove a student", [f"{s['id']} — {s['name']}" for s in students])
        wipe = st.checkbox("Also delete this student's past attendance records",
                           help="Off = keep their history as an audit record (recommended for real use).")
        if st.button("Remove", type="secondary"):
            rid = target.split(" — ")[0]
            if wipe:
                delete_attendance(student_id=rid)
            keep = [s for s in students if s["id"] != rid]
            pd.DataFrame(keep, columns=["id", "name"]).to_csv(STUDENTS_CSV, index=False)
            emb = os.path.join(EMB_DIR, f"{rid}.npy")
            if os.path.exists(emb):
                os.remove(emb)
            st.success(f"Removed {target}")
            st.rerun()

# ---------------- Records ----------------
elif page == "Records & Reports":
    hero("Records & Reports", "Day-wise records and the full attendance register")
    students, days = load_register()
    if not days:
        st.info("No attendance recorded yet.")
    else:
        d = st.selectbox("Date", list(reversed(list(days))))
        st.dataframe(pd.read_csv(os.path.join(ATTENDANCE_DIR, f"{d}.csv")),
                     width="stretch", hide_index=True)
        with st.expander("🗑️ Delete records"):
            sure = st.checkbox(f"Yes, permanently delete all attendance for {d}")
            if st.button(f"Delete {d}", disabled=not sure):
                n = delete_attendance(day=d)
                st.success(f"Deleted {n} record(s) for {d}.")
                st.rerun()
        st.subheader("Full register")
        if not students:
            st.info("No students are registered right now, so the full register is empty. "
                    "Past day-wise records above are kept.")
        else:
            grid = pd.DataFrame([{"Reg. No": s["id"], "Name": s["name"],
                                  **{day: ("P" if s["id"] in p else "A") for day, p in days.items()}}
                                 for s in students], columns=["Reg. No", "Name", *days])
            st.dataframe(grid.style.map(lambda v: "background-color:rgba(34,197,94,0.14);color:#22c55e;font-weight:600" if v == "P" else
                                        ("background-color:rgba(239,68,68,0.14);color:#ef4444;font-weight:600" if v == "A" else ""),
                                        subset=list(days)),
                         width="stretch", hide_index=True)
        path, _, _ = build_report()
        with open(path, "rb") as f:
            st.download_button("⬇️ Download Excel register", f, file_name="attendance_report.xlsx",
                               type="primary")

# ---------------- How it works ----------------
else:
    hero("How it works", "The computer vision pipeline behind the system")
    st.markdown("""
**Pipeline (per video frame)**

1. **Face detection — YuNet.** A lightweight CNN locates every face and 5 landmarks
   (eyes, nose tip, mouth corners).
2. **Alignment.** The landmarks drive a similarity transform that warps each face to a
   standard 112×112 crop, removing tilt and scale differences.
3. **Feature extraction — SFace.** A CNN trained with a margin-based loss maps the face to a
   128-dimensional unit vector (embedding).
4. **Matching.** Cosine similarity against each student's mean embedding. The best match is
   accepted only above the threshold; otherwise the face is *Unknown*.
5. **Temporal confirmation.** In live mode a student must be recognised in 5 consecutive
   frames before being marked — suppressing single-frame false positives.

6. **Liveness (anti-spoofing).** Head yaw is estimated from the landmarks as
   `(nose_x − eyes_mid_x) / eye_distance`. A real head turning left and right swings this
   past ±0.22; a flat photo cannot, because tilting it only compresses the whole image
   uniformly. The student must turn both ways within 10 seconds, in continuous view.

7. **Screen detection.** A YOLOX object detector (COCO) runs every 2nd frame. While a
   *cell phone* is visible anywhere in the frame, nobody can be marked. If a face lies
   inside a detected *cell phone, laptop or TV*, it is rejected as a spoof — this stops a
   friend holding up a phone with a photo **or a live video call**. Faces wider than 40% of
   the frame are told to move back, so a phone can't be held so close that its edges are hidden.

**Privacy:** only numeric embeddings are stored, never face photos.

**Limitations:** a large monitor with its edges out of view, or a high-quality projection,
can still fool an ordinary RGB webcam. Fully closing that gap needs depth/IR sensors
(like Face ID) or a physically supervised kiosk.
Accuracy also drops in poor lighting and with masks.
""")
