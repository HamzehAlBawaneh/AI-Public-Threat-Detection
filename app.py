import cv2
import numpy as np
import pickle
import os
import sqlite3
import time
import threading
import face_recognition
import pandas as pd

from flask import Flask, render_template, Response, jsonify, send_file
from ultralytics import YOLO
from datetime import datetime

# ==========================
# Paths  (keep original)
# ==========================

BASE_DIR = r"C:\Users\ASUS\OneDrive\Desktop\v2mohannad\projectDM_final\projectDM_final"

DB_PATH         = os.path.join(BASE_DIR, "incidents.db")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.pkl")
FACE_DB         = os.path.join(BASE_DIR, "face_database")
RECORDS_DIR     = os.path.join(BASE_DIR, "records")
CSV_PATH        = os.path.join(BASE_DIR, "incidents.csv")

os.makedirs(FACE_DB,     exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)

# ==========================
# Flask App
# ==========================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# ==========================
# Load Models ONCE at Startup
# ONNX format for faster CPU inference (~2x vs .pt)
# ==========================

weapon_model = YOLO(
    os.path.join(BASE_DIR, "best.onnx"),
    task="detect"
)

person_model = YOLO(
    os.path.join(BASE_DIR, "yolov8n (1).onnx"),
    task="detect"
)

# ==========================
# CV Thresholding Constants
# ==========================

# minimum confidence score to accept a weapon detection
# below this value the detection is discarded as unreliable
WEAPON_CONF_THRESHOLD = 0.60

# cosine similarity threshold for face matching
# 1.0 = identical faces, 0.0 = completely different
# 0.75 means faces must share 75% directional similarity in embedding space
FACE_SIMILARITY_THRESHOLD = 0.75

# mean pixel brightness below which CLAHE preprocessing is applied
# range 0-255; 85 catches typical indoor low-light conditions
LOW_LIGHT_THRESHOLD = 85

# NMS IoU threshold: boxes overlapping more than this fraction are merged
# lower = stricter merging (fewer duplicate boxes)
NMS_IOU_THRESHOLD = 0.45

# minimum person bounding box area in pixels to attempt face recognition
# boxes smaller than this mean the person is too far for a reliable embedding
MIN_PERSON_BOX_AREA = 3000

# run face recognition every N frames to reduce latency
# face_recognition library is CPU-heavy so we throttle it
FACE_RECOGNITION_EVERY_N_FRAMES = 3

# ==========================
# Shared State (thread-safe)
# ==========================

latest_detection = {
    "weapon":        "None",
    "person":        "Unknown",
    "confidence":    0,
    "threat_level":  "NONE",
    "preprocessing": "normal"
}

last_detection_time = {}
state_lock          = threading.Lock()

# ==========================
# Database
# ==========================

def create_database():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            weapon_type TEXT,
            confidence  REAL,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()

create_database()

if not os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({}, f)

def save_incident(person, weapon, confidence):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO incidents (person_name, weapon_type, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        person,
        weapon,
        round(confidence, 2),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# ==========================
# CV: Threat Level
# ==========================

def calculate_threat_level(weapon, person):
    if weapon != "None" and person != "Unknown":
        return "HIGH"
    elif weapon != "None":
        return "MEDIUM"
    elif person != "Unknown":
        return "LOW"
    else:
        return "NONE"

# ==========================
# CV: Adaptive Preprocessing
# CLAHE histogram equalization for low-light surveillance frames
# Applied only when mean brightness falls below LOW_LIGHT_THRESHOLD
# CLAHE works on local image tiles unlike global equalization
# so it avoids over-brightening uniform bright regions
# ==========================

def preprocess_frame(frame):
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    mode            = "normal"

    if mean_brightness < LOW_LIGHT_THRESHOLD:
        clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv          = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        frame        = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        mode         = "low-light enhanced"

    return frame, mode

# ==========================
# CV: Face Recognition
# Cosine similarity on 128-dim face embeddings
# ==========================

def cosine_similarity_manual(a, b):
    a      = np.array(a)
    b      = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def get_or_create_person(face_crop):
    rgb_face  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)

    if len(encodings) == 0:
        return None

    new_embedding = encodings[0]

    with open(EMBEDDINGS_PATH, "rb") as f:
        db = pickle.load(f)

    best_match = None
    best_score = 0.0

    for person_name, stored_embedding in db.items():
        score = cosine_similarity_manual(new_embedding, stored_embedding)
        if score > best_score:
            best_score = score
            best_match = person_name

    if best_score > FACE_SIMILARITY_THRESHOLD:
        return best_match

    person_id   = len(db) + 1
    person_name = "Person_" + str(person_id)
    db[person_name] = new_embedding.tolist()

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(db, f)

    os.makedirs(os.path.join(FACE_DB, person_name), exist_ok=True)
    return person_name

def save_face_image(face_crop, person_name):
    folder   = os.path.join(FACE_DB, person_name)
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
    if existing >= 4:
        return
    cv2.imwrite(os.path.join(folder, "frame" + str(existing + 1) + ".jpg"), face_crop)

# ==========================
# Camera — opened once, buffer minimized
# ==========================

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE,   1)

print("Camera opened:", camera.isOpened())

# ==========================
# CV Frame Generator
# Pipeline per frame:
#   1. read frame
#   2. adaptive CLAHE preprocessing (thresholding on brightness)
#   3. weapon detection  — ONNX, conf+iou thresholds, best detection only
#   4. person detection  — ONNX, class filter (person only)
#   5. face recognition  — throttled every N frames, min box area guard
#   6. threat level calculation
#   7. annotate frame, update shared state, yield JPEG
# ==========================

frame_counter = 0

def generate_frames():
    global frame_counter

    while True:
        success, frame = camera.read()

        if not success:
            time.sleep(0.05)
            continue

        frame_counter += 1

        # step 1: adaptive preprocessing
        processed_frame, preprocess_mode = preprocess_frame(frame)

        # step 2: weapon detection (ONNX, confidence threshold)
        weapon_results = weapon_model.predict(
            source=processed_frame,
            conf=WEAPON_CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
            device="cpu"
        )   

        weapon_detected  = False
        weapon_name      = "None"
        weapon_conf      = 0.0
        weapon_center_x  = 0
        weapon_center_y  = 0
        best_conf        = 0.0

        for result in weapon_results:
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                if conf <= best_conf:
                    continue

                best_conf       = conf
                weapon_detected = True
                weapon_name     = weapon_model.names[cls]
                weapon_conf     = conf

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                weapon_center_x = (x1 + x2) // 2
                weapon_center_y = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    weapon_name + " " + str(round(conf, 2)),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

        # step 3: person detection (only when weapon found)
        person_name        = "Unknown"
        closest_person_box = None

        if weapon_detected:
            person_results = person_model.predict(
                source=processed_frame,
                conf=0.50,
                verbose=False,
                classes=[0],
                device="cpu"
            )

            min_distance = float("inf")

            for person_result in person_results:
                for person_box in person_result.boxes:
                    px1, py1, px2, py2 = map(int, person_box.xyxy[0])
                    box_area = (px2 - px1) * (py2 - py1)

                    if box_area < MIN_PERSON_BOX_AREA:
                        continue

                    pcx = (px1 + px2) // 2
                    pcy = (py1 + py2) // 2
                    dist = (weapon_center_x - pcx) ** 2 + (weapon_center_y - pcy) ** 2

                    if dist < min_distance:
                        min_distance       = dist
                        closest_person_box = (px1, py1, px2, py2)

            # step 4: face recognition (throttled)
            if closest_person_box is not None and (frame_counter % FACE_RECOGNITION_EVERY_N_FRAMES == 0):
                px1, py1, px2, py2 = closest_person_box
                person_crop        = frame[py1:py2, px1:px2]
                face_locations     = face_recognition.face_locations(person_crop)

                if len(face_locations) > 0:
                    top, right, bottom, left = face_locations[0]

                    cv2.rectangle(
                        frame,
                        (px1 + left,  py1 + top),
                        (px1 + right, py1 + bottom),
                        (255, 0, 0), 2
                    )

                    face_crop   = person_crop[top:bottom, left:right]
                    person_name = get_or_create_person(face_crop)

                    if person_name is None:
                        person_name = "Unknown"
                    else:
                        save_face_image(face_crop, person_name)

                    cv2.putText(
                        frame,
                        person_name,
                        (px1 + left, py1 + top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2
                    )

        # step 5: threat level
        threat_level  = calculate_threat_level(weapon_name, person_name)

        threat_colors = {
            "HIGH":   (0, 0, 255),
            "MEDIUM": (0, 140, 255),
            "LOW":    (0, 255, 255),
            "NONE":   (0, 255, 0)
        }

        cv2.putText(
            frame,
            "threat: " + threat_level,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, threat_colors.get(threat_level, (0, 255, 0)), 2
        )

        cv2.putText(
            frame,
            "pre: " + preprocess_mode,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (200, 200, 200), 1
        )

        # step 6: save incident (5 second cooldown per key)
        if weapon_detected:
            key          = person_name + "_" + weapon_name
            current_time = time.time()
            if key not in last_detection_time or (current_time - last_detection_time[key]) >= 5:
                last_detection_time[key] = current_time
                save_incident(person_name, weapon_name, weapon_conf)

        # update shared state
        with state_lock:
            latest_detection["weapon"]        = weapon_name
            latest_detection["person"]        = person_name
            latest_detection["confidence"]    = round(weapon_conf, 2)
            latest_detection["threat_level"]  = threat_level
            latest_detection["preprocessing"] = preprocess_mode

        # encode at 85% JPEG — good quality, reduced payload = lower latency
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/latest_detection")
def latest():
    with state_lock:
        data = dict(latest_detection)
    return jsonify(data)

@app.route("/records")
def records():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents ORDER BY id DESC")
    rows   = cursor.fetchall()
    conn.close()

    html = (
        "<html><head><style>"
        "body{font-family:Arial;background:#0f172a;color:white;padding:20px}"
        "h2{color:#38bdf8}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #334155;padding:10px;text-align:left}"
        "th{background:#1e293b;color:#38bdf8}"
        "tr:nth-child(even){background:#1e293b}"
        ".HIGH{color:#ef4444;font-weight:bold}"
        ".MEDIUM{color:#f97316;font-weight:bold}"
        ".LOW{color:#facc15;font-weight:bold}"
        "</style></head><body>"
        "<h2>incident records</h2>"
        "<table><tr><th>id</th><th>person</th><th>weapon</th><th>confidence</th><th>timestamp</th></tr>"
    )

    for row in rows:
        threat = calculate_threat_level(row[2], row[1])
        html  += (
            "<tr>"
            "<td>" + str(row[0]) + "</td>"
            "<td>" + str(row[1]) + "</td>"
            "<td class='" + threat + "'>" + str(row[2]) + "</td>"
            "<td>" + str(row[3]) + "</td>"
            "<td>" + str(row[4]) + "</td>"
            "</tr>"
        )

    html += "</table></body></html>"
    return html

@app.route("/export_csv")
def export_csv():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("SELECT * FROM incidents", conn)
    conn.close()
    df.to_csv(CSV_PATH, index=False)
    return (
        "<html><body style='font-family:Arial;background:#0f172a;color:white;padding:20px'>"
        "<h2>csv exported</h2>"
        "<p>saved to: " + CSV_PATH + "</p>"
        "</body></html>"
    )

@app.route("/faces")
def faces():
    html = (
        "<html><head><style>"
        "body{font-family:Arial;background:#0f172a;color:white;padding:20px}"
        "h1,h2{color:#38bdf8}"
        "img{margin:10px;border:2px solid #334155;border-radius:8px}"
        "</style></head><body><h1>face database</h1>"
    )

    if not os.path.exists(FACE_DB) or not os.listdir(FACE_DB):
        return html + "<p>no faces saved yet</p></body></html>"

    for person in os.listdir(FACE_DB):
        person_folder = os.path.join(FACE_DB, person)
        if not os.path.isdir(person_folder):
            continue
        html += "<h2>" + person + "</h2>"
        for img in os.listdir(person_folder):
            html += "<img src='/face_image/" + person + "/" + img + "' width='150'>"

    html += "</body></html>"
    return html

@app.route("/face_image/<person>/<image>")
def face_image(person, image):
    return send_file(os.path.join(FACE_DB, person, image))

# ==========================
# Run
# ==========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )
