from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import threading
import time
import pandas as pd
import os
from collections import deque
from ultralytics import YOLO
import datetime
import json
from threading import Lock

app = Flask(__name__)


# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)
print("Torch Version:", torch.__version__)
model = YOLO("yolov8n.pt").to(device)

video_sources = ["testvideo6.mp4", "testvideo5.mp4", "testvideo4.mp4", "testvideo3.mp4"]
caps = [cv2.VideoCapture(src) for src in video_sources]
for cap in caps:
    if not cap.isOpened():
        print(f"Error: Unable to open video source.")

vehicle_counts = {"road1": 0, "road2": 0, "road3": 0, "road4": 0}
current_green_road = "road1"
signal_timers = {"road1": 10, "road2": 10, "road3": 10, "road4": 10}
remaining_time = signal_timers[current_green_road]

EXCEL_FILE = "traffic_data.xlsx"
file_lock = Lock()

if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Road", "Vehicle Count", "Green Light Duration"])
    with file_lock:
        df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")

def detect_vehicles(frame):
    """Detect vehicles using YOLOv8 and return the count."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    count = sum(1 for obj in results[0].boxes.cls if obj in [2, 3, 5, 7])
    return count

def generate_frames(video_index, road):
    cap = caps[video_index]
    while True:
        success, frame = cap.read()
        if not success:
            print(f"[{road}] Failed to read frame. Restarting video...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        print(f"[{road}] Frame read successfully")
        count = detect_vehicles(frame)
        vehicle_counts[road] = count
        
        cv2.putText(frame, f"Vehicles: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/video_feed/<road>')
def video_feed(road):
    road_index = {"road1": 0, "road2": 1, "road3": 2, "road4": 3}
    return Response(generate_frames(road_index[road], road),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_vehicle_counts')
def get_vehicle_counts():
    return jsonify(vehicle_counts)

@app.route('/get_signal_status')
def get_signal_status():
    return jsonify({
        "green_road": current_green_road,
        "duration": remaining_time
    })

def log_traffic_data(road, count, duration):
    """Logs traffic data to Excel."""
    with file_lock:
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        new_entry = {"Timestamp": datetime.datetime.now(), "Road": road, "Vehicle Count": count, "Green Light Duration": duration}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")

def control_traffic_signals():
    global current_green_road, remaining_time
    roads = deque(["road1", "road2", "road3", "road4"])
    while True:
        current_green_road = roads[0]
        count = vehicle_counts[current_green_road]
        green_time = max(10, min(60, 10 + (count // 5) * 2))
        signal_timers[current_green_road] = green_time
        remaining_time = green_time
        
        log_traffic_data(current_green_road, count, green_time)
        
        while remaining_time > 0:
            time.sleep(1)
            remaining_time -= 1
        roads.rotate(-1)

@app.route("/get_traffic_metrics")
def get_traffic_metrics():
    with file_lock:
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today_data = df[df["Timestamp"].astype(str).str.startswith(today)]

    total_vehicles_today = today_data["Vehicle Count"].sum()
    avg_waiting_time = today_data["Green Light Duration"].mean() if not today_data.empty else 0
    peak_hour = today_data.groupby(today_data["Timestamp"].astype(str).str[11:13])["Vehicle Count"].sum().idxmax() if not today_data.empty else "N/A"
    total_vehicles_recorded = df["Vehicle Count"].sum()

    return jsonify({
        "total_vehicles_today": int(total_vehicles_today),
        "avg_waiting_time": float(avg_waiting_time),
        "peak_hour": str(peak_hour),
        "total_vehicles_recorded": int(total_vehicles_recorded)
    })

@app.route("/get_peak_hour_trends")
def get_peak_hour_trends():
    filter_type = request.args.get("filter", "day")
    with file_lock:
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    if filter_type == "day":
        df["Hour"] = df["Timestamp"].dt.hour
        grouped = df.groupby("Hour")["Vehicle Count"].sum()
    elif filter_type == "week":
        df["Day"] = df["Timestamp"].dt.day
        grouped = df.groupby("Day")["Vehicle Count"].sum()
    else:
        df["Week"] = df["Timestamp"].dt.isocalendar().week
        grouped = df.groupby("Week")["Vehicle Count"].sum()

    return jsonify({"labels": grouped.index.tolist(), "values": grouped.values.tolist()})

threading.Thread(target=control_traffic_signals, daemon=True).start()

# 2 may 2025
@app.route("/get_road_metrics")
def get_road_metrics():
    period = request.args.get("period", "today")

    with file_lock:
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    now = datetime.datetime.now()

    if period == "today":
        df = df[df["Timestamp"].dt.date == now.date()]
    elif period == "week":
        df = df[df["Timestamp"].dt.isocalendar().week == now.isocalendar().week]
    elif period == "month":
        df = df[df["Timestamp"].dt.month == now.month]

    metrics = {}
    for road in ["road1", "road2", "road3", "road4"]:
        road_df = df[df["Road"] == road]
        total_vehicles = road_df["Vehicle Count"].sum()
        avg_green_time = road_df["Green Light Duration"].mean() if not road_df.empty else 0
        expected_rate = 1.5  # benchmark vehicles per sec
        efficiency = ((total_vehicles / avg_green_time) / expected_rate) * 100 if avg_green_time else 0
        efficiency = min(efficiency, 100)
        # efficiency = (total_vehicles / avg_green_time) * 2 if avg_green_time else 0  # basic efficiency formula

        metrics[road] = {
            "vehicle_count": int(total_vehicles),
            "avg_green_time": round(avg_green_time, 1),
            "efficiency": round(min(efficiency, 90)) 
        }

    return jsonify(metrics)


if __name__ == '__main__':
    app.run(debug=True) 