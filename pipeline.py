import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from dotenv import load_dotenv
import os
from inference import get_model   # ✅ Roboflow Inference SDK

###############################################################################
# Config
###############################################################################
PORT = 9090
SOURCES = [
    # f'finals_verification_vids/phase1-pistol-continuous/cam3_joey.mp4',
    # f'finals_verification_vids/phase1-pistol-continuous/cam4_joey.mp4',
    # f'finals_verification_vids/phase1-pistol-continuous/cam5_joey.mp4',
    # "rtsp://192.168.1.111:554/profile2/media.smp",
    # "rtsp://192.168.1.151:554/profile2/media.smp",
    # "rtsp://192.168.1.114:554/profile2/media.smp",
]

# Load your Roboflow model once
# Replace with your workspace/project/version
print("🔄 Loading Roboflow model...")
load_dotenv()
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
model = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY)  
print("✅ Model ready")

CONF_TH = 0.6
IOU_TH = 0.8

###############################################################################
# Utilities
###############################################################################
def draw_boxes(frame, preds):
    for p in preds:
        # convert from center-based coords → top-left/bottom-right
        x1 = int(p.x - p.width/2)
        y1 = int(p.y - p.height/2)
        x2 = int(p.x + p.width/2)
        y2 = int(p.y + p.height/2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (38, 14, 194), 2)
        cv2.putText(frame, f"{p.class_name}:{p.confidence:.2f}",
                    (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (38, 14, 194), 2)
    return frame


###############################################################################
# Per-camera capture loop
###############################################################################
frames = {}   # dictionary of {source_id: latest jpeg bytes}

def capture_loop(src, cam_id):
    global frames
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ Could not open {src}")
        return

    print(f"✅ Capture started for {src}")

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (640, 640))
        results = model.infer(img, confidence=CONF_TH, iou_threshold=IOU_TH)

        preds = results[0].predictions if results else []
        if preds:
            img = draw_boxes(img, preds)

        # Encode for MJPEG
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frames[cam_id] = buf.tobytes()

        # time.sleep(1/15)  # ~15 FPS

###############################################################################
# MJPEG Server
###############################################################################
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/stream"):
            self.send_response(404); self.end_headers(); return

        try:
            cam_id = int(self.path.split("/")[-1])
        except:
            cam_id = 0

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        global frames
        while True:
            if cam_id not in frames:
                time.sleep(0.05)
                continue
            frame = frames[cam_id]
            self.wfile.write(b"--frame\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            time.sleep(0.05)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Start capture threads
    for cam_id, src in enumerate(SOURCES):
        t = threading.Thread(target=capture_loop, args=(src, cam_id), daemon=True)
        t.start()

    # Start server
    server = ThreadedHTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    print(f"🌐 Open VLC at http://127.0.0.1:{PORT}/stream/0 (or /stream/1 …)")
    server.serve_forever()
