import cv2
import time
import threading
import numpy as np
import tensorflow as tf
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

###############################################################################
# Config
###############################################################################
PORT = 9090
SOURCES = [
    f'finals_verification_vids/phase1-pistol-continuous/cam3_joey.mp4',
    f'finals_verification_vids/phase1-pistol-continuous/cam4_joey.mp4',
    f'finals_verification_vids/phase1-pistol-continuous/cam5_joey.mp4',
    # "rtsp://user:pass@ip:554/stream"
]

SAVED_MODEL_DIR = "detectionmodel"  # path to your RF-DETR export
INFER_W, INFER_H = 608, 608
CONF_TH = 0.35
IOU_TH = 0.5
MAX_BOXES = 50

###############################################################################
# Global shared model (loaded once)
###############################################################################
print("🔄 Loading model once...")
model = tf.saved_model.load(SAVED_MODEL_DIR)
infer = model.signatures["serving_default"]
print("✅ Model ready")

###############################################################################
# Utilities
###############################################################################
def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (INFER_W, INFER_H)).astype(np.float32) / 255.0
    img = img[np.newaxis, ...]
    return img

def postprocess_tf(pred_dict, orig_w, orig_h):
    value = next(iter(pred_dict.values()))
    boxes = value[:, :, 0:4]
    confs = value[:, :, 4:]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    yxxy = tf.concat([y1, x1, y2, x2], axis=-1)

    nms_boxes, nms_scores, nms_classes, valid = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(yxxy, (tf.shape(yxxy)[0], -1, 1, 4)),
        scores=tf.reshape(confs, (tf.shape(confs)[0], -1, tf.shape(confs)[-1])),
        max_output_size_per_class=MAX_BOXES,
        max_total_size=MAX_BOXES,
        iou_threshold=IOU_TH,
        score_threshold=CONF_TH
    )

    y1n, x1n, y2n, x2n = tf.split(nms_boxes[0], 4, axis=-1)
    boxes_xyxy = np.stack([
        (x1n.numpy() * orig_w).flatten(),
        (y1n.numpy() * orig_h).flatten(),
        (x2n.numpy() * orig_w).flatten(),
        (y2n.numpy() * orig_h).flatten()
    ], axis=-1)

    scores = nms_scores[0].numpy()
    classes = nms_classes[0].numpy()
    valid_n = int(valid[0].numpy())

    return boxes_xyxy[:valid_n], scores[:valid_n], classes[:valid_n]

def draw_boxes(frame, boxes, scores, classes, class_names=["gun"]):
    for (x1,y1,x2,y2), sc, cl in zip(boxes, scores, classes):
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 2)
        cv2.putText(frame, f"{class_names[int(cl)]}:{sc:.2f}",
                    (int(x1), max(0,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,255), 2)
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
            print(f"⚠️ Failed to grab from {src}")
            break

        # Run detection
        inp = preprocess(img)
        preds = infer(tf.constant(inp, dtype=tf.float32))
        boxes, scores, classes = postprocess_tf(preds, img.shape[1], img.shape[0])

        if len(boxes) > 0:
            img = draw_boxes(img, boxes, scores, classes)

        # Encode for MJPEG
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frames[cam_id] = buf.tobytes()

###############################################################################
# MJPEG Server
###############################################################################
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/stream"):
            self.send_response(404); self.end_headers(); return

        # Extract camera id: /stream/0, /stream/1, etc.
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
