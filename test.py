import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import threading

PORT = 9090
SOURCE = f'finals_verification_vids/phase1-pistol-continuous/cam3_joey.mp4'   # change to "video.mp4" or your rtsp://... URL

# Simple frame buffer
frame = None

def capture_loop():
    global frame
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Could not open source:", SOURCE)
        return

    print("✅ Capture started...")
    while True:
        ok, img = cap.read()
        if not ok:
            print("⚠️ Failed to grab frame")
            break
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buf.tobytes()
        time.sleep(1/50)  # ~20 FPS

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404); self.end_headers(); return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        global frame
        while True:
            if frame is None:
                time.sleep(0.05)
                continue
            self.wfile.write(b"--frame\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            time.sleep(0.05)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    # Start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # Start MJPEG server
    server = ThreadedHTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    print(f"🌐 MJPEG server running at http://127.0.0.1:{PORT}/stream")
    server.serve_forever()
