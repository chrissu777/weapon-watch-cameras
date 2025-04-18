import cv2
import numpy as np
import threading
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from byte_tracker import BYTETracker  # your local file
from sklearn.metrics.pairwise import cosine_similarity
import torchreid

# Load ReID model (OSNet)
reid_model = torchreid.models.build_model(
    name='osnet_ain_x1_0', 
    num_classes=1000, 
    loss='softmax', 
    pretrained=True
)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
reid_model.to(device)
reid_model.eval()

# ReID image transform
reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global person identity database
global_db = []
global_id_counter = 0
db_lock = threading.Lock()

def generate_new_global_id():
    global global_id_counter
    global_id_counter += 1
    return global_id_counter

def get_embedding(image, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or x2 - x1 < 10 or y2 - y1 < 10:
        return None
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = reid_transform(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img).cpu().numpy().flatten()
    return feat / np.linalg.norm(feat)

def match_embedding(embedding, cam_id):
    with db_lock:
        best_match = None
        max_sim = 0
        for entry in global_db:
            if entry['cam_id'] == cam_id:
                continue
            sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
            if sim > 0.7 and sim > max_sim:
                max_sim = sim
                best_match = entry['global_id']
        return best_match if best_match else generate_new_global_id()

def process_camera(source, cam_id):
    print(f"Starting camera {cam_id}")
    cap = cv2.VideoCapture(source)
    model = YOLO("yolov8n.pt")
    tracker = BYTETracker(type('Args', (), {
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'frame_rate': 30,
        'use_byte': True,
        'mot20': False
    })())

    local_to_global = {}

    frame_count = 0
    max_frames = 1000  # Limit for testing

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'./results/camera_{cam_id}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened() and frame_count < max_frames:
        frame_count += 1
        # print(f"Processing frame {frame_count} from camera {cam_id}")
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.item())
            cls = int(box.cls.item())
            if cls == 0 and conf > 0.3:
                detections.append([x1, y1, x2, y2, conf])

        dets = np.array(detections)
        online_targets = tracker.update(dets, frame.shape[:2], frame.shape[:2])

        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id

            if track_id not in local_to_global:
                embedding = get_embedding(frame, (x1, y1, x2, y2))
                if embedding is not None:
                    global_id = match_embedding(embedding, cam_id)
                    local_to_global[track_id] = global_id
                    with db_lock:
                        global_db.append({
                            'embedding': embedding,
                            'cam_id': cam_id,
                            'global_id': global_id
                        })

            global_id = local_to_global.get(track_id, -1)
            
            if global_id % 5 == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"GID: {global_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # cv2.imshow(f"Camera {cam_id}", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        

        # Inside the while loop, after drawing on the frame:
        out.write(frame)

    cap.release()
    out.release()

# Define camera sources
camera_sources = [
    "./videos/Wildtrack/cam1.mp4",
    "./videos/Wildtrack/cam2.mp4",
    "./videos/Wildtrack/cam3.mp4",
    "./videos/Wildtrack/cam4.mp4"
]

# Launch threads for each camera
threads = []
for i, source in enumerate(camera_sources):
    t = threading.Thread(target=process_camera, args=(source, i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
