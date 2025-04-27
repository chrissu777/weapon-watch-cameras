import cv2
import numpy as np
import threading
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import torchreid

# Load ReID model (OSNet with IBN for better performance)
reid_model = torchreid.models.build_model(
    name='osnet_ibn_x1_0',  # Better-performing ReID model
    num_classes=1000,
    loss='softmax',
    pretrained=True
)
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
reid_model.to(device)
reid_model.eval()

# ReID image transform
reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),  # Standard input size for ReID models
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
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

def match_embedding(embedding):
    with db_lock:
        try:
            best_match = None
            max_sim = 0
            for entry in global_db:
                sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
                if sim > 0.7 and sim > max_sim:
                    max_sim = sim
                    best_match = entry['global_id']
            if best_match:
                entry = next(e for e in global_db if e['global_id'] == best_match)
                entry['embedding'] = (entry['embedding'] * entry['count'] + embedding) / (entry['count'] + 1)
                entry['count'] += 1
                return best_match
            else:
                global_id = generate_new_global_id()
                global_db.append({
                    'embedding': embedding,  # Replace this with the moving average
                    'global_id': global_id,
                    'count': 1  # Track the number of embeddings averaged
                })
                # Limit the size of global_db
                if len(global_db) > 1000:  # Example limit
                    global_db.pop(0)
                return global_id
        finally:
            pass

def process_camera(source, cam_id):
    try:
        print(f"Starting camera {cam_id}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        model = YOLO("yolov8n.pt")

        frame_count = 0
        max_frames = 500  # Limit for testing

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'./results/camera_{cam_id}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened() and frame_count < max_frames:
            print(f"Processing frame {frame_count} from camera {cam_id}")
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {frame_count} from camera {cam_id}")
                break

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert to integers
                conf = float(box.conf.item())
                cls = int(box.cls.item())
                if cls == 0 and conf > 0.3:  # Class 0 is "person"
                    embedding = get_embedding(frame, (x1, y1, x2, y2))
                    if embedding is not None:
                        global_id = match_embedding(embedding)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
                        cv2.putText(frame, f"GID: {global_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Add text
            out.write(frame)

    except Exception as e:
        print(f"Error in camera {cam_id}: {e}")
    finally:
        print(f"Releasing resources for camera {cam_id}")
        cap.release()
        out.release()

# Define camera sources
camera_sources = [
    "./videos/running1.mp4",
    "./videos/running2.mp4",
]

# Launch threads for each camera
threads = []
for i, source in enumerate(camera_sources):
    t = threading.Thread(target=process_camera, args=(source, i))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

# process_camera("./videos/angle1.mp4", 0)