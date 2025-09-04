import cv2
import numpy as np
import torch
import queue
import threading

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import firebase_admin
from firebase_admin import credentials, firestore

ACTIVE = False 
embeddings = []
embeddings_lock = threading.Lock()

def track_worker(q_track, cam_id, school, yolo_model, reid_model, reid_transform, q_display=None, cam_name=None):
    # Firebase init
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    school_ref = db.collection('schools').document(school)

    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Listen for "Active Event" flag
    def on_snapshot(docs, changes, ts):
        global ACTIVE
        ACTIVE = docs[0].to_dict().get('Active Event', False)

    school_ref.on_snapshot(on_snapshot)

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

    def match_embedding(embedding, embeddings, create_new):
        best_match = None
        max_sim = 0
        for entry in embeddings:
            sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
            if sim > 0.95 and sim > max_sim:
                max_sim = sim
                best_match = entry['id']
        if best_match:
            entry = next(e for e in embeddings if e['id'] == best_match)
            entry['embedding'] = ((np.array(entry['embedding']) * entry['count'] + embedding) / (entry['count'] + 1)).tolist()
            entry['count'] += 1
            return best_match, max_sim
        elif create_new:
            if len(embeddings) > 1000:
                embeddings.pop(0)
            new_id = len(embeddings) + 1
            with embeddings_lock:
                embeddings.append({
                    'embedding': embedding.tolist(),
                    'id': new_id,
                    'count': 1
                })
            return new_id, 1.0
        return None, 0.0

    cam_ref = school_ref.collection("cameras").document(cam_id)

    frame_count = 0
    try:
        while True:
            frame = q_track.get()  # blocks until a frame arrives
            
            frame_count += 1
            
            # Track all people in the frame for display (every frame for smooth GUI)
            person_boxes = yolo_model(frame, verbose=False)[0].boxes
            tracking_results = []  # Store tracking results for GUI
            
            # Store detection info for global coordination
            detection_info = {
                'cam_id': cam_id,
                'similarity': 0.0,
                'has_detection': False
            }
            
            # Process every other frame for embedding matching to reduce computational load
            if frame_count % 2 == 0:
                doc = school_ref.get().to_dict()
                detected_id = doc.get("detected_cam_id", "")

            # Only add tracking results for shooters (will be populated during embedding matching)
            
            # Do heavy embedding matching and Firebase coordination only every other frame
            if frame_count % 2 == 0 and detected_id != "":
                if detected_id == cam_id:
                    bbox = cam_ref.get().to_dict().get("bboxes", [0, 0, 0, 0])
                    if sum(bbox) != 0:
                        weapon_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        closest_person = None
                        min_distance = float('inf')

                        for person_box in person_boxes:
                            conf = float(person_box.conf.item())
                            cls = int(person_box.cls.item())
                            if cls == 0 and conf > 0.3:
                                x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                                person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                                distance = np.linalg.norm(np.array(weapon_center) - np.array(person_center))
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_person = (x1, y1, x2, y2, conf)

                        if closest_person is not None:
                            x1, y1, x2, y2, conf = closest_person
                            embedding = get_embedding(frame, (x1, y1, x2, y2))
                            if embedding is not None:
                                shooter_id, similarity = match_embedding(embedding, embeddings, create_new=True)
                                detection_info['similarity'] = similarity
                                detection_info['has_detection'] = True
                                
                                # Add shooter to tracking results
                                tracking_results.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'person_id': shooter_id,
                                    'is_shooter': True,
                                    'confidence': conf
                                })
                else:
                    if len(embeddings) > 0:
                        best_match = None
                        best_similarity = 0.0
                        best_person = None
                        
                        for person_box in person_boxes:
                            conf = float(person_box.conf.item())
                            cls = int(person_box.cls.item())
                            if cls == 0 and conf > 0.3:
                                x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                                embedding = get_embedding(frame, (x1, y1, x2, y2))
                                if embedding is not None:
                                    result = match_embedding(embedding, embeddings, create_new=False)
                                    if result[0] is not None:
                                        shooter_id, similarity = result
                                        if similarity > best_similarity:
                                            best_similarity = similarity
                                            best_match = shooter_id
                                            best_person = (x1, y1, x2, y2, conf)
                        
                        if best_match is not None:
                            detection_info['similarity'] = best_similarity
                            detection_info['has_detection'] = True
                            
                            # Add shooter to tracking results
                            x1, y1, x2, y2, conf = best_person
                            tracking_results.append({
                                'bbox': (x1, y1, x2, y2),
                                'person_id': best_match,
                                'is_shooter': False,  # Will be updated by global coordination
                                'confidence': conf
                            })
                
                # Update camera's detection info in Firebase for global coordination
                if detection_info['has_detection']:
                    cam_ref.update({
                        "detection_similarity": detection_info['similarity'],
                        "detection_timestamp": firestore.SERVER_TIMESTAMP
                    })
                
                # Global coordination: only allow best match to have shooter_detected = True
                cameras_ref = school_ref.collection("cameras")
                all_cameras = cameras_ref.stream()
                
                best_camera = None
                best_similarity = 0.0
                
                for camera_doc in all_cameras:
                    camera_data = camera_doc.to_dict()
                    cam_similarity = camera_data.get("detection_similarity", 0.0)
                    if cam_similarity > best_similarity:
                        best_similarity = cam_similarity
                        best_camera = camera_doc.id
                
                # Update shooter_detected for all cameras
                all_cameras_again = cameras_ref.stream()
                for camera_doc in all_cameras_again:
                    is_best = camera_doc.id == best_camera and best_similarity > 0.0
                    camera_doc.reference.update({"shooter_detected": is_best})
                
                # Update tracking results based on global coordination
                current_cam_is_best = cam_id == best_camera and best_similarity > 0.0
                for track in tracking_results:
                    # Only show as shooter if this camera has the best match
                    track['is_shooter'] = track['is_shooter'] and current_cam_is_best
            
            # Always check for shooter matches in current frame if we have embeddings
            # This ensures shooter tracking continues even after weapon detection stops
            with embeddings_lock:
                if len(tracking_results) == 0 and len(embeddings) > 0:
                    best_match = None
                    best_similarity = 0.0
                    best_person = None
                    
                    for person_box in person_boxes:
                        conf = float(person_box.conf.item())
                        cls = int(person_box.cls.item())
                        if cls == 0 and conf > 0.3:
                            x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                            embedding = get_embedding(frame, (x1, y1, x2, y2))
                            if embedding is not None:
                                result = match_embedding(embedding, embeddings, create_new=False)
                                if result[0] is not None:
                                    shooter_id, similarity = result
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = shooter_id
                                        best_person = (x1, y1, x2, y2, conf)
                    
                    if best_match is not None:
                        x1, y1, x2, y2, conf = best_person
                        
                        # Determine if this should show as shooter based on best similarity across all cameras
                        # For now, always show as shooter if we found a match - global coordination will handle it
                        is_shooter = True
                        tracking_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'person_id': best_match,
                            'is_shooter': is_shooter,
                            'confidence': conf
                        })
            
            # Draw both weapon detection boxes and tracking boxes on the current frame
            if q_display and cam_name:
                try:
                    # Create annotated frame
                    annotated_frame = frame.copy()
                    
                    # Draw weapon detection box if this camera detected a weapon
                    if frame_count % 2 == 0 and detected_id != "":
                        if detected_id == cam_id:
                            bbox = cam_ref.get().to_dict().get("bboxes", [0, 0, 0, 0])
                            if sum(bbox) != 0:
                                # Draw weapon detection box in blue
                                x1, y1, x2, y2 = map(int, bbox[:4])
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for weapon
                                cv2.putText(annotated_frame, "Weapon Detected", (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw tracking boxes only for shooters
                    shooter_count = 0
                    for track in tracking_results:
                        if track['person_id'] is not None and track.get('is_shooter', False):  # Only draw shooters
                            shooter_count += 1
                            bbox = track['bbox']
                            person_id = track['person_id']
                            confidence = track.get('confidence', 0.0)
                            
                            x1, y1, x2, y2 = bbox
                            
                            # Red for shooter
                            color = (0, 0, 255)
                            label = f"Shooter {person_id}"
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Draw confidence score
                            conf_text = f"{confidence:.2f}"
                            cv2.putText(annotated_frame, conf_text, (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Send the annotated frame with both weapon and tracking boxes to GUI
                    gui_cam_id = int(cam_name[-1])  # Extract number from "Camera X"
                    q_display.put((gui_cam_id, cam_name, annotated_frame), timeout=0.01)
                except queue.Full:
                    pass  # Skip if queue is full

    except KeyboardInterrupt:
        print(f"[{cam_id}] Shutting down.")

# for testing tracking only
# def test_reidentification(video_path, output_path=None):
#     """
#     Test re-identification functionality on a video file.
    
#     Args:
#         video_path (str): Path to the input video file
#         output_path (str, optional): Path to save output video with tracking
#     """
#     print(f"[TEST] Testing re-identification on video: {video_path}")
    
#     # Initialize device
#     device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#     print(f"[TEST] Using device: {device}")
    
#     # Initialize models
#     print("[TEST] Loading YOLO model...")
#     yolo = YOLO("yolov8n.pt")
#     yolo.fuse()
    
#     print("[TEST] Loading ReID model...")
#     reid_model = SimpleReIDModel(feature_dim=512)
#     reid_model.to(device).eval()
    
#     # Initialize transform
#     reid_transform = transforms.Compose([
#         transforms.Resize((256, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[ERROR] Could not open video: {video_path}")
#         return
    
#     # Get video properties
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"[TEST] Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
#     # Initialize video writer if output path is provided
#     writer = None
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#         print(f"[TEST] Output video will be saved to: {output_path}")
    
#     # Initialize tracking variables
#     person_embeddings = []  # Store person embeddings
#     person_tracks = {}  # Store person tracks
#     next_person_id = 1
    
#     def get_embedding(image, box):
#         """Extract embedding from person crop"""
#         x1, y1, x2, y2 = map(int, box)
#         crop = image[y1:y2, x1:x2]
#         if crop.size == 0 or x2 - x1 < 10 or y2 - y1 < 10:
#             return None
#         img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#         img = reid_transform(Image.fromarray(img)).unsqueeze(0).to(device)
#         with torch.no_grad():
#             feat = reid_model(img).cpu().numpy().flatten()
#         return feat / np.linalg.norm(feat)
    
#     def match_person(embedding, threshold=0.7):
#         """Match person embedding to existing tracks"""
#         if not person_embeddings:
#             return None
        
#         best_match = None
#         max_sim = 0
        
#         for i, stored_embedding in enumerate(person_embeddings):
#             sim = cosine_similarity([embedding], [stored_embedding])[0][0]
#             if sim > threshold and sim > max_sim:
#                 max_sim = sim
#                 best_match = i
        
#         return best_match
    
#     frame_count = 0
#     print("[TEST] Starting video processing...")
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
#             if frame_count % 30 == 0:  # Print progress every 30 frames
#                 print(f"[TEST] Processing frame {frame_count}/{total_frames}")
            
#             # Detect people using YOLO
#             results = yolo(frame, verbose=False)
#             person_boxes = results[0].boxes
            
#             current_frame_tracks = {}
            
#             for person_box in person_boxes:
#                 conf = float(person_box.conf.item())
#                 cls = int(person_box.cls.item())
                
#                 # Only process person detections with good confidence
#                 if cls == 0 and conf > 0.3:  # class 0 is person in COCO
#                     x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                    
#                     # Extract embedding
#                     embedding = get_embedding(frame, (x1, y1, x2, y2))
#                     if embedding is None:
#                         continue
                    
#                     # Try to match with existing tracks
#                     matched_id = match_person(embedding)
                    
#                     if matched_id is not None:
#                         # Update existing track
#                         person_id = matched_id + 1
#                         person_embeddings[matched_id] = embedding  # Update embedding
#                         current_frame_tracks[person_id] = (x1, y1, x2, y2)
#                         print(f"[TEST] Frame {frame_count}: Person {person_id} re-identified (confidence: {conf:.2f})")
#                     else:
#                         # Create new track
#                         person_id = next_person_id
#                         person_embeddings.append(embedding)
#                         current_frame_tracks[person_id] = (x1, y1, x2, y2)
#                         next_person_id += 1
#                         print(f"[TEST] Frame {frame_count}: New person {person_id} detected (confidence: {conf:.2f})")
                    
#                     # Draw bounding box and ID
#                     color = (0, 255, 0) if matched_id is not None else (0, 0, 255)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f"Person {person_id}", (x1, y1-10), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # Update global tracks
#             person_tracks[frame_count] = current_frame_tracks
            
#             # Write frame to output video
#             if writer:
#                 writer.write(frame)
            
#             # Display frame (optional - uncomment to see real-time processing)
#             cv2.imshow('Re-identification Test', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     except KeyboardInterrupt:
#         print("\n[TEST] Processing interrupted by user")
    
#     finally:
#         cap.release()
#         if writer:
#             writer.release()
#         cv2.destroyAllWindows()
        
#         # Print summary
#         print(f"\n[TEST] Processing complete!")
#         print(f"[TEST] Total frames processed: {frame_count}")
#         print(f"[TEST] Total unique persons tracked: {len(person_embeddings)}")
#         print(f"[TEST] Frames with detections: {len([f for f in person_tracks.values() if f])}")
        
#         if output_path:
#             print(f"[TEST] Output video saved to: {output_path}")


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python track.py <video_path> [output_path]")
#         print("Example: python track.py test_video.mp4 output_tracked.mp4")
#         sys.exit(1)
    
#     video_path = sys.argv[1]
#     output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
#     test_reidentification(video_path, output_path)