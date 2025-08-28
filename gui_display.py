import cv2
import queue
import threading
import numpy as np
from collections import defaultdict
import time

class MultiCameraDisplay:
    def __init__(self, num_cameras=6):
        self.num_cameras = num_cameras
        self.latest_frames = {}
        self.tracking_results = {}  # Store tracking results for each camera
        self.frame_lock = threading.Lock()
        self.window_name = "Weapon Detection - Multi Camera View"
        
    def update_frame(self, cam_id, cam_name, frame):
        """Update the latest frame for a specific camera"""
        with self.frame_lock:
            self.latest_frames[cam_id] = (cam_name, frame)
    
    def update_tracking(self, cam_id, tracking_results):
        """Update tracking results for a specific camera"""
        with self.frame_lock:
            self.tracking_results[cam_id] = tracking_results
    
    def draw_tracking_boxes(self, frame, tracking_results):
        """Draw tracking boxes and IDs on a frame"""
        if not tracking_results:
            return frame
        
        for track in tracking_results:
            bbox = track['bbox']
            person_id = track['person_id']
            is_shooter = track.get('is_shooter', False)
            confidence = track.get('confidence', 0.0)
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on whether it's a shooter or regular person
            if is_shooter:
                color = (0, 0, 255)  # Red for shooter
                label = f"Shooter {person_id}"
            else:
                color = (0, 255, 0)  # Green for regular person
                label = f"Person {person_id}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence score
            conf_text = f"{confidence:.2f}"
            cv2.putText(frame, conf_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def create_grid_display(self):
        """Create a 2x3 grid display for 6 cameras"""
        # Grid configuration for 6 cameras (2 rows, 3 columns)
        grid_h, grid_w = 2, 3
        cell_h, cell_w = 240, 320  # Resize each camera view
        
        # Create empty grid
        grid_image = np.zeros((grid_h * cell_h, grid_w * cell_w, 3), dtype=np.uint8)
        
        with self.frame_lock:
            for i in range(1, self.num_cameras + 1):  # Cameras are numbered 1-6
                row = (i - 1) // grid_w
                col = (i - 1) % grid_w
                
                if i in self.latest_frames:
                    cam_name, frame = self.latest_frames[i]
                    
                    # Draw tracking boxes if available
                    if i in self.tracking_results:
                        frame = self.draw_tracking_boxes(frame, self.tracking_results[i])
                    
                    # Resize frame to fit grid cell
                    resized_frame = cv2.resize(frame, (cell_w, cell_h))
                    
                    # Add camera name overlay
                    cv2.putText(resized_frame, cam_name, (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(resized_frame, timestamp, (10, cell_h - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Add tracking status
                    if i in self.tracking_results and self.tracking_results[i]:
                        track_count = len(self.tracking_results[i])
                        cv2.putText(resized_frame, f"Tracking: {track_count}", (10, cell_h - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    # Create placeholder for missing camera
                    resized_frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.putText(resized_frame, f"Cam-{i} (No Signal)", (10, cell_h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Place in grid
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                grid_image[y1:y2, x1:x2] = resized_frame
        
        return grid_image

def gui_display_worker(gui_queues, shutdown_flag=None):
    """Main GUI display worker thread"""
    display = MultiCameraDisplay()
    
    print(f"[INFO] Starting GUI display worker with {len(gui_queues)} camera queues")
    
    try:
        while True:
            if shutdown_flag and shutdown_flag.is_set():
                break
                
            # Check all GUI queues for new frames and tracking data
            frames_updated = False
            for cam_id, q_gui in gui_queues.items():
                try:
                    while True:  # Process all available items
                        item = q_gui.get_nowait()
                        
                        if isinstance(item, tuple) and len(item) == 3:
                            if item[0] == 'tracking':
                                # Handle tracking results
                                _, cam_id_recv, tracking_results = item
                                display.update_tracking(cam_id_recv, tracking_results)
                                frames_updated = True
                            else:
                                # Handle regular frame data
                                cam_id_recv, cam_name, frame = item
                                display.update_frame(cam_id_recv, cam_name, frame)
                                frames_updated = True
                        elif isinstance(item, tuple) and len(item) == 3:
                            # Handle regular frame data (backward compatibility)
                            cam_id_recv, cam_name, frame = item
                            display.update_frame(cam_id_recv, cam_name, frame)
                            frames_updated = True
                        elif item is None:
                            # Sentinel value - end of stream
                            break
                            
                except queue.Empty:
                    pass  # No items available for this camera
            
            # Create and display grid (only if we have frames to show)
            grid_image = display.create_grid_display()
            cv2.imshow(display.window_name, grid_image)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[INFO] GUI window closed by user")
                if shutdown_flag:
                    shutdown_flag.set()
                break
                
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)  # ~30 FPS update rate
            
    except KeyboardInterrupt:
        print("\n[INFO] GUI display worker received keyboard interrupt")
    except Exception as e:
        print(f"[ERROR] GUI display worker error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] GUI display worker terminated")