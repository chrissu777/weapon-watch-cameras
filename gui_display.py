import cv2
import queue
import threading
import time

import numpy as np

class MultiCameraDisplay:
    def __init__(self, num_cameras):
        self.num_cameras = num_cameras
        self.latest_frames = {}
        self.frame_lock = threading.Lock()
        self.window_name = "Weapon Detection - Multi Camera View"
        
    def update_frame(self, cam_id, cam_name, frame):
        """Update the latest frame for a specific camera"""
        with self.frame_lock:
            self.latest_frames[int(cam_name[-1])] = (cam_name, frame)
    
    def create_grid_display(self):
        """Create a 2x3 grid display for 6 cameras"""
        # Grid configuration for 6 cameras (2 rows, 3 columns)
        grid_h, grid_w = 2, 3
        cell_h, cell_w = 450, 600  # Resize each camera view
         
        # Create empty grid
        grid_image = np.zeros((grid_h * cell_h, grid_w * cell_w, 3), dtype=np.uint8)
        
        with self.frame_lock:
            for i in range(1, self.num_cameras + 1):
                row = (i - 1) // grid_w
                col = (i - 1) % grid_w
                
                if i in self.latest_frames:
                    try:
                        cam_name, frame = self.latest_frames[i]
                        if frame is not None:
                            # Resize frame to fit grid cell
                            resized_frame = cv2.resize(frame, (cell_w, cell_h))
                            
                            # Add cam name and timestamp
                            timestamp = time.strftime("%H:%M:%S")
                            cv2.putText(resized_frame, f"{cam_name}, {timestamp}", (10, cell_h - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        else:
                            resized_frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                            cv2.putText(resized_frame, f"{cam_name} (No Frame)", (10, cell_h - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    except Exception as e:
                        print(f"[WARNING] Error processing frame for camera {i}: {e}")
                        resized_frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                        cv2.putText(resized_frame, f"{cam_name} (Error)", (10, cell_h - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    resized_frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.putText(resized_frame, f"Cam-{i} (No Signal)", (10, cell_h - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Place in grid
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                grid_image[y1:y2, x1:x2] = resized_frame
                
                # Add bounding box around each camera feed
                border_color = (100, 100, 100)  # Gray border
                border_thickness = 2
                cv2.rectangle(grid_image, (x1, y1), (x2-1, y2-1), border_color, border_thickness)
        
        return grid_image

def gui_display_worker(gui_queues, shutdown_flag=None):
    """Main GUI display worker thread"""
    display = MultiCameraDisplay(len(gui_queues))
        
    try:
        while True:
            if shutdown_flag and shutdown_flag.is_set():
                break
                
            # Check all GUI queues for new frames
            for cam_id, q_gui in gui_queues.items():
                try:
                    while True:  # Process all available frames
                        cam_id_recv, cam_name, frame = q_gui.get_nowait()
                        display.update_frame(cam_id_recv, cam_name, frame)
                except queue.Empty:
                    pass  # No frames available for this camera
            
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