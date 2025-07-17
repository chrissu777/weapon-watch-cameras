import os
import shutil
import cv2
import glob
import time
import argparse

from tqdm import tqdm
import tensorflow as tf

from detect import detect

def run_script(vids_path, vids_type, output_dir, grayscale):
    # Suppress logging warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"
    
    vid_links = glob.glob(os.path.join(vids_path, f"*.{vids_type}"))
    
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    detection_model = tf.saved_model.load('detectionmodel')
    infer_weapon = detection_model.signatures['serving_default']
    print(f"DETECTION MODEL LOADED")

    start_time = time.time()
    for video_path in vid_links:
        cam_name = os.path.basename(video_path).split(".")[0]
        print(f"\nPlaying: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=os.path.basename(video_path), unit='frame')

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if i%2 == 0:
                detect(frame, cam_name, infer_weapon, i, output_dir, grayscale)

            i += 1
            pbar.update(1)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        cap.release()
        pbar.close()
        
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.2f} minutes")

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the detection testing script."
    )
    parser.add_argument(
        "-p",
        "--vids_path",
        type=str,
        nargs="?",
        default="finals_verification/mp4_vids/phase2",
        help="directory to play vids from",
    )
    parser.add_argument(
        "-t",
        "--vids_type",
        type=str,
        nargs="?",
        default="mp4",
        help="type of file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        default="finals_verification/phase2",
        help="directory to save the detected images.",
    )
    parser.add_argument(
        "-g",
        "--grayscale",
        type=bool,
        nargs="?",
        default=False,
        help="grayscale frames or not",
    )
    args = parser.parse_args()
    
    run_script(
        vids_path=args.vids_path, 
        vids_type=args.vids_type, 
        output_dir=args.output_dir, 
        grayscale=args.grayscale
    )