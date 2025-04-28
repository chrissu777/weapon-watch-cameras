# process.py
import cv2
import time
from multiprocessing import Process, Queue

from detect import detect_worker
from record import record_worker
from stream import RTSPStream

def frame_reader(rtsp_url, q_detect, q_record):
    stream = RTSPStream(rtsp_url)
    
    while True:
        frame = stream.read()
        
        if frame is not None:
            q_detect.put(frame)
            q_record.put(frame)
        else:
            print("INVALID FRAME")
            time.sleep(0.1)

    stream.stop()

def process(rtsp_url, cam_id, school):
    q_detect = Queue(maxsize=32)   # bounded so we don’t OOM
    q_record = Queue(maxsize=32)

    p_read   = Process(target=frame_reader, args=(rtsp_url, q_detect, q_record), name="reader")
    p_detect = Process(target=detect_worker, args=(q_detect, cam_id, school),    name="detector")
    p_record = Process(target=record_worker, args=(q_record, cam_id),            name="recorder")
    
    p_read.start()
    p_detect.start()
    p_record.start()

    p_read.join()
    p_detect.join()
    p_record.join()