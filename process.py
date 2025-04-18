import multiprocessing
import cv2

from detect import detect

def process(rtsp_stream, camera_path):
    detect(rtsp_stream, camera_path)
    rtsp_stream.stop()
    cv2.destroyAllWindows()