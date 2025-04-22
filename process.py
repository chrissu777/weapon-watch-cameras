import cv2

from detect import detect

def process(rtsp_stream, db, school, building, floor, cam_id):    
    detect(rtsp_stream, db, school, building, floor, cam_id)
    rtsp_stream.stop()
    cv2.destroyAllWindows()