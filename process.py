import cv2

from detect import detect

def process(stream, db, cam_id, school):    
    detect(stream, db, cam_id, school)
    stream.stop()
    cv2.destroyAllWindows()