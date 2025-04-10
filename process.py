import multiprocessing
import cv2

from confirm import confirm
from record import record
from detect import detect

def process(rtsp_stream):
    confirm_q = multiprocessing.Queue()
    record_q = multiprocessing.Queue()

    confirm_p = multiprocessing.Process(target=confirm, args=(confirm_q,))
    confirm_p.start()

    record_p = multiprocessing.Process(target=record, args=(record_q,))
    record_p.start()

    detect(confirm_q, record_q, rtsp_stream)
    rtsp_stream.stop()
    cv2.destroyAllWindows()

    confirm_q.close()
    record_q.close()

    confirm_q.join_thread()
    record_q.join_thread()

    confirm_p.join()
    record_p.join()  