import cv2
import threading

class RTSPStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame = None
        self.running = True
        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):        
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            else:
                self.running = False
                break
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()


# if __name__ == '__main__':
#     rtsp_url = "rtsp://ww:weaponwatch1@192.168.1.177:554/profile2/media.smp"
#     rtsp_stream = RTSPStream(rtsp_url)

#     while True:
#         frame = rtsp_stream.read()
#         if frame is not None:
#             cv2.imshow('RTSP Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     rtsp_stream.stop()
#     cv2.destroyAllWindows()