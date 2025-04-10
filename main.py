import json
import multiprocessing

from stream import RTSPStream
from process import process

def worker(rtsp_url):
    stream = RTSPStream(rtsp_url)
    process(stream)

if __name__ == '__main__':
    with open('camera_info.json') as f:
        data = json.load(f)

    processes = []
    for cam in data['IDEA factory']['floor 1']:
        rtsp_url = cam['video link']
        p = multiprocessing.Process(target=worker, args=(rtsp_url,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()