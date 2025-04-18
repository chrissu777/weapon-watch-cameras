import json
import multiprocessing

from stream import RTSPStream
from process import process

def streamer(rtsp_url, camera_path):
    stream = RTSPStream(rtsp_url)
    process(stream, camera_path)

if __name__ == '__main__':
    with open('camera_info.json') as f:
        data = json.load(f)

    processes = []
    i = 0
    for cam in data['IDEA factory']['floor 1']:
        rtsp_url = cam['video link']
        p = multiprocessing.Process(target=streamer, args=(rtsp_url, data['IDEA factory']['floor 1'][i],))
        processes.append(p)
        p.start()
        i += 1

    for p in processes:
        p.join()