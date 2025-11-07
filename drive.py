# drive.py (Phiên bản cập nhật để visualize)

import asyncio, base64, json, time, os
from io import BytesIO
from multiprocessing import Process, Queue, Manager
import cv2, numpy as np, websockets
from PIL import Image
import config
from controller import CarController
from traffic_sign_detection import *
from functools import partial

file_path = os.path.dirname(os.path.realpath(__file__))
traffic_sign_model = cv2.dnn.readNetFromONNX(
    fr"{file_path}\traffic_sign_classifier_lenet_v3.onnx")

def process_traffic_sign_loop(g_image_queue, processed_sign):
    while True:
        if g_image_queue.empty():
            time.sleep(0.01) 
            continue
        image = g_image_queue.get()
        detected_signs = detect_traffic_signs(image, traffic_sign_model)
        if detected_signs:
            best_sign = detected_signs[0] 
            processed_sign[:] = [best_sign['class'], time.time(), best_sign['bbox']]
        else:
            if processed_sign and processed_sign[1] != 0 and time.time() - processed_sign[1] > 2.0:
                processed_sign[:] = [None, 0, None]


async def process_image(websocket, car_controller, processed_sign, g_image_queue):
    async for message in websocket:
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = image.copy() if config.DEBUG_MODE else None
        sign_ = processed_sign[0] if processed_sign else None
        throttle, steering_angle, done, debug_mask = car_controller.get_control_signals(image, sign=sign_, draw=draw)
        if not g_image_queue.full():
            g_image_queue.put(image)
        if config.DEBUG_MODE and draw is not None:
            bbox_ = processed_sign[2] if processed_sign and processed_sign[2] else None
            if sign_ and bbox_:
                x, y, w, h = bbox_
                cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(draw, sign_, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Result", draw)
            if debug_mask is not None:
                cv2.imshow("Lane Segmentation", debug_mask)
            cv2.waitKey(1)
        message = json.dumps({"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)


async def main():
    car_controller = CarController()
    
    with Manager() as manager:
        processed_sign = manager.list([None, 0, None])
        g_image_queue = Queue(maxsize=5)
        p = Process(target=process_traffic_sign_loop, args=(g_image_queue, processed_sign))
        p.start()
        handler = partial(process_image, car_controller=car_controller, processed_sign=processed_sign, g_image_queue=g_image_queue)
        async with websockets.serve(handler, "0.0.0.0", 4567, ping_interval=None):
            await asyncio.Future()

if __name__ == '__main__':
    if config.DEBUG_MODE:
        cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Lane Segmentation", cv2.WINDOW_AUTOSIZE)
        
    asyncio.run(main())