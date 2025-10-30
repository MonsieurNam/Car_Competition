# drive.py
import asyncio, base64, json, time, os
from io import BytesIO
# === THÊM MỚI: Import Queue trực tiếp từ multiprocessing ===
from multiprocessing import Process, Queue, Manager
import cv2, numpy as np, websockets
from PIL import Image
import config
from controller import CarController
from traffic_sign_detection import *
from functools import partial

# Initalize traffic sign classifier
file_path = os.path.dirname(os.path.realpath(__file__))
traffic_sign_model = cv2.dnn.readNetFromONNX(
    fr"{file_path}\traffic_sign_classifier_lenet_v3.onnx")

# === BỎ ĐI: Không khởi tạo queue ở global scope nữa ===
# g_image_queue = Queue(maxsize=5) 

# Function to run sign classification model continuously
def process_traffic_sign_loop(g_image_queue, processed_sign):
    # ... (code bên trong hàm này giữ nguyên) ...
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()
        draw = image.copy()
        detected_signs = list(detect_traffic_signs(image, traffic_sign_model, draw=draw))
        if detected_signs:
            processed_sign[:] = [detected_signs[0], time.time()]
        else:
            if processed_sign and processed_sign[1] != 0 and time.time() - processed_sign[1] > 2.0:
                processed_sign[:] = [None, 0]


# async def process_image... (hàm này giữ nguyên)
async def process_image(websocket, car_controller, processed_sign, g_image_queue):
    async for message in websocket:
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = image.copy() if config.DEBUG_MODE else None
        sign_ = processed_sign[0] if processed_sign else None
        throttle, steering_angle, done = car_controller.get_control_signals(image, sign=sign_, draw=draw)
        if not g_image_queue.full():
            g_image_queue.put(image)
        if config.DEBUG_MODE and draw is not None:
            cv2.imshow("Result", draw)
            cv2.waitKey(1)
        message = json.dumps({"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)


# async def main... (hàm này được cập nhật)
async def main():
    car_controller = CarController()
    
    with Manager() as manager:
        processed_sign = manager.list([None, 0])
        
        # === THÊM MỚI: Khởi tạo queue ở đây, trong cùng scope với manager ===
        g_image_queue = Queue(maxsize=5)

        p = Process(target=process_traffic_sign_loop, args=(g_image_queue, processed_sign))
        p.start()
        
        # === CẬP NHẬT: Truyền cả g_image_queue vào handler ===
        handler = partial(process_image, car_controller=car_controller, processed_sign=processed_sign, g_image_queue=g_image_queue)
        
        async with websockets.serve(handler, "0.0.0.0", 4567, ping_interval=None):
            await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())