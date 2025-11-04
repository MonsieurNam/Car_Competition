# Gray_lane_cal.py (Phiên bản tái cấu trúc)

import cv2, math, random, time, numpy as np, config
from simple_pid import PID

# === LỚP CARCONTROLLER MỚI, BAO BỌC TẤT CẢ LOGIC ===
class CarController:
    def __init__(self):
        # Khởi tạo PID và StateManager BÊN TRONG controller
        self.pid_controller = PID(config.PID_KP, config.PID_KI, config.PID_KD, setpoint=0)
        self.pid_controller.output_limits = config.PID_OUTPUT_LIMITS
        self.pid_controller.auto_mode = True
        
        self.state_manager = self.StateManager(self.pid_controller)
        
        self.avg_area_buffer = []
        self.throttle = config.MAX_THROTTLE
        print("CarController initialized.")

    # === Chuyển StateManager thành lớp lồng (nested class) ===
    class StateManager:
        def __init__(self, pid_controller_ref):
            self.state = 'FOLLOWING_LANE'
            self.turn_direction = None
            self.action_start_time = 0
            # Giữ một tham chiếu đến PID controller bên ngoài
            self.pid_controller = pid_controller_ref 
            print("StateManager initialized. Initial state: FOLLOWING_LANE")

        def reset(self):
            self.state = 'FOLLOWING_LANE'
            self.turn_direction = None
            self.action_start_time = 0
            self.pid_controller.reset()
            print("State reset to FOLLOWING_LANE")

        def decide_turn_direction(self, intersec_dirs, sign):
            print(f"DEBUG inside decide_turn_direction -> Sign: '{sign}', Detected Dirs: {intersec_dirs}")
            if sign and 'no_' not in sign:
                print(f"Decision: Trusting sign '{sign}'")
                return sign
            if sign and 'no_' in sign:
                forbidden_dir = sign.split('_')[1]
                allowed_dirs = [d for d in intersec_dirs if d != forbidden_dir]
                if allowed_dirs:
                    decision = allowed_dirs[0]
                    print(f"Decision: Obeying forbidden sign '{sign}', choosing '{decision}'")
                    return decision
            if 'straight' in intersec_dirs:
                print("Decision: No valid sign, defaulting to 'straight'")
                return 'straight'
            if len(intersec_dirs) > 0:
                decision = random.choice(intersec_dirs)
                print(f"Decision: No other option, choosing '{decision}'")
                return decision
            print("Decision: Could not determine direction.")
            return None

        def process(self, steering_from_pid, throttle_from_pid, is_intersection, sign, intersec_dirs, lane_line_count, draw=None):
            if self.state == 'FOLLOWING_LANE' and is_intersection:
                self.state = 'APPROACHING_INTERSECTION'
                print("State change: FOLLOWING_LANE -> APPROACHING_INTERSECTION")
            elif self.state == 'APPROACHING_INTERSECTION':
                self.turn_direction = self.decide_turn_direction(intersec_dirs, sign)
                if self.turn_direction:
                    self.state = 'EXECUTING_TURN'
                    self.action_start_time = time.time()
                    print(f"State change: -> EXECUTING_TURN (Decision: {self.turn_direction})")
            elif self.state == 'EXECUTING_TURN':
                time_in_turn = time.time() - self.action_start_time
                if time_in_turn > config.MAX_TURNING_TIME or \
                   (time_in_turn > config.MIN_TURNING_TIME and lane_line_count == 2):
                    self.reset()
            final_steering, final_throttle = steering_from_pid, throttle_from_pid
            if self.state == 'APPROACHING_INTERSECTION':
                final_throttle = config.THROTTLE_AT_INTERSECTION_APPROACH
            elif self.state == 'EXECUTING_TURN':
                self.blinker(self.turn_direction, draw)
                if self.turn_direction == 'left':
                    final_steering, final_throttle = config.TURN_LEFT_STEERING, config.THROTTLE_AT_TURN
                elif self.turn_direction == 'right':
                    final_steering, final_throttle = config.TURN_RIGHT_STEERING, config.THROTTLE_AT_TURN
                elif self.turn_direction == 'straight':
                    final_steering, final_throttle = steering_from_pid, config.THROTTLE_AT_TURN
            return final_steering, final_throttle, (self.state != 'FOLLOWING_LANE')
        
        def blinker(self, dir, draw=None):
            # ... (logic blinker giữ nguyên) ...
            if draw is not None:
                if dir == 'left': cv2.arrowedLine(draw, (int(0.3*draw.shape[1]), int(0.8*draw.shape[0])), (int(0.25*draw.shape[1]), int(0.8*draw.shape[0])), (92, 42, 130), 5)
                elif dir == 'right': cv2.arrowedLine(draw, (int(0.7*draw.shape[1]), int(0.8*draw.shape[0])), (int(0.75*draw.shape[1]), int(0.8*draw.shape[0])), (92, 42, 130), 5)
                elif dir == 'straight': cv2.arrowedLine(draw, (draw.shape[1]//2, int(0.8*draw.shape[0])), (draw.shape[1]//2, int(0.75*draw.shape[0])), (92, 42, 130), 5)


    def detect_intersection(self, contours, image):
        # ... (logic detect_intersection giữ nguyên) ...
        out = []
        img_left, img_right, img_top = image[:, :41], image[:, image.shape[1]-41:], image[:80, :]
        black_cons, _ = cv2.findContours(cv2.bitwise_not(img_top), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_cons = [x for x in black_cons if cv2.contourArea(x) > 2000]
        means = {'left': np.mean(img_left), 'straight': len(black_cons), 'right': np.mean(img_right)}
        threshold = {'left': config.INTERSECTION_LEFT_THRESHOLD, 'straight': config.INTERSECTION_STRAIGHT_THRESHOLD, 'right': config.INTERSECTION_RIGHT_THRESHOLD}
        for dir in means.keys():
            if threshold[dir] <= means[dir]: out.append(dir)
        return out

    # === Chuyển cal_steering thành một phương thức của lớp ===
    def get_control_signals(self, image, sign=None, draw=None):
        # ... (toàn bộ logic của cal_steering cũ sẽ nằm ở đây, và nó sẽ dùng self.pid_controller, self.state_manager, ...)
        original_height, original_width = image.shape[:2]
        frame = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
        roi_start_y = int(config.IMG_HEIGHT * config.ROI_Y_START_RATIO)
        roi = frame[roi_start_y:, :]
        img_height, img_width = roi.shape[:2]
        img_center = img_width // 2
        
        # ... (phần xử lý mask và tìm contour giữ nguyên)
        lower_gray, upper_gray = np.array(config.GRAY_LANE_LOWER_HSV), np.array(config.GRAY_LANE_UPPER_HSV)
        kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_gray = cv2.inRange(hsv_img, lower_gray, upper_gray)
        mask_gray_clean = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_gray_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        steering_from_pid, throttle_from_pid = 0.0, config.MIN_THROTTLE
        current_area, is_intersection = 0, False
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            current_area = cv2.contourArea(largest_contour)
            if current_area > config.INTERSECTION_MIN_AREA_THRESHOLD: is_intersection = True
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                lane_center_x = int(M['m10'] / M['m00'])
                deviation = lane_center_x - img_center
                steering_from_pid = self.pid_controller(-deviation)
                throttle_range = config.MAX_THROTTLE - config.MIN_THROTTLE
                throttle_from_pid = config.MAX_THROTTLE - (abs(steering_from_pid) * throttle_range)
        
        intersec_dirs = []
        if is_intersection: intersec_dirs = self.detect_intersection(largest_contour, mask_gray_clean)
        
        lane_line_count = 0
        if contours:
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w > img_width * 0.8: lane_line_count = 2
            else: lane_line_count = 1
        
        final_steering, final_throttle, is_turning = self.state_manager.process(
            steering_from_pid, throttle_from_pid, is_intersection, 
            sign, intersec_dirs, lane_line_count, draw
        )

        if sign == 'stop': final_throttle = 0
        self.throttle = final_throttle
        
        if draw is not None and config.DEBUG_MODE:
            # ... (phần visualization giữ nguyên) ...
            cv2.putText(draw, f"State: {self.state_manager.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            if self.state_manager.turn_direction: cv2.putText(draw, f"Decision: {self.state_manager.turn_direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(draw, f"Area: {current_area:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(draw, f"Steer: {final_steering:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(draw, f"Throttle: {final_throttle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(draw, (0, int(original_height * config.ROI_Y_START_RATIO)), (original_width, int(original_height * config.ROI_Y_START_RATIO)), (255, 0, 0), 2)
            
        return final_throttle, final_steering, not is_turning