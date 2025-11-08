# controller.py

import cv2
import math
import random
import time
import numpy as np
import config
from simple_pid import PID

class CarController:
    """
    Lớp chính đóng gói toàn bộ logic điều khiển xe tự hành.
    Sử dụng kiến trúc Máy Trạng Thái Hữu Hạn (FSM) và khả năng "nhìn xa"
    để đưa ra các quyết định điều khiển thông minh.
    """
    def __init__(self):
        """Khởi tạo bộ điều khiển."""
        self.pid_controller = PID(config.PID_KP, config.PID_KI, config.PID_KD, setpoint=0)
        self.pid_controller.output_limits = config.PID_OUTPUT_LIMITS
        self.pid_controller.auto_mode = True
        self.state_manager = self.StateManager(self.pid_controller)
        self.last_valid_sign = None
        self.last_sign_time = 0
        self.sign_memory_duration = 2.0 

        self.throttle = config.MAX_THROTTLE

        self.was_long_straight_last_frame = False
        self.debug_image_counter = 0

        print("CarController with Long-Range Perception initialized.")

    class StateManager:
        """
        Lớp lồng (nested class) quản lý trạng thái của xe (FSM).
        Các trạng thái: FOLLOWING_LANE, APPROACHING_INTERSECTION, EXECUTING_TURN.
        """
        def __init__(self, pid_controller_ref):
            self.state = 'FOLLOWING_LANE'
            self.turn_direction = None
            self.action_start_time = 0
            self.pid_controller = pid_controller_ref
            print("StateManager initialized. Initial state: FOLLOWING_LANE")

        def reset(self):
            """Reset trạng thái về ban đầu sau khi hoàn thành một hành động phức tạp."""
            self.state = 'FOLLOWING_LANE'
            self.turn_direction = None
            self.action_start_time = 0
            self.pid_controller.reset()
            print("State reset to FOLLOWING_LANE")

        def decide_turn_direction(self, intersec_dirs, last_sign, is_long_straight):
            """
            Logic ra quyết định hướng đi tại giao lộ theo chuỗi quy tắc ưu tiên.
            Ưu tiên: Biển báo -> Đường thẳng dài -> Rẽ trái -> Rẽ phải.
            """
            print(f"DEBUG -> Sign: '{last_sign}', Paths: {intersec_dirs}, IsLongStraight: {is_long_straight}")

            if last_sign:
                if 'no_' in last_sign:
                    forbidden_dir = last_sign.split('_')[1]
                    allowed_dirs = [d for d in intersec_dirs if d != forbidden_dir]
                    if 'straight' in allowed_dirs and is_long_straight: decision = 'straight'
                    elif 'left' in allowed_dirs: decision = 'left'
                    elif 'right' in allowed_dirs: decision = 'right'
                    else: decision = None
                    print(f"Decision: Obeying forbidden sign '{last_sign}', choosing default '{decision}' from {allowed_dirs}")
                    return decision
                elif last_sign in intersec_dirs:
                    print(f"Decision: Trusting and executing sign '{last_sign}'.")
                    return last_sign
            
            if 'straight' in intersec_dirs and is_long_straight:
                print("Decision: No sign. Defaulting to long straight path.")
                return 'straight'
            
            if 'right' in intersec_dirs:
                print("Decision: No long straight path. Defaulting to 'right'.")
                return 'right'
            
            if 'left' in intersec_dirs:
                print("Decision: No straight or right path. Last resort is 'left'.")
                return 'left'

            print("Decision: No valid path found.")
            return None

        def process(self, steering_from_pid, throttle_from_pid, is_intersection, last_sign, intersec_dirs, lane_line_count, draw, is_long_straight):
            """Hàm chính của FSM, xử lý chuyển đổi trạng thái và hành động."""
            if self.state == 'FOLLOWING_LANE' and is_intersection:
                self.state = 'APPROACHING_INTERSECTION'
                print("State change: FOLLOWING_LANE -> APPROACHING_INTERSECTION")
            
            elif self.state == 'APPROACHING_INTERSECTION':
                self.turn_direction = self.decide_turn_direction(intersec_dirs, last_sign, is_long_straight)
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
            
            is_turning = (self.state != 'FOLLOWING_LANE')
            return final_steering, final_throttle, is_turning

        def blinker(self, direction, draw=None):
            """Vẽ mũi tên chỉ hướng rẽ (xi-nhan)."""
            if draw is not None:
                h, w = draw.shape[:2]
                color = (255, 180, 0)
                if direction == 'left': cv2.arrowedLine(draw, (int(w*0.3), int(h*0.8)), (int(w*0.2), int(h*0.8)), color, 6)
                elif direction == 'right': cv2.arrowedLine(draw, (int(w*0.7), int(h*0.8)), (int(w*0.8), int(h*0.8)), color, 6)
                elif direction == 'straight': cv2.arrowedLine(draw, (w//2, int(h*0.8)), (w//2, int(h*0.7)), color, 6)

    def is_straight_path_long(self, full_frame):
        """
        Phân tích vùng ảnh phía xa để xác định đường có thẳng và dài không.
        """
        h, w = full_frame.shape[:2]
        start_y = int(h * config.FAR_ROI_Y_START_RATIO)
        end_y = int(h * config.FAR_ROI_Y_END_RATIO)
        far_roi = full_frame[start_y:end_y, :]

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        hsv_img = cv2.cvtColor(far_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < config.FAR_ROI_MIN_CONTOUR_AREA:
                continue
            x, y, cont_w, cont_h = cv2.boundingRect(c)
            if cont_w == 0 or cont_h == 0:
                continue
            aspect_ratio = cont_h / cont_w
            if aspect_ratio <= 1.0:
                continue
            valid_contours.append(c)

        if len(valid_contours) == 2:
            moments1, moments2 = cv2.moments(valid_contours[0]), cv2.moments(valid_contours[1])
            if moments1['m00'] > 0 and moments2['m00'] > 0:
                cx1, cx2 = int(moments1['m10'] / moments1['m00']), int(moments2['m10'] / moments2['m00'])
                if (cx1 - w//2) * (cx2 - w//2) < 0: # Kiểm tra 2 contour ở 2 bên tâm ảnh
                    return True, mask 
        
        return False, mask 

    def detect_intersection_paths(self, lane_mask):
        """
        Phát hiện các hướng đi có thể có tại giao lộ.
        *** CẬP NHẬT LOGIC ***
        """
        available_paths = []
        h, w = lane_mask.shape
        
        left_roi = lane_mask[h//2:, :w//3]
        right_roi = lane_mask[h//2:, -w//3:]

        center_roi = lane_mask[:h//2, int(w*0.35):int(w*0.65)]

        if cv2.countNonZero(left_roi) / left_roi.size > 0.10: 
            available_paths.append('left')
        
        if cv2.countNonZero(center_roi) / center_roi.size > 0.15: 
            available_paths.append('straight')
            
        if cv2.countNonZero(right_roi) / right_roi.size > 0.10: 
            available_paths.append('right')
            
        return available_paths

    def get_control_signals(self, image, sign=None, draw=None):
        """Hàm chính, nhận ảnh đầu vào và trả về tín hiệu điều khiển."""
        original_height, original_width = image.shape[:2]
        frame = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
        
        if sign:
            self.last_valid_sign = sign
            self.last_sign_time = time.time()
        if time.time() - self.last_sign_time > self.sign_memory_duration:
            self.last_valid_sign = None

        roi_start_y = int(config.IMG_HEIGHT * config.ROI_Y_START_RATIO)
        pid_roi = frame[roi_start_y:, :]
        pid_img_height, pid_img_width = pid_roi.shape[:2]
        pid_img_center = pid_img_width // 2
        
        lower_gray, upper_gray = np.array(config.GRAY_LANE_LOWER_HSV), np.array(config.GRAY_LANE_UPPER_HSV)
        kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
        
        hsv_img_pid = cv2.cvtColor(pid_roi, cv2.COLOR_BGR2HSV)
        mask_gray_pid = cv2.inRange(hsv_img_pid, lower_gray, upper_gray)
        mask_gray_clean_pid = cv2.morphologyEx(mask_gray_pid, cv2.MORPH_CLOSE, kernel)
        contours_pid, _ = cv2.findContours(mask_gray_clean_pid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        steering_from_pid, throttle_from_pid = 0.0, config.MIN_THROTTLE
        is_intersection, current_area = False, 0
        largest_contour_pid = None
        
        if contours_pid:
            largest_contour_pid = max(contours_pid, key=cv2.contourArea)
            current_area = cv2.contourArea(largest_contour_pid)
            if current_area > config.INTERSECTION_MIN_AREA_THRESHOLD: 
                is_intersection = True # KÍCH HOẠT logic giao lộ
            
            M = cv2.moments(largest_contour_pid)
            if M['m00'] > 0:
                lane_center_x = int(M['m10'] / M['m00'])
                deviation = lane_center_x - pid_img_center
                steering_from_pid = self.pid_controller(-deviation)
                throttle_range = config.MAX_THROTTLE - config.MIN_THROTTLE
                throttle_from_pid = config.MAX_THROTTLE - (abs(steering_from_pid) * throttle_range)
        
        intersec_dirs, is_long_straight = [], False
        debug_far_mask = None 
        mask_gray_clean_intersec = None 


        
        intersec_start_y = int(config.IMG_HEIGHT * config.INTERSEC_ROI_Y_START_RATIO)
        intersec_end_y = int(config.IMG_HEIGHT * config.INTERSEC_ROI_Y_END_RATIO)

        intersection_roi_frame = frame[intersec_start_y:intersec_end_y, :]
        
        hsv_img_intersec = cv2.cvtColor(intersection_roi_frame, cv2.COLOR_BGR2HSV)
        mask_gray_intersec = cv2.inRange(hsv_img_intersec, lower_gray, upper_gray)
        mask_gray_clean_intersec = cv2.morphologyEx(mask_gray_intersec, cv2.MORPH_CLOSE, kernel)
        
        if is_intersection:
            intersec_dirs = self.detect_intersection_paths(mask_gray_clean_intersec)
            
            is_long_straight, debug_far_mask = self.is_straight_path_long(frame) 
            
            if is_long_straight and not self.was_long_straight_last_frame:
                filename = f"debug_long_straight_{self.debug_image_counter}.jpg"
                h, w = original_height, original_width
                start_y = int(h * config.FAR_ROI_Y_START_RATIO)
                end_y = int(h * config.FAR_ROI_Y_END_RATIO)
                
                debug_img_to_save = image.copy()
                cv2.rectangle(debug_img_to_save, (0, start_y), (w, end_y), (0, 255, 255), 2) 
                cv2.putText(debug_img_to_save, "FAR_ROI_CHECK (NOT_HORIZONTAL)", (10, start_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imwrite(filename, debug_img_to_save)
                print(f"DEBUG: Đã lưu ảnh {filename} (is_long_straight chuyển sang True)")
                if debug_far_mask is not None:
                    mask_filename = f"debug_long_straight_MASK_{self.debug_image_counter}.jpg"
                    debug_mask_to_save = cv2.cvtColor(debug_far_mask, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(mask_filename, debug_mask_to_save)
                    print(f"DEBUG: Đã lưu ảnh mask {mask_filename}")
                
                self.debug_image_counter += 1
            
            self.was_long_straight_last_frame = is_long_straight
        else:
             self.was_long_straight_last_frame = False
        
        lane_line_count = 0
        if largest_contour_pid is not None:
            x, y, w, h = cv2.boundingRect(largest_contour_pid)
            if w > pid_img_width * 0.8: lane_line_count = 2
            else: lane_line_count = 1
        
        final_steering, final_throttle, is_turning = self.state_manager.process(
            steering_from_pid, throttle_from_pid, is_intersection, 
            self.last_valid_sign, intersec_dirs, lane_line_count, draw, is_long_straight
        )
        
        done_turning = not is_turning
        if self.last_valid_sign == 'stop':
            final_throttle = 0
        self.throttle = final_throttle
        
        debug_mask = None
        if draw is not None and config.DEBUG_MODE:
            cv2.putText(draw, f"State: {self.state_manager.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(draw, f"Sign Memory: {self.last_valid_sign}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(draw, f"Paths: {intersec_dirs}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(draw, f"IsLongStraight: {is_long_straight}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(draw, f"Steer: {final_steering:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(draw, f"Throttle: {final_throttle:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_start_draw = int(original_height * config.INTERSEC_ROI_Y_START_RATIO) # 60%
            y_half_draw  = int(original_height * (config.INTERSEC_ROI_Y_START_RATIO + (config.INTERSEC_ROI_Y_END_RATIO - config.INTERSEC_ROI_Y_START_RATIO) / 2)) # 70%
            y_end_draw   = int(original_height * config.INTERSEC_ROI_Y_END_RATIO)    # 80%

            w_third_draw = int(original_width / 3)
            w_35_draw = int(original_width * 0.35)
            w_65_draw = int(original_width * 0.65)
            
            cv2.rectangle(draw, (0, y_half_draw), (w_third_draw, y_end_draw), (0, 255, 0), 2)
            cv2.putText(draw, "LEFT_PATH", (5, y_half_draw + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.rectangle(draw, (w_35_draw, y_start_draw), (w_65_draw, y_half_draw), (255, 0, 0), 2)
            cv2.putText(draw, "STRAIGHT_PATH", (w_35_draw + 5, y_start_draw + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.rectangle(draw, (original_width - w_third_draw, y_half_draw), (original_width - 1, y_end_draw), (0, 0, 255), 2)
            cv2.putText(draw, "RIGHT_PATH", (original_width - w_third_draw + 5, y_half_draw + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.line(draw, (0, int(original_height * config.ROI_Y_START_RATIO)), (original_width, int(original_height * config.ROI_Y_START_RATIO)), (255, 0, 0), 2) # PID ROI (80%)
            far_start_y = int(original_height * config.FAR_ROI_Y_START_RATIO) # Far ROI (50%)
            far_end_y = int(original_height * config.FAR_ROI_Y_END_RATIO)     # Far ROI (75%)
            cv2.rectangle(draw, (0, far_start_y), (original_width, far_end_y), (0, 255, 255), 2)

            if is_intersection:
                debug_mask = cv2.cvtColor(mask_gray_clean_intersec, cv2.COLOR_GRAY2BGR)
                h_dbg, w_dbg = debug_mask.shape[:2]
                
                cv2.rectangle(debug_mask, (int(w_dbg*0.35), 0), (int(w_dbg*0.65), h_dbg//2), (255, 0, 0), 2)
                
                cv2.rectangle(debug_mask, (0, h_dbg//2), (w_dbg//3, h_dbg-1), (0, 255, 0), 2)

                cv2.rectangle(debug_mask, (w_dbg - (w_dbg//3), h_dbg//2), (w_dbg-1, h_dbg-1), (0, 0, 255), 2)
            else:
                debug_mask = cv2.cvtColor(mask_gray_clean_pid, cv2.COLOR_GRAY2BGR)
                if 'lane_center_x' in locals():
                    h_dbg, w_dbg = debug_mask.shape[:2]
                    cv2.line(debug_mask, (pid_img_center, 0), (pid_img_center, h_dbg), (0, 0, 255), 2)
                    cv2.line(debug_mask, (lane_center_x, 0), (lane_center_x, h_dbg), (0, 255, 0), 2)

        return final_throttle, final_steering, done_turning, debug_mask