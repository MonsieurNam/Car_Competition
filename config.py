# config.py

# =============================================================================
# CHẾ ĐỘ DEBUG
# =============================================================================
# Đặt thành True để hiển thị các cửa sổ hình ảnh xử lý
DEBUG_MODE = True


# =============================================================================
# CẤU HÌNH XỬ LÝ ẢNH
# =============================================================================
IMG_WIDTH = 320
IMG_HEIGHT = 280

# Vùng Quan Tâm (Region of Interest - ROI)
# Chỉ xử lý phần ảnh từ ROI_Y_START (tính từ đỉnh) đến cuối ảnh (tỷ lệ 0.0 -> 1.0)
ROI_Y_START_RATIO = 0.8


# =============================================================================
# CẤU HÌNH PHÁT HIỆN LÀN ĐƯỜNG MÀU XÁM
# =============================================================================
# Ngưỡng màu HSV cho làn đường màu xám
GRAY_LANE_LOWER_HSV = [0, 0, 50]
GRAY_LANE_UPPER_HSV = [180, 50, 150]

# Kích thước kernel cho các phép toán hình thái học
MORPH_KERNEL_SIZE = (3, 3)


# =============================================================================
# CẤU HÌNH BỘ ĐIỀU KHIỂN PID
# =============================================================================
# Các hệ số P, I, D được lấy từ Dự án 1 để làm điểm khởi đầu
PID_KP = 0.02 # Bắt đầu với 0.01 hoặc 0.02
PID_KI = 0.00
PID_KD = 0.0058
PID_OUTPUT_LIMITS = (-1.0, 1.0)
# =============================================================================
# CẤU HÌNH TỐC ĐỘ ĐỘNG (THROTTLE)
# =============================================================================
MAX_THROTTLE = 0.88
MIN_THROTTLE = 0.3
THROTTLE_AT_INTERSECTION_APPROACH = 0.64
THROTTLE_AT_TURN = 0.45
MIN_TURNING_TIME = 0.2 
MAX_TURNING_TIME = 0.3 

# =============================================================================
# CẤU HÌNH LOGIC GIAO LỘ
# =============================================================================
# Ngưỡng diện tích tối thiểu để xác định một vùng có thể là giao lộ
INTERSECTION_MIN_AREA_THRESHOLD = 16000

TURN_LEFT_STEERING = -0.8
TURN_RIGHT_STEERING = 0.8

FAR_ROI_Y_START_RATIO = 0.50
FAR_ROI_Y_END_RATIO   = 0.75 
FAR_ROI_MIN_CONTOUR_AREA = 50