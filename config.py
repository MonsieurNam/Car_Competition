# config.py

# =============================================================================
# CHẾ ĐỘ DEBUG
# =============================================================================
# Đặt thành True để hiển thị các cửa sổ hình ảnh xử lý, False để chạy ở chế độ thi đấu (tối đa hiệu năng)
DEBUG_MODE = True


# =============================================================================
# CẤU HÌNH XỬ LÝ ẢNH
# =============================================================================
# Kích thước ảnh sẽ được resize trước khi xử lý để tăng tốc độ
IMG_WIDTH = 320
IMG_HEIGHT = 280

# Vùng Quan Tâm (Region of Interest - ROI)
# Chỉ xử lý phần ảnh từ ROI_Y_START (tính từ đỉnh) đến cuối ảnh (tỷ lệ 0.0 -> 1.0)
ROI_Y_START_RATIO = 0.75


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
PID_KD = 0.003

# Giới hạn giá trị output của PID để tránh góc lái quá lớn (-1.0 là rẽ trái tối đa, 1.0 là rẽ phải tối đa)
PID_OUTPUT_LIMITS = (-1.0, 1.0)

ANOMALY_DEVIATION_THRESHOLD = 60
# =============================================================================
# CẤU HÌNH TỐC ĐỘ ĐỘNG (THROTTLE)
# =============================================================================
# Tốc độ tối đa khi xe chạy trên đường thẳng
MAX_THROTTLE = 1

# Tốc độ tối thiểu khi xe vào cua gắt
MIN_THROTTLE = 0.3

# Tốc độ của xe khi đang tiếp cận một giao lộ đã được phát hiện
THROTTLE_AT_INTERSECTION_APPROACH = 0.45

THROTTLE_AT_TURN = 0.2

# Thời gian tối thiểu và tối đa cho một hành động rẽ
# Xe sẽ ở trạng thái rẽ trong ít nhất MIN giây và nhiều nhất MAX giây
MIN_TURNING_TIME = 0.5 # (giây)
MAX_TURNING_TIME = 0.7 # (giây)

# =============================================================================
# CẤU HÌNH LOGIC GIAO LỘ
# =============================================================================
# Ngưỡng diện tích tối thiểu để xác định một vùng có thể là giao lộ
# GIÁ TRỊ NÀY CẦN ĐƯỢC TINH CHỈNH DỰA TRÊN THỰC TẾ SAU KHI RESIZE ẢNH
INTERSECTION_MIN_AREA_THRESHOLD = 16000

# (Tùy chọn cho Giải pháp 2) Tỷ lệ để phát hiện sự tăng vọt so với diện tích trung bình
INTERSECTION_AREA_SPIKE_RATIO = 1.5

# Ngưỡng để xác định hướng đi có tồn tại ở giao lộ hay không
INTERSECTION_LEFT_THRESHOLD = 100
INTERSECTION_RIGHT_THRESHOLD = 60
INTERSECTION_STRAIGHT_THRESHOLD = 2 # Số lượng contour màu đen tối thiểu

# Giá trị góc lái CỘNG THÊM khi rẽ tại giao lộ
# Lưu ý: PID đã xử lý việc bám làn, giá trị này chỉ để "ép" xe rẽ dứt khoát hơn
TURN_LEFT_STEERING = -0.75
TURN_RIGHT_STEERING = 0.75

# Khoảng cách offset để tính toán góc lái khi chỉ thấy 1 bên làn ở giao lộ
# (Giá trị cũ là 220, giảm 1 nửa do resize ảnh)
LANE_OFFSET_AT_INTERSECTION = 110