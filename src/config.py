import os
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục models & samples
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "output"

# Đảm bảo các thư mục tồn tại
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đường dẫn mô hình YOLO phát hiện biển số
YOLO_MODEL_PATH = str(MODELS_DIR / "license_plate_detector.pt")

# Ngưỡng phát hiện
DETECTION_CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.40

# Danh sách mã tỉnh thành phố hợp lệ của Việt Nam
VN_PROVINCE_CODES = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng", "16": "Hải Phòng",
    "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ", "20": "Thái Nguyên", "21": "Yên Bái",
    "22": "Tuyên Quang", "23": "Hà Giang", "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
    "32": "Hà Nội", "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình", "36": "Thanh Hóa",
    "37": "Nghệ An", "38": "Hà Tĩnh", "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông",
    "49": "Lâm Đồng", "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh",
    "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh",
    "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh", "60": "Đồng Nai",
    "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang", "64": "Vĩnh Long", "65": "Cần Thơ",
    "66": "Đồng Tháp", "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh",
    "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình", "74": "Quảng Trị",
    "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên",
    "79": "Khánh Hòa", "80": "Cục CSGT / Bộ ngành", "81": "Gia Lai", "82": "Kon Tum",
    "83": "Sóc Trăng", "84": "Trà Vinh", "85": "Ninh Thuận", "86": "Bình Thuận",
    "88": "Vĩnh Phúc", "89": "Hưng Yên", "90": "Hà Nam", "92": "Quảng Nam",
    "93": "Bình Phước", "94": "Bạc Liêu", "95": "Hậu Giang", "97": "Bắc Kạn",
    "98": "Bắc Giang", "99": "Bắc Ninh"
}

# Bảng chuyển đổi ký tự hay bị nhầm lẫn khi OCR
# Khi ở vị trí số nhưng bị nhận diện thành chữ
CHAR_TO_NUM = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0',
    'I': '1', 'L': '1', '|': '1', 'T': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'B': '8',
    'P': '9', 'q': '9'
}

# Khi ở vị trí chữ cái nhưng bị nhận diện thành số
NUM_TO_CHAR = {
    '0': 'D',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}
