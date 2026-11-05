import re
from src.config import VN_PROVINCE_CODES, CHAR_TO_NUM, NUM_TO_CHAR


class PlatePostprocessor:
    """
    Module hậu xử lý và chuẩn hóa chuỗi biển số xe Việt Nam:
    - Làm sạch ký tự lạ, dấu chấm, dấu phẩy thừa
    - Sửa lỗi nhận dạng phổ biến theo vị trí (chữ số vs chữ cái)
    - Nhận biết kiểu biển số (1 dòng: ô tô dài; 2 dòng: xe máy / ô tô vuông)
    - Tra cứu tỉnh thành cấp biển
    - Chuẩn hóa định dạng hiển thị đẹp (VD: 59-F1 123.45, 30A-999.99)
    """

    @classmethod
    def clean_raw_text(cls, text: str) -> str:
        """Loại bỏ ký tự đặc biệt, chỉ giữ chữ cái và số."""
        if not text:
            return ""
        # Đưa về chữ hoa và chỉ giữ chữ cái + chữ số
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned

    @classmethod
    def correct_characters_contextually(cls, text: str, is_two_line: bool = False) -> str:
        """
        Sửa lỗi nhận diện dựa trên ngữ cảnh cấu trúc biển số Việt Nam:
        - 2 ký tự đầu luôn là SỐ (Mã tỉnh)
        - Ký tự tiếp theo là sê-ri CHỮ CÁI
        - Các ký tự cuối cùng là CHỮ SỐ (Số thứ tự đăng ký xe)
        """
        raw = list(cls.clean_raw_text(text))
        n = len(raw)
        if n < 6:
            return "".join(raw)

        # 1. Hai ký tự đầu luôn là SỐ
        for i in [0, 1]:
            if raw[i] in CHAR_TO_NUM:
                raw[i] = CHAR_TO_NUM[raw[i]]

        # 2. Xử lý các ký tự cuối cùng: luôn là SỐ (4 hoặc 5 số cuối)
        tail_num_count = 5 if n >= 8 else 4
        tail_start_idx = n - tail_num_count
        for i in range(tail_start_idx, n):
            if raw[i] in CHAR_TO_NUM:
                raw[i] = CHAR_TO_NUM[raw[i]]

        # 3. Ký tự sê-ri ở giữa:
        # Trường hợp xe máy: Vị trí thứ 3 là CHỮ CÁI, Vị trí thứ 4 có thể là SỐ hoặc CHỮ (VD: 59-F1 hoặc 59-FA)
        # Trường hợp ô tô: Vị trí thứ 3 là CHỮ CÁI (VD: 30A, 51K), hoặc có thể thêm 1 chữ cái (VD: 30AB, 51LD)
        if n >= 3 and raw[2] in NUM_TO_CHAR:
            raw[2] = NUM_TO_CHAR[raw[2]]

        return "".join(raw)

    @classmethod
    def format_plate(cls, text_lines: list[str]) -> dict:
        """
        Xử lý danh sách các dòng văn bản từ OCR và định dạng thành biển số xe hoàn chỉnh.
        text_lines: Danh sách các dòng được OCR đọc được (thường từ 1 đến 2 dòng).
        """
        if not text_lines:
            return {
                "plate_number": "UNKNOWN",
                "formatted": "UNKNOWN",
                "province": "Không rõ",
                "vehicle_type": "Không rõ",
                "is_valid": False
            }

        # Nếu có nhiều dòng (biển 2 dòng)
        cleaned_lines = [cls.clean_raw_text(line) for line in text_lines if cls.clean_raw_text(line)]
        is_two_line = len(cleaned_lines) >= 2

        if is_two_line:
            # Dòng 1: mã tỉnh + sê-ri (VD: 59F1 hoặc 29A)
            # Dòng 2: các số đuôi (VD: 12345 hoặc 9999)
            line1 = cls.clean_raw_text(cleaned_lines[0])
            line2 = cls.clean_raw_text("".join(cleaned_lines[1:]))

            combined = line1 + line2
            corrected = cls.correct_characters_contextually(combined, is_two_line=True)
            split_idx = len(line1)
            corr_line1 = corrected[:split_idx]
            corr_line2 = corrected[split_idx:]
            formatted, vehicle_type = cls._format_two_line(corr_line1, corr_line2)
            plate_number = f"{corr_line1}-{corr_line2}"
        else:
            single = "".join(cleaned_lines)
            corrected = cls.correct_characters_contextually(single, is_two_line=False)
            formatted, vehicle_type = cls._format_single_line(corrected)
            plate_number = corrected

        # Tra cứu tỉnh thành từ 2 chữ số đầu
        province_code = corrected[:2] if len(corrected) >= 2 else ""
        province = VN_PROVINCE_CODES.get(province_code, "Không rõ tỉnh thành")

        # Kiểm tra tính hợp lệ sơ bộ
        is_valid = bool(province_code in VN_PROVINCE_CODES and len(corrected) >= 7)

        return {
            "plate_number": plate_number,
            "formatted": formatted,
            "clean_raw": corrected,
            "province": province,
            "province_code": province_code,
            "vehicle_type": vehicle_type,
            "is_valid": is_valid
        }

    @classmethod
    def _format_two_line(cls, line1: str, line2: str) -> tuple[str, str]:
        """Định dạng cho biển 2 dòng (Xe máy hoặc ô tô biển vuông)."""
        # line1: VD '59F1' hoặc '30A'
        # line2: VD '12345' hoặc '8888'
        l1 = line1
        if len(line1) >= 3:
            # Tách mã tỉnh: VD 59-F1 hoặc 30-A
            l1 = f"{line1[:2]}-{line1[2:]}"

        l2 = line2
        # Xử lý trường hợp đuôi 6 số do dấu chấm bị đọc nhầm thành số (VD: 888.88 -> 888288)
        if len(line2) == 6:
            # Loại bỏ ký tự ở vị trí ngăn cách (index 3)
            line2 = line2[:3] + line2[4:]

        if len(line2) == 5:
            # 5 số: 123.45
            l2 = f"{line2[:3]}.{line2[3:]}"
        elif len(line2) == 4:
            # 4 số: 1234
            l2 = line2

        # Phán đoán loại phương tiện:
        # Xe máy thường có 4 ký tự dòng 1 (2 số tỉnh + chữ sê-ri + số sê-ri, VD 59-F1)
        # Ô tô biển vuông thường có 3 ký tự dòng 1 (VD 51-A, 30-C)
        if len(line1) >= 4 and line1[3].isdigit():
            vehicle_type = "Xe máy (Mô tô)"
        else:
            vehicle_type = "Ô tô (Biển vuông) / Xe máy"

        return f"{l1} {l2}", vehicle_type

    @classmethod
    def _format_single_line(cls, text: str) -> tuple[str, str]:
        """Định dạng cho biển 1 dòng (Ô tô biển dài)."""
        # VD: 30A12345 -> 30A-123.45
        # VD: 51K98765 -> 51K-987.65
        n = len(text)
        if n < 7:
            return text, "Không xác định"

        # Tách mã tỉnh (2 số)
        prov = text[:2]

        # Kiểm tra số lượng chữ cái sê-ri (1 chữ cái như 30A hoặc 2 chữ cái như 30LD, 51CD)
        idx = 2
        while idx < n and text[idx].isalpha():
            idx += 1

        series = text[2:idx]
        tail = text[idx:]

        # Xử lý trường hợp đuôi 6 số do dấu chấm bị đọc nhầm thành số
        if len(tail) == 6:
            tail = tail[:3] + tail[4:]

        formatted_tail = tail
        if len(tail) == 5:
            formatted_tail = f"{tail[:3]}.{tail[3:]}"

        formatted = f"{prov}{series}-{formatted_tail}"
        return formatted, "Ô tô (Biển dài)"
