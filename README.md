# HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE THÔNG MINH

Hệ thống nhận diện biển số xe máy và ô tô Việt Nam tự động (**Automatic License Plate Recognition - ALPR**) ứng dụng học sâu (Deep Learning) và thị giác máy tính (Computer Vision).

---
## 🌟 Điểm nổi bật của dự án

- **Phát hiện vùng biển số (License Plate Detection)**
  - Tích hợp mô hình **YOLOv8** chuyên biệt nhận diện biển số xe.
  - Tự động tải mô hình từ Hugging Face nếu chưa có sẵn trong thư mục.

- **Tiền xử lý ảnh thông minh (Image Preprocessing)**
  - Tự động cân chỉnh độ nghiêng.
  - Tăng cường độ tương phản cục bộ và khử nhiễu.

- **Nhận dạng ký tự (OCR Engine)**
  - Nhận diện ký tự chính xác cao bằng **EasyOCR**.
  - Tự động phân tách dòng trên và dòng dưới.

- **Hậu xử lý chuyên biệt cho biển số Việt Nam (Postprocessing)**
  - Tự động sửa lỗi nhầm lẫn OCR theo ngữ cảnh (vị trí số/chữ).
  - Tra cứu mã tỉnh/thành phố trực tiếp (63 tỉnh thành).
  - Phân loại phương tiện (Ô tô biển dài / Ô tô biển vuông / Xe máy).

- **Giao diện Desktop hiện đại (Tkinter GUI)**


---
## 📁 Cấu trúc thư mục

---
## ⚡ Hướng dẫn cài đặt

### 📌 Yêu cầu hệ thống
- Hệ điều hành: Windows, macOS hoặc Linux.
- Đã cài đặt **Python 3.9 đến 3.13** (khuyến nghị tích chọn *"Add Python to PATH"* khi cài đặt).

---

## 🛠️ Quy tắc Nhận diện & Chuẩn hóa Biển số Việt Nam

Hệ thống nhận diện và chuẩn hóa 2 định dạng biển số xe tại Việt Nam:

1. **Biển số Ô tô (Biển 1 dòng - Biển dài):**
   - Cấu trúc: `[Mã tỉnh][Sê-ri] - [Số thứ tự]` (Ví dụ: `30A-888.88`, `51K-678.90`).
   - Tự động xác định: `30` -> Hà Nội, `51` -> TP. Hồ Chí Minh.
2. **Biển số Xe máy & Ô tô biển vuông (Biển 2 dòng):**
   - Dòng 1: `[Mã tỉnh] - [Sê-ri]` (Ví dụ: `59-F1`, `43-B1`).
   - Dòng 2: `[Số thứ tự]` (Ví dụ: `123.45`, `567.89`).
   - Tự động ghép nối các dòng và loại bỏ dấu chấm/nhiễu thừa.

---