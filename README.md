# HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE THÔNG MINH

Hệ thống nhận diện biển số xe máy và ô tô Việt Nam tự động (**Automatic License Plate Recognition - ALPR**) ứng dụng học sâu (Deep Learning) và thị giác máy tính (Computer Vision).

---
## 🌟 Điểm nổi bật của dự án

- **Phát hiện vùng biển số (License Plate Detection)**
  - Tích hợp mô hình **YOLOv8** chuyên biệt nhận diện biển số xe.
  - Tự động tải mô hình từ Hugging Face nếu chưa có sẵn trong thư mục.
  - Thuật toán dự phòng **OpenCV Morphological & Contour Filter** khi góc chụp phức tạp.
  - Khử trùng lặp box và tự động bắt trọn vẹn biển số 2 dòng xe máy.

- **Tiền xử lý ảnh thông minh (Image Preprocessing)**
  - Tự động cân chỉnh độ nghiêng (*Deskew*).
  - Tăng cường độ tương phản cục bộ (*CLAHE*) và khử nhiễu (*Bilateral Filter*).
  - Nhị phân hóa thích ứng (*Otsu Threshold*).

- **Nhận dạng ký tự (OCR Engine)**
  - Nhận diện ký tự chính xác cao bằng **EasyOCR**.
  - Tự động phân tách dòng trên và dòng dưới.

- **Hậu xử lý chuyên biệt cho biển số Việt Nam (Postprocessing)**
  - Tự động sửa lỗi nhầm lẫn OCR theo ngữ cảnh (vị trí số/chữ).
  - Tra cứu mã tỉnh/thành phố trực tiếp (63 tỉnh thành).
  - Phân loại phương tiện (Ô tô biển dài / Ô tô biển vuông / Xe máy).

- **Giao diện Desktop hiện đại (Tkinter GUI)**
  - Xem ảnh, video, nhận diện trực tiếp qua **Webcam/Camera**.
  - Xem ảnh phóng to vùng cắt biển số (*Crop*) và ảnh nhị phân.
  - Quản lý lịch sử nhận diện và xuất dữ liệu ra file **CSV/Excel**.

---
## 📁 Cấu trúc thư mục

```text
DOC_BIEN_SO_XE/
├── models/
│   ├── license_plate_detector.pt # Trọng số mô hình YOLOv8 (6.2 MB)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── app_gui.py                 # Giao diện đồ họa Desktop (Tkinter)
│   ├── config.py                  # Cấu hình ngưỡng & từ điển 63 tỉnh thành
│   ├── detector.py                # Bộ phát hiện biển số (YOLOv8 + OpenCV fallback)
│   ├── logic.py                   # Chuẩn hóa biển số xe VN & sửa lỗi ngữ cảnh
│   ├── main.py                    # Trình khởi chạy đa năng (CLI/GUI/Webcam/Video)
│   ├── ocr_engine.py              # Nhận dạng ký tự (EasyOCR)
│   ├── pipeline.py                # ALPR Pipeline xâu chuỗi toàn bộ quá trình
│   ├── preprocessor.py            # Tiền xử lý (xoay thẳng, CLAHE, nhị phân hóa)
│   └── visualizer.py              # Vẽ bounding box & nhãn tiếng Việt Unicode
├── samples/                       # Ảnh mẫu xe ô tô và xe máy (anh1.jpg -> anh5.jpg)
├── output/                        # Thư mục lưu kết quả nhận diện
│   └── .gitkeep
├── generate_samples.py            # Script sinh dữ liệu ảnh mẫu
├── run_app.bat                    # File khởi chạy tự động 1-click trên Windows
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── .gitignore                     # Cấu hình bỏ qua file rác
└── README.md                      # Tài liệu hướng dẫn sử dụng
```

---
## ⚡ Hướng dẫn cài đặt

### 📌 Yêu cầu hệ thống
- Hệ điều hành: Windows, macOS hoặc Linux.
- Đã cài đặt **Python 3.9 đến 3.13** (khuyến nghị tích chọn *"Add Python to PATH"* khi cài đặt).

---

### Cách 1: Khởi chạy 1-Click trên Windows (Đơn giản nhất 🚀)
Nếu bạn dùng Windows, chỉ cần **nhấp đúp chuột vào file `run_app.bat`**:
- Script sẽ tự động tạo môi trường ảo `.venv` (nếu chưa có).
- Tự động cài đặt đầy đủ các thư viện phụ thuộc từ `requirements.txt`.
- Tự động mở giao diện phần mềm!

---

### Cách 2: Cài đặt và chạy bằng dòng lệnh (Dành cho mọi hệ điều hành)

#### Bước 1: Clone dự án về máy tính
```bash
git clone https://github.com/SinhhDev/DOC_BIEN_SO_XE
cd DOC_BIEN_SO_XE
```

#### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
- **Trên Windows:**
  ```terminal
  python -m venv .venv
  source .venv/Scripts/activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

#### Bước 4: Khởi chạy phần mềm

**Mở giao diện người dùng đồ họa (GUI Desktop):**
```bash
python src/main.py
```

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