# HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE THÔNG MINH (ALPR)
### 🚗 Automatic License Plate Recognition System for Vietnam Vehicles 🇻🇳

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OCR](https://img.shields.io/badge/OCR-EasyOCR-green.svg)](https://github.com/JaidedAI/EasyOCR)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%204.8+-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Hệ thống nhận diện biển số xe máy và ô tô Việt Nam tự động (**Automatic License Plate Recognition - ALPR**) ứng dụng mô hình học sâu (**YOLOv8**), công nghệ nhận dạng ký tự quang học (**EasyOCR**) và thị giác máy tính (**OpenCV**), đi kèm giao diện đồ họa Desktop hiện đại (**Tkinter GUI**) và giao diện dòng lệnh (**CLI**) linh hoạt.

---

## 📑 Mục lục
1. [Điểm nổi bật của dự án](#-điểm-nổi-bật-của-dự-án)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
   - [Cách 1: Khởi chạy 1-Click trên Windows (Khuyến nghị)](#cách-1-khởi-chạy-1-click-trên-windows-khuyến-nghị-)
   - [Cách 2: Cài đặt & chạy qua dòng lệnh (Windows / macOS / Linux)](#cách-2-cài-đặt--chạy-qua-dòng-lệnh-windows--macos--linux)
4. [Hướng dẫn sử dụng chi tiết](#-hướng-dẫn-sử-dụng-chi-tiết)
5. [Quy chuẩn biển số xe & Thuật toán sửa lỗi OCR](#-quy-chuẩn-biển-số-xe--thuật-toán-sửa-lỗi-ocr)

---

## 🌟 Điểm nổi bật của dự án

- **🎯 Phát hiện vùng biển số chính xác cao (License Plate Detection)**
  - Tích hợp mô hình học sâu **YOLOv8** chuyên biệt nhận diện biển số xe trong đa dạng điều kiện môi trường.
  - Tự động kiểm tra và tải trọng số mô hình từ **Hugging Face Hub** nếu máy tính chưa có sẵn.
  - Thuật toán dự phòng tự động **OpenCV Morphological & Contour Filter** khi góc chụp phức tạp hoặc khuất sáng.
  - Khử trùng lặp box (Non-Maximum Suppression) và tự động nhận diện trọn vẹn cả biển số 1 dòng và 2 dòng.

- **🔬 Tiền xử lý ảnh thông minh (Image Preprocessing)**
  - Tự động cân chỉnh góc nghiêng (*Deskew*) giúp ký tự luôn thẳng hàng trước khi đưa vào OCR.
  - Tăng cường độ tương phản cục bộ thích ứng (*CLAHE*) và khử nhiễu (*Bilateral Filter*).
  - Nhị phân hóa thích ứng (*Otsu Threshold*) tối ưu độ tương phản giữa chữ và nền biển số.

- **🔤 Nhận dạng ký tự chuyên sâu (OCR Engine)**
  - Nhận diện ký tự chính xác cao với bộ bóc tách ngôn ngữ tiếng Anh / số từ **EasyOCR**.
  - Tự động gom nhóm và phân tách dòng trên/dòng dưới dựa trên tọa độ hình học `y-center`.

- **🇻🇳 Hậu xử lý chuẩn hóa biển số Việt Nam (Post-processing Logic)**
  - Tự động sửa lỗi nhầm lẫn OCR theo vị trí ngữ cảnh (ví dụ: nhầm `0` thành `O/D`, nhầm `8` thành `B`, nhầm `1` thành `I/L`, v.v.).
  - Tự động tra cứu mã tỉnh/thành phố trực tiếp (bao trọn 63 tỉnh thành phố Việt Nam và Cục CSGT/Bộ ngành).
  - Phân loại phương tiện tự động: **Ô tô biển dài**, **Ô tô biển vuông**, **Xe máy**.

- **🖥️ Giao diện Desktop hiện đại (Modern Tkinter GUI)**
  - Giao diện chủ đề tối (Dark Theme) trực quan, hiển thị nhanh và tối ưu hóa luồng đa nhiệm (`threading`).
  - Hỗ trợ tải ảnh từ máy tính, chọn ảnh mẫu có sẵn, mở file video và phát hiện thời gian thực qua **Webcam/Camera**.
  - Xem chi tiết: ảnh gốc có bounding box, vùng cắt biển số (*Crop*), ảnh nhị phân (*Binary*).
  - Bảng lịch sử nhận diện chi tiết (STT, Thời gian, Biển số, Loại xe, Tỉnh thành, Độ tin cậy) kèm tính năng **Xuất báo cáo CSV/Excel**.

---

## 📁 Cấu trúc thư mục

```text
DOC_BIEN_SO_XE/
├── models/
│   ├── license_plate_detector.pt  # Trọng số mô hình YOLOv8 (~6.2 MB)
│   └── .gitkeep
├── src/
│   ├── __init__.py                # Khởi tạo package src & export components
│   ├── app_gui.py                 # Giao diện đồ họa Desktop (Tkinter UI)
│   ├── config.py                  # Cấu hình ngưỡng phát hiện & từ điển 63 tỉnh thành
│   ├── detector.py                # Bộ phát hiện biển số (YOLOv8 + OpenCV fallback)
│   ├── logic.py                   # Chuẩn hóa biển số VN & sửa lỗi nhầm OCR
│   ├── main.py                    # Trình chạy đa năng (CLI / GUI / Webcam / Video)
│   ├── ocr_engine.py              # Bộ nhận dạng ký tự (EasyOCR)
│   ├── pipeline.py                # ALPR Pipeline kết nối toàn bộ quy trình
│   ├── preprocessor.py            # Tiền xử lý (xoay thẳng, CLAHE, nhị phân hóa)
│   └── visualizer.py              # Vẽ bounding box & nhãn tiếng Việt Unicode
├── samples/                       # Thư mục ảnh mẫu xe ô tô và xe máy
├── output/                        # Thư mục lưu ảnh và video kết quả nhận diện
│   └── .gitkeep
├── scripts/
│   └── generate_samples.py        # Script sinh dữ liệu ảnh mẫu tự động
├── run_app.bat                    # File khởi chạy 1-click tự động trên Windows
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── .gitignore                     # Cấu hình loại bỏ file rác khi đẩy Git
└── README.md                      # Tài liệu hướng dẫn sử dụng dự án
```

---

## ⚡ Hướng dẫn cài đặt

### 📌 Yêu cầu hệ thống
- Hệ điều hành: **Windows 10/11**, **macOS** hoặc **Linux**.
- Đã cài đặt **Python 3.9 đến 3.13** (khuyến nghị tích chọn *"Add Python to PATH"* khi cài đặt trên Windows).
- *(Tùy chọn)* Card đồ họa NVIDIA với **CUDA** nếu muốn tăng tốc nhận diện thời gian thực.

---

### Cách 1: Khởi chạy 1-Click trên Windows (Khuyến nghị 🚀)

Nếu sử dụng Windows, bạn chỉ cần **nhấp đúp chuột vào file `run_app.bat`**:
- Script sẽ tự động phát hiện và khởi tạo môi trường ảo `.venv` (nếu chưa có).
- Tự động cập nhật `pip` và tải đầy đủ thư viện phụ thuộc từ `requirements.txt`.
- Tự động mở giao diện ứng dụng!

---

### Cách 2: Cài đặt & chạy qua dòng lệnh (Windows / macOS / Linux)

#### Bước 1: Clone dự án về máy
```bash
git clone https://github.com/SinhhDev/DOC_BIEN_SO_XE.git
cd DOC_BIEN_SO_XE
```

#### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
- **Trên Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Hướng dẫn sử dụng chi tiết

###  Sử dụng Giao diện Desktop (GUI)
Để khởi chạy ứng dụng với giao diện đồ họa trực quan:
```bash
python src/main.py --gui
```
*(hoặc chạy `python src/main.py` không truyền tham số, ứng dụng sẽ mặc định mở GUI).*

**Các thao tác trên giao diện:**
1. **Mở ảnh từ máy:** Nhấn nút *"Chọn ảnh"* để chọn ảnh xe từ máy tính.
2. **Chọn ảnh mẫu:** Nhấp vào danh sách ảnh có sẵn trong thư mục `samples/`.
3. **Mở Webcam/Camera:** Nhấn *"Bật Webcam"* để bắt đầu nhận diện biển số liên tục từ camera.
4. **Xem chi tiết:** Quan sát khung hình chính, vùng biển số cắt phóng to (*Crop*) và ảnh nhị phân (*Binary*).
5. **Xuất báo cáo:** Nhấn *"Xuất file CSV"* để lưu lại toàn bộ bảng lịch sử nhận diện phục vụ thống kê/quản lý bãi xe.

---

## 🛠️ Quy chuẩn biển số xe & Thuật toán sửa lỗi OCR

Hệ thống được thiết kế chuyên biệt để chuẩn hóa và phân loại chính xác các định dạng biển số lưu hành tại Việt Nam:

### 1. Định dạng biển số
| Loại biển số | Cấu trúc chuẩn | Ví dụ | Quy tắc hiển thị |
| :--- | :--- | :--- | :--- |
| **Ô tô biển dài (1 dòng)** | `[Mã tỉnh][Sê-ri] - [Số thứ tự]` | `30A-888.88`, `51K-678.90` | Ghép liền mạch, tự động nhận biết xe con/xe tải |
| **Xe máy & Ô tô vuông (2 dòng)** | Dòng 1: `[Mã tỉnh] - [Sê-ri]`<br/>Dòng 2: `[Số thứ tự]` | Dòng 1: `59-F1`<br/>Dòng 2: `123.45` | Tự động phân cụm theo trục tung (`y-center`) |

### 2. Thuật toán sửa lỗi nhầm lẫn OCR theo ngữ cảnh
Mô hình OCR thường hay nhầm lẫn giữa các ký tự có hình dạng tương đồng. Hệ thống áp dụng ma trận sửa lỗi dựa trên vị trí chuẩn của biển số:
- **Tại vị trí số:** Nếu OCR đọc thành chữ cái, tự động chuyển về chữ số tương ứng:
  - `O`, `D`, `Q`, `U` $\rightarrow$ `0`
  - `I`, `L`, `|`, `T` $\rightarrow$ `1`
  - `Z` $\rightarrow$ `2` | `A` $\rightarrow$ `4` | `S` $\rightarrow$ `5` | `G`, `b` $\rightarrow$ `6` | `B` $\rightarrow$ `8` | `P`, `q` $\rightarrow$ `9`
- **Tại vị trí chữ cái sê-ri:** Nếu OCR đọc nhầm thành số, tự động chuyển về chữ cái:
  - `0` $\rightarrow$ `D` | `1` $\rightarrow$ `I` | `2` $\rightarrow$ `Z` | `4` $\rightarrow$ `A` | `5` $\rightarrow$ `S` | `6` $\rightarrow$ `G` | `8` $\rightarrow$ `B`

### 3. Tự động tra cứu 63 tỉnh thành phố
Hệ thống tích hợp từ điển tra cứu mã đầu số giao thông vận tải Việt Nam:
- `29, 30, 31, 32, 33` $\rightarrow$ Hà Nội
- `50, 51, 52, 53, 54, 55, 56, 57, 58, 59` $\rightarrow$ TP. Hồ Chí Minh
- `43` $\rightarrow$ Đà Nẵng | `15, 16` $\rightarrow$ Hải Phòng | `60` $\rightarrow$ Đồng Nai | `61` $\rightarrow$ Bình Dương...
- `80` $\rightarrow$ Cục Cảnh sát Giao thông / Cơ quan trực thuộc Trung ương.
