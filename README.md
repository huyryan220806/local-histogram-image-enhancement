# Đồ Án Cuối Kỳ: Tăng Cường Ảnh Cục Bộ Dựa Trên Histogram

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.2-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.4-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.7-11557C?style=for-the-badge)](https://matplotlib.org/)
[![Git](https://img.shields.io/badge/Git-2.51.2-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)

---

## 📌 Thông Tin Đồ Án

* **Môn học**: Nhập Môn Xử Lý Ảnh Số (Digital Image Processing)
* **Lớp học phần**: `253_71ITAI40803_01`
* **Giảng viên hướng dẫn**: **TS. Vũ Thanh Hiền**
* **Trường**: Đại học Văn Lang (VLU)
* **Chủ đề 6**: **"Tăng cường ảnh cục bộ dựa trên histogram"**

---

## 👥 Thành Viên Nhóm & Phân Công Nhiệm Vụ

| STT | Họ và Tên | MSSV | Vai Trò | Nội Dung Công Việc Phụ Trách | Hoàn Thành |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | **Nguyễn Đình Huy** | `2474802010140` | Nhóm trưởng | Nghiên cứu Global HE, thiết kế kiến trúc phần mềm, lập trình module `global_enhancement.py`, `comparison.py` (tính 5 metrics), phát triển Web App Streamlit (`app.py`) và tổng hợp Báo cáo. | 100% |
| **2** | **Lê Quyết Tiến** | `2474802010386` | Thành viên | Nghiên cứu CLAHE cục bộ, lập trình module `local_enhancement.py` (CLAHE & khảo sát đa tham số), chạy thực nghiệm thu thập số liệu, tối ưu UI Streamlit và thiết kế Slide. | 100% |

---

## 🛠️ Danh Mục Công Cụ & Thư Viện Sử Dụng (Tech Stack)

| Logo / Công cụ | Tên Thư Viện & Công Cụ | Phiên Bản | Vai Trò & Chức Năng Trong Dự Án |
| :---: | :--- | :---: | :--- |
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | **Python** | `3.13.0` | Ngôn ngữ lập trình chính triển khai toàn bộ hệ thống xử lý ảnh và giao diện web. |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | **OpenCV (opencv-python)** | `4.13.0` | Xử lý biến đổi ma trận ảnh, chuyển đổi không gian màu YCrCb, cân bằng HE và CLAHE. |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | **Streamlit** | `1.59.2` | Framework xây dựng giao diện ứng dụng Web tương tác người dùng (`app.py`). |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | **NumPy** | `2.3.4` | Tính toán đại số tuyến tính, biến đổi ma trận ảnh nhị phân và đọc/ghi file Unicode. |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) | **Matplotlib** | `3.10.7` | Trực quan hóa biểu đồ Histogram, đường cong tích lũy CDF và biểu đồ cột so sánh metrics. |
| ![VS Code](https://img.shields.io/badge/-VS_Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white) | **Visual Studio Code** | `1.95+` | Môi trường phát triển tích hợp (IDE) quản lý mã nguồn và Terminal. |
| ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) | **Git & GitHub** | `2.51.2` | Quản lý phiên bản mã nguồn (VCS) và lưu trữ repository trực tuyến. |

---

## 📂 Cấu Trúc Mã Nguồn Project

```text
DoAnCK/
├── input/                      # Thư mục chứa ảnh đầu vào
│   └── MU_Trident.jpg          # Ảnh thử nghiệm chính
├── output/                     # Kết quả xuất tự động
│   ├── global/                 # Kết quả ảnh Global HE
│   ├── local/                  # Kết quả ảnh CLAHE
│   ├── comparison/             # Ảnh so sánh side-by-side & thử nghiệm tham số
│   ├── histograms/             # Biểu đồ Histogram & đường CDF
│   ├── metrics/                # Bảng chỉ số & biểu đồ tổng hợp
│   └── bao_cao_ket_qua.txt     # Báo cáo kết quả định lượng dạng text
├── utils.py                    # Đọc/ghi ảnh Unicode (tiếng Việt), tính histogram, CDF
├── global_enhancement.py       # Cân bằng Histogram toàn cục (Global HE manual & OpenCV)
├── local_enhancement.py        # Cân bằng Histogram cục bộ (CLAHE manual & OpenCV, khảo sát tham số)
├── comparison.py               # Tính 5 chỉ số metrics (Mean, Std, Entropy, PSNR, SSIM) & vẽ biểu đồ
├── report_generator.py         # Tạo bảng chỉ số định lượng và biểu đồ tổng hợp
├── main.py                     # Script điều khiển chính (Pipeline chạy batch tự động)
├── app.py                      # Giao diện Web tương tác Streamlit
├── run_app.bat                 # File nhấp đúp chạy nhanh ứng dụng Web Streamlit
├── .gitignore                  # Cấu hình bỏ qua các file tạm
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt môi trường
Yêu cầu **Python 3.10 trở lên**. Cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
pip install numpy opencv-python matplotlib streamlit
```

---

### 2. Khởi chạy Giao diện Web (Streamlit)

#### **Cách 1: Khởi chạy từ Terminal (Khuyên dùng)**
Mở Terminal tại thư mục dự án và gõ lệnh:

```bash
streamlit run app.py
```
*(Hoặc dùng lệnh đầy đủ: `py -3.13 -m streamlit run app.py`)*

#### **Cách 2: Nhấp đúp chuột**
Nhấp đúp chuột vào file **`run_app.bat`** trong thư mục dự án.

Ứng dụng web sẽ tự động mở tại địa chỉ: **`http://localhost:8501`**

---

### 3. Chạy Pipeline Batch (Command Line)
Để xử lý hàng loạt tất cả ảnh trong thư mục `input/` và xuất kết quả ra thư mục `output/`:

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 main.py
```

---

## 📊 Kết Quả Đo Đạc Định Lượng Thực Nghiệm (`MU_Trident.jpg`)

Dưới đây là bảng kết quả đo đạc các chỉ số định lượng giữa hai phương pháp trên ảnh thử nghiệm **`MU_Trident.jpg`**:

| Chỉ số Đánh Giá (Metric) | Ảnh Gốc | Global HE | CLAHE | Nhận Xét Đánh Giá |
| :--- | :---: | :---: | :---: | :--- |
| **Mean (Độ sáng TB)** | 119.2 | 128.4 | 126.3 | Cả 2 phương pháp giúp cân bằng lại dải sáng trung bình. |
| **Std Dev (Độ tương phản)** | 56.0 | 71.5 | 63.5 | Global HE tăng tương phản quá gắt; CLAHE giữ mức hài hòa. |
| **Entropy (Lượng thông tin)** | 7.78 | 7.75 | **7.92** | **CLAHE vượt trội** (phong phú và nổi bật chi tiết nhất). |
| **PSNR (dB)** | — | **22.12 dB** | 17.75 dB | Global HE giữ khoảng cách bình phương nhỏ hơn so với ảnh gốc. |
| **SSIM (Tương đồng cấu trúc)** | 1.0000 | **0.9433** | 0.8663 | Global HE giữ cấu trúc phẳng; CLAHE làm nổi bật dải cục bộ. |

---

## 📈 Ưu Điểm & Hạn Chế Của Các Thuật Toán

### 1. Global Histogram Equalization (Global HE)
* **Ưu điểm**: Thuật toán đơn giản, tốc độ tính toán cực nhanh, không cần điều chỉnh tham số đầu vào.
* **Hạn chế**: Thất bại trên ảnh có độ phân bổ ánh sáng không đồng đều; dễ gây cháy sáng ở vùng đã sáng và làm nổi hạt nhiễu trên các vùng phẳng.

### 2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
* **Ưu điểm**:
  * Cải thiện chi tiết cục bộ vượt trội tại các vùng tối hoặc bóng râm.
  * Cơ chế **Clip Limit** giới hạn ngưỡng cắt dải histogram giúp ngăn ngừa khuếch đại nhiễu.
  * **Nội suy song tuyến tính (Bilinear Interpolation)** xóa bỏ triệt để hiệu ứng phân khối (block artifacts) giữa các tile.
  * Đạt độ entropy lượng thông tin cao nhất ($7.92$).
* **Hạn chế**: Chi phí tính toán lớn hơn Global HE; cần lựa chọn các tham số (`clipLimit`, `tileGridSize`) phù hợp cho từng loại ảnh.

---

## 🎓 Trường Đại học Văn Lang (VLU) — Khoa Công Nghệ Thông Tin
* **Giảng viên hướng dẫn**: TS. Vũ Thanh Hiền
* **Sinh viên thực hiện**: Nguyễn Đình Huy & Lê Quyết Tiến
