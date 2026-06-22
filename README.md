# Đồ Án Cuối Kỳ: Tăng Cường Ảnh Cục Bộ Dựa Trên Histogram (Local Image Enhancement Based on Histogram)

Dự án này là đồ án môn học **Nhập Môn Xử Lý Ảnh Số** với đề tài: **"Tăng cường ảnh cục bộ dựa trên histogram"** nhắm tới mục tiêu cải thiện chất lượng chi tiết cục bộ trong các hình ảnh có độ sáng hoặc độ tương phản không đồng đều.

Dự án triển khai, so sánh và đánh giá hai thuật toán:
1. **Global Histogram Equalization (HE)**: Cân bằng histogram toàn cục.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Cân bằng histogram cục bộ thích ứng giới hạn độ tương phản (cả phiên bản tự cài đặt và phiên bản OpenCV).

---

## Cấu Trúc Mã Nguồn

| File | Chức năng |
| :--- | :--- |
| [utils.py](utils.py) | Các hàm bổ trợ: Đọc/ghi ảnh Unicode (hỗ trợ tiếng Việt), vẽ histogram, hiển thị ảnh. |
| [global_enhancement.py](global_enhancement.py) | Triển khai cân bằng histogram toàn cục (Global HE) tự thiết kế & OpenCV trên ảnh xám và ảnh màu. |
| [local_enhancement.py](local_enhancement.py) | Triển khai thuật toán CLAHE tự thiết kế & OpenCV trên ảnh xám và ảnh màu. Thử nghiệm thay đổi tham số. |
| [comparison.py](comparison.py) | Tính toán 5 chỉ số đánh giá (Mean, Std Dev, Entropy, PSNR, SSIM) và vẽ các biểu đồ so sánh. |
| [report_generator.py](report_generator.py) | Xuất báo cáo tổng kết, bảng chỉ số định lượng, và biểu đồ so sánh tổng hợp dưới dạng hình ảnh và văn bản. |
| [main.py](main.py) | File khởi chạy chính của toàn bộ hệ thống xử lý ảnh. |

---

## Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu cầu hệ thống
* **Python**: Khuyến nghị phiên bản 3.10 trở lên.
* **Thư viện cần cài đặt**:
  ```bash
  pip install numpy opencv-python matplotlib
  ```

### 2. Chạy chương trình
1. Đặt các ảnh đầu vào cần tăng cường vào thư mục `input/` hoặc `Tăng Cường Ảnh/`.
2. Mở terminal tại thư mục dự án và chạy lệnh sau:
   * Trên Windows (sử dụng launcher):
     ```powershell
     $env:PYTHONIOENCODING='utf-8'
     py -3.13 main.py
     ```
   * Hoặc lệnh tiêu chuẩn:
     ```bash
     python main.py
     ```
3. Xem kết quả được tạo tự động tại thư mục `output/`.

---

## Kết Quả So Sánh Định Lượng

Dưới đây là kết quả đo đạc trung bình trên bộ dữ liệu ảnh thử nghiệm:

| Chỉ số (Metric) | Ảnh Gốc | Global HE | CLAHE | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| **PSNR (dB)** | - | 11.19 | **35.25** | CLAHE giữ độ tương đồng cao với ảnh gốc, tránh méo ảnh. |
| **SSIM** | 1.0000 | 0.7361 | **0.9906** | SSIM của CLAHE gần tuyệt đối (~0.99) - bảo toàn cấu trúc rất tốt. |
| **Entropy** | 1.34 | 1.47 | **1.82** | CLAHE tăng lượng thông tin chi tiết đáng kể so với ảnh gốc & Global HE. |
| **Độ lệch chuẩn (Std)** | 35.8 | 87.3 | 44.3 | Global HE làm tăng quá mức độ tương phản dẫn đến cháy sáng. |

---

## Đánh Giá Ưu & Nhược Điểm

### 1. Global Histogram Equalization (Toàn cục)
* **Ưu điểm**: Cực kỳ đơn giản, tốc độ tính toán nhanh, không cần tinh chỉnh tham số đầu vào.
* **Nhược điểm**: 
  * Không hiệu quả đối với các ảnh có độ phân bổ ánh sáng không đồng đều (ví dụ: một nửa quá sáng, một nửa quá tối).
  * Dễ xảy ra hiện tượng cháy sáng ở vùng sáng và mất chi tiết ở vùng tối.
  * Khuếch đại cả các hạt nhiễu (noise).

### 2. CLAHE (Cục bộ)
* **Ưu điểm**:
  * Cải thiện chi tiết cục bộ vô cùng hiệu quả, xử lý xuất sắc các vùng bóng tối hoặc chói sáng cục bộ.
  * Giới hạn độ tương phản (`clipLimit`) giúp hạn chế tối đa việc khuếch đại nhiễu.
  * Nội suy song tuyến tính (Bilinear Interpolation) loại bỏ các đường biên phân mảnh khối (block artifacts).
* **Nhược điểm**:
  * Thuật toán phức tạp hơn, tốn nhiều chi phí tính toán hơn.
  * Cần lựa chọn các tham số phù hợp (`clipLimit` và `tileGridSize`) để đạt hiệu quả tối ưu cho từng loại ảnh.

---

### Tác giả
* Họ và tên: **Nguyễn Đình Huy**
* Mã số sinh viên: **2474802010140**
* Khoa: Công Nghệ Thông Tin
* Trường Đại học Văn Lang (VLU)
### Đồng tác giả
* Họ và tên: **Lê Quyết Tiến**
* Mã số sinh viên: **2474802010386**
* Khoa: Công Nghệ Thông Tin
* Trường Đại học Văn Lang (VLU)
