# Đồ Án Cuối Kỳ: Tăng Cường Ảnh Cục Bộ Dựa Trên Histogram

Đồ án môn học **Nhập Môn Xử Lý Ảnh Số** với đề tài: **"Tăng cường ảnh cục bộ dựa trên histogram"**, nhằm mục tiêu cải thiện chất lượng chi tiết cục bộ trong các hình ảnh có độ sáng hoặc độ tương phản không đồng đều.

Dự án triển khai, so sánh và đánh giá hai thuật toán:
1. **Global Histogram Equalization (HE)** — Cân bằng histogram toàn cục.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)** — Cân bằng histogram cục bộ thích ứng giới hạn độ tương phản (cả phiên bản tự cài đặt và phiên bản OpenCV).

---

## Thông Tin Lớp Học Phần

| | |
| :--- | :--- |
| **Lớp học phần** | 253_71ITAI40803_01 |
| **Môn học** | Nhập Môn Xử Lý Ảnh Số |
| **GVHD** | TS. Vũ Thanh Hiền |

---

## Thành Viên Nhóm

| STT | Họ và tên | MSSV |
| :---: | :--- | :--- |
| 1 | Nguyễn Đình Huy | 2474802010140 |
| 2 | Lê Quyết Tiến | 2474802010386 |

**Trường Đại học Văn Lang (VLU)**

---

## Cấu Trúc Mã Nguồn

| File | Chức năng |
| :--- | :--- |
| `utils.py` | Các hàm bổ trợ: Đọc/ghi ảnh Unicode (hỗ trợ tiếng Việt), vẽ histogram, hiển thị ảnh. |
| `global_enhancement.py` | Triển khai cân bằng histogram toàn cục (Global HE) tự thiết kế và OpenCV trên ảnh xám và ảnh màu. |
| `local_enhancement.py` | Triển khai thuật toán CLAHE tự thiết kế và OpenCV trên ảnh xám và ảnh màu. Thử nghiệm thay đổi tham số. |
| `comparison.py` | Tính toán 5 chỉ số đánh giá (Mean, Std Dev, Entropy, PSNR, SSIM) và vẽ các biểu đồ so sánh. |
| `report_generator.py` | Xuất báo cáo tổng kết, bảng chỉ số định lượng, và biểu đồ so sánh tổng hợp dưới dạng hình ảnh và văn bản. |
| `main.py` | File khởi chạy chính của toàn bộ hệ thống xử lý ảnh (chạy batch trên thư mục). |
| `app.py` | Giao diện web Streamlit cho phép upload ảnh và tương tác trực tiếp. |

---

## Giao Diện Web (Streamlit)

Dự án có tích hợp giao diện web sử dụng **Streamlit**, cho phép người dùng:

- **Upload ảnh** từ máy tính (JPG, PNG, BMP, TIFF)
- **So sánh trực quan**: Ảnh gốc vs Global HE vs CLAHE cạnh nhau
- **Điều chỉnh tham số**: clipLimit và tileGridSize bằng slider
- **Xem Histogram và CDF**: Biểu đồ phân bố mức xám
- **Xem Metrics**: PSNR, SSIM, Entropy, Std Dev với nhận xét tự động
- **Thử nghiệm tham số**: Khám phá ảnh hưởng của nhiều bộ tham số CLAHE khác nhau
- **Tải xuống**: Lưu ảnh kết quả và biểu đồ về máy

### Chạy giao diện Streamlit

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại: **http://localhost:8501**

---

## Hướng Dẫn Cài Đặt và Chạy

### 1. Yêu cầu hệ thống
* **Python**: Khuyến nghị phiên bản 3.10 trở lên.
* **Thư viện cần cài đặt**:
  ```bash
  pip install numpy opencv-python matplotlib streamlit
  ```

### 2. Chạy xử lý batch (command line)
1. Đặt các ảnh đầu vào cần tăng cường vào thư mục `input/`.
2. Mở terminal tại thư mục dự án và chạy:
   ```bash
   python main.py
   ```
3. Xem kết quả được tạo tự động tại thư mục `output/`.

### 3. Chạy giao diện web (Streamlit)
```bash
streamlit run app.py
```

---

## Kết Quả So Sánh Định Lượng

Dưới đây là kết quả đo đạc trung bình trên bộ dữ liệu ảnh thử nghiệm:

| Chỉ số | Ảnh gốc | Global HE | CLAHE | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| **PSNR (dB)** | — | 11.19 | **35.25** | CLAHE giữ độ tương đồng cao với ảnh gốc, tránh méo ảnh. |
| **SSIM** | 1.0000 | 0.7361 | **0.9906** | SSIM của CLAHE gần tuyệt đối (~0.99) — bảo toàn cấu trúc rất tốt. |
| **Entropy** | 1.34 | 1.47 | **1.82** | CLAHE tăng lượng thông tin chi tiết đáng kể so với ảnh gốc và Global HE. |
| **Độ lệch chuẩn** | 35.8 | 87.3 | 44.3 | Global HE làm tăng quá mức độ tương phản dẫn đến cháy sáng. |

---

## Đánh Giá Ưu và Nhược Điểm

### 1. Global Histogram Equalization (Toàn cục)
* **Ưu điểm**: Cực kỳ đơn giản, tốc độ tính toán nhanh, không cần tinh chỉnh tham số đầu vào.
* **Nhược điểm**:
  * Không hiệu quả đối với các ảnh có độ phân bổ ánh sáng không đồng đều.
  * Dễ xảy ra hiện tượng cháy sáng ở vùng sáng và mất chi tiết ở vùng tối.
  * Khuếch đại cả các hạt nhiễu (noise).

### 2. CLAHE (Cục bộ)
* **Ưu điểm**:
  * Cải thiện chi tiết cục bộ vô cùng hiệu quả, xử lý xuất sắc các vùng bóng tối hoặc chói sáng cục bộ.
  * Giới hạn độ tương phản (clipLimit) giúp hạn chế tối đa việc khuếch đại nhiễu.
  * Nội suy song tuyến tính (Bilinear Interpolation) loại bỏ các đường biên phân mảnh khối (block artifacts).
* **Nhược điểm**:
  * Thuật toán phức tạp hơn, tốn nhiều chi phí tính toán hơn.
  * Cần lựa chọn các tham số phù hợp (clipLimit và tileGridSize) để đạt hiệu quả tối ưu cho từng loại ảnh.
