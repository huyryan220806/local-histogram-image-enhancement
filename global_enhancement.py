"""
global_enhancement.py - Module tăng cường ảnh TOÀN CỤC dựa trên Histogram Equalization.

Phương pháp:
- Histogram Equalization (HE) cổ điển trên ảnh grayscale
- HE trên ảnh màu thông qua không gian màu YCrCb (chỉ xử lý kênh Y)
"""

import cv2
import numpy as np
from utils import to_grayscale


# ==============================================================================
# HISTOGRAM EQUALIZATION TOÀN CỤC - THỦ CÔNG
# ==============================================================================

def histogram_equalization_manual(gray_img):
    """
    Cân bằng histogram thủ công (không dùng cv2.equalizeHist).

    Thuật toán:
        1. Tính histogram h(k) với k = 0..255
        2. Tính CDF: C(k) = sum(h(i)) for i=0..k
        3. Chuẩn hóa CDF: C_norm(k) = round((C(k) - C_min) / (M*N - C_min) * 255)
        4. Ánh xạ lại mức xám: output(x,y) = C_norm(input(x,y))

    Args:
        gray_img (numpy.ndarray): Ảnh grayscale (2D, uint8).

    Returns:
        numpy.ndarray: Ảnh sau khi cân bằng histogram.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Đầu vào phải là ảnh grayscale 2D.")

    M, N = gray_img.shape
    total_pixels = M * N

    # Bước 1: Tính histogram
    hist = np.zeros(256, dtype=np.float64)
    for k in range(256):
        hist[k] = np.sum(gray_img == k)

    # Bước 2: Tính CDF
    cdf = hist.cumsum()

    # Bước 3: Chuẩn hóa CDF
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Bước 4: Ánh xạ lại mức xám
    output = cdf_normalized[gray_img]

    return output


# ==============================================================================
# HISTOGRAM EQUALIZATION TOÀN CỤC - OPENCV
# ==============================================================================

def histogram_equalization_opencv(gray_img):
    """
    Cân bằng histogram sử dụng hàm cv2.equalizeHist().

    Args:
        gray_img (numpy.ndarray): Ảnh grayscale (2D, uint8).

    Returns:
        numpy.ndarray: Ảnh sau khi cân bằng histogram.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Đầu vào phải là ảnh grayscale 2D.")

    return cv2.equalizeHist(gray_img)


# ==============================================================================
# HISTOGRAM EQUALIZATION CHO ẢNH MÀU
# ==============================================================================

def histogram_equalization_color(bgr_img):
    """
    Cân bằng histogram cho ảnh màu BGR.

    Phương pháp:
        1. Chuyển BGR → YCrCb
        2. Áp dụng HE chỉ trên kênh Y (luminance/độ sáng)
        3. Chuyển ngược YCrCb → BGR

    Lý do: Nếu áp dụng HE riêng trên từng kênh R, G, B sẽ gây sai lệch màu.
    Kênh Y trong YCrCb tách biệt thông tin độ sáng khỏi thông tin màu sắc.

    Args:
        bgr_img (numpy.ndarray): Ảnh màu BGR (3 kênh, uint8).

    Returns:
        numpy.ndarray: Ảnh màu BGR sau khi cân bằng histogram.
    """
    if bgr_img is None or len(bgr_img.shape) != 3:
        raise ValueError("Đầu vào phải là ảnh màu 3 kênh.")

    # Chuyển sang YCrCb
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

    # Tách các kênh
    channels = list(cv2.split(ycrcb))

    # Cân bằng histogram kênh Y (chỉ số 0)
    channels[0] = cv2.equalizeHist(channels[0])

    # Gộp lại và chuyển về BGR
    ycrcb_eq = cv2.merge(channels)
    result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    return result


# ==============================================================================
# HÀM CHÍNH - XỬ LÝ TĂNG CƯỜNG TOÀN CỤC
# ==============================================================================

def apply_global_enhancement(img, method='opencv'):
    """
    Áp dụng tăng cường toàn cục cho ảnh (grayscale hoặc màu).

    Args:
        img (numpy.ndarray): Ảnh đầu vào (BGR hoặc grayscale).
        method (str): Phương pháp - 'opencv' hoặc 'manual'.

    Returns:
        dict: Kết quả gồm:
            - 'gray_original': Ảnh grayscale gốc
            - 'gray_enhanced': Ảnh grayscale sau HE
            - 'color_enhanced': Ảnh màu sau HE (nếu đầu vào là ảnh màu)
    """
    result = {}

    # Chuyển sang grayscale
    gray = to_grayscale(img)
    result['gray_original'] = gray

    # Áp dụng HE trên grayscale
    if method == 'manual':
        result['gray_enhanced'] = histogram_equalization_manual(gray)
    else:
        result['gray_enhanced'] = histogram_equalization_opencv(gray)

    # Áp dụng HE trên ảnh màu (nếu có)
    if len(img.shape) == 3:
        result['color_enhanced'] = histogram_equalization_color(img)

    return result
