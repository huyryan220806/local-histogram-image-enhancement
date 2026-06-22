"""
local_enhancement.py - Module tăng cường ảnh CỤC BỘ dựa trên CLAHE.

Phương pháp:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
  + Chia ảnh thành các tile (block) nhỏ
  + Cân bằng histogram riêng cho từng tile
  + Giới hạn clip limit để tránh khuếch đại nhiễu (noise)
  + Nội suy bilinear tại biên các tile để tránh hiệu ứng khối (block artifact)
"""

import cv2
import numpy as np
from utils import to_grayscale


# ==============================================================================
# CLAHE - THỦ CÔNG (SIMPLIFIED)
# ==============================================================================

def clahe_manual_simplified(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    """
    Minh họa nguyên lý CLAHE đơn giản hóa.

    Thuật toán:
        1. Chia ảnh thành các block (tile) kích thước bằng nhau
        2. Với mỗi tile:
           a. Tính histogram
           b. Cắt ngưỡng (clip) histogram: giới hạn giá trị tối đa mỗi bin
           c. Phân phối lại phần bị cắt cho tất cả các bin
           d. Tính CDF và cân bằng histogram cục bộ
        3. Nội suy bilinear kết quả giữa các tile lân cận

    Note: Phiên bản đơn giản hóa, dùng để minh họa nguyên lý.
          Kết quả chính xác hơn dùng cv2.createCLAHE().

    Args:
        gray_img (numpy.ndarray): Ảnh grayscale (2D, uint8).
        clip_limit (float): Ngưỡng cắt histogram (càng cao → tương phản càng mạnh).
        tile_size (tuple): Số tile theo (hàng, cột).

    Returns:
        numpy.ndarray: Ảnh sau khi áp dụng CLAHE.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Đầu vào phải là ảnh grayscale 2D.")

    h, w = gray_img.shape
    tile_rows, tile_cols = tile_size

    # Tính kích thước mỗi tile
    tile_h = h // tile_rows
    tile_w = w // tile_cols

    # Tạo ảnh đầu ra
    output = np.zeros_like(gray_img)

    for r in range(tile_rows):
        for c in range(tile_cols):
            # Xác định vùng tile
            y1 = r * tile_h
            y2 = (r + 1) * tile_h if r < tile_rows - 1 else h
            x1 = c * tile_w
            x2 = (c + 1) * tile_w if c < tile_cols - 1 else w

            tile = gray_img[y1:y2, x1:x2]
            tile_pixels = tile.shape[0] * tile.shape[1]

            # Bước 1: Tính histogram của tile
            hist = np.zeros(256, dtype=np.float64)
            for k in range(256):
                hist[k] = np.sum(tile == k)

            # Bước 2: Clip histogram
            # Ngưỡng clip = clip_limit * (số pixel / số bin) / 100
            clip_threshold = clip_limit * tile_pixels / 256

            excess = 0
            for k in range(256):
                if hist[k] > clip_threshold:
                    excess += hist[k] - clip_threshold
                    hist[k] = clip_threshold

            # Phân phối lại phần dư cho tất cả bin
            redistribute = excess / 256
            hist += redistribute

            # Bước 3: Tính CDF và cân bằng
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min()
            cdf_normalized = np.round((cdf - cdf_min) / (tile_pixels - cdf_min) * 255)
            cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)

            # Ánh xạ lại mức xám cho tile
            output[y1:y2, x1:x2] = cdf_normalized[tile]

    return output


# ==============================================================================
# CLAHE - OPENCV
# ==============================================================================

def clahe_opencv(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng CLAHE sử dụng cv2.createCLAHE().

    Args:
        gray_img (numpy.ndarray): Ảnh grayscale (2D, uint8).
        clip_limit (float): Ngưỡng giới hạn tương phản.
            - Giá trị nhỏ (1-2): Tăng cường nhẹ, ít nhiễu
            - Giá trị lớn (4-8): Tăng cường mạnh, có thể khuếch đại nhiễu
        tile_grid_size (tuple): Kích thước lưới tile (rows, cols).
            - Nhỏ (4,4): Vùng lớn, hiệu quả gần global HE
            - Lớn (32,32): Vùng nhỏ, cực kỳ cục bộ

    Returns:
        numpy.ndarray: Ảnh sau khi áp dụng CLAHE.
    """
    if gray_img is None or len(gray_img.shape) != 2:
        raise ValueError("Đầu vào phải là ảnh grayscale 2D.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


# ==============================================================================
# CLAHE CHO ẢNH MÀU
# ==============================================================================

def clahe_color(bgr_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng CLAHE cho ảnh màu BGR.

    Phương pháp:
        1. Chuyển BGR → YCrCb
        2. Áp dụng CLAHE chỉ trên kênh Y (luminance)
        3. Chuyển ngược YCrCb → BGR

    Args:
        bgr_img (numpy.ndarray): Ảnh màu BGR (3 kênh, uint8).
        clip_limit (float): Ngưỡng giới hạn tương phản.
        tile_grid_size (tuple): Kích thước lưới tile.

    Returns:
        numpy.ndarray: Ảnh màu BGR sau CLAHE.
    """
    if bgr_img is None or len(bgr_img.shape) != 3:
        raise ValueError("Đầu vào phải là ảnh màu 3 kênh.")

    # Chuyển sang YCrCb
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))

    # Áp dụng CLAHE trên kênh Y
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels[0] = clahe.apply(channels[0])

    # Gộp lại và chuyển về BGR
    ycrcb_clahe = cv2.merge(channels)
    result = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    return result


# ==============================================================================
# THỬ NGHIỆM NHIỀU THAM SỐ CLAHE
# ==============================================================================

def experiment_clahe_params(gray_img, clip_limits=None, tile_sizes=None):
    """
    Thử nghiệm CLAHE với nhiều bộ tham số khác nhau.

    Args:
        gray_img (numpy.ndarray): Ảnh grayscale gốc.
        clip_limits (list): Danh sách clip_limit cần thử. Mặc định [2.0, 3.0, 5.0].
        tile_sizes (list): Danh sách tile_grid_size cần thử. Mặc định [(4,4), (8,8), (16,16)].

    Returns:
        list[dict]: Danh sách kết quả, mỗi phần tử gồm:
            - 'clip_limit': Giá trị clip limit
            - 'tile_size': Kích thước tile
            - 'result': Ảnh kết quả
            - 'label': Nhãn mô tả
    """
    if clip_limits is None:
        clip_limits = [2.0, 3.0, 5.0]
    if tile_sizes is None:
        tile_sizes = [(4, 4), (8, 8), (16, 16)]

    results = []
    for cl in clip_limits:
        for ts in tile_sizes:
            enhanced = clahe_opencv(gray_img, clip_limit=cl, tile_grid_size=ts)
            results.append({
                'clip_limit': cl,
                'tile_size': ts,
                'result': enhanced,
                'label': f'CLAHE (clip={cl}, tile={ts[0]}x{ts[1]})'
            })

    return results


# ==============================================================================
# HÀM CHÍNH - XỬ LÝ TĂNG CƯỜNG CỤC BỘ
# ==============================================================================

def apply_local_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Áp dụng tăng cường cục bộ cho ảnh (grayscale hoặc màu).

    Args:
        img (numpy.ndarray): Ảnh đầu vào (BGR hoặc grayscale).
        clip_limit (float): Ngưỡng clip limit.
        tile_grid_size (tuple): Kích thước tile.

    Returns:
        dict: Kết quả gồm:
            - 'gray_original': Ảnh grayscale gốc
            - 'gray_enhanced': Ảnh grayscale sau CLAHE
            - 'color_enhanced': Ảnh màu sau CLAHE (nếu đầu vào là ảnh màu)
            - 'params': Dict tham số đã sử dụng
    """
    result = {}

    # Chuyển sang grayscale
    gray = to_grayscale(img)
    result['gray_original'] = gray

    # Áp dụng CLAHE trên grayscale
    result['gray_enhanced'] = clahe_opencv(gray, clip_limit, tile_grid_size)

    # Áp dụng CLAHE trên ảnh màu (nếu có)
    if len(img.shape) == 3:
        result['color_enhanced'] = clahe_color(img, clip_limit, tile_grid_size)

    result['params'] = {
        'clip_limit': clip_limit,
        'tile_grid_size': tile_grid_size
    }

    return result
