"""
utils.py - Hàm tiện ích cho đồ án Tăng cường ảnh cục bộ dựa trên histogram.

Xử lý đường dẫn Unicode (tiếng Việt), đọc/ghi ảnh, chuyển đổi màu.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


# ==============================================================================
# ĐỌC / GHI ẢNH HỖ TRỢ UNICODE PATH
# ==============================================================================

def imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """
    Đọc ảnh hỗ trợ đường dẫn có ký tự Unicode (tiếng Việt).
    Sử dụng np.fromfile thay vì cv2.imread trực tiếp.

    Args:
        filepath (str): Đường dẫn tới file ảnh.
        flags (int): Cờ đọc ảnh của OpenCV (mặc định cv2.IMREAD_COLOR).

    Returns:
        numpy.ndarray: Ma trận ảnh, hoặc None nếu đọc thất bại.
    """
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print(f"[Lỗi] Không thể đọc ảnh: {filepath} - {e}")
        return None


def imwrite_unicode(filepath, img, params=None):
    """
    Ghi ảnh hỗ trợ đường dẫn có ký tự Unicode (tiếng Việt).
    Sử dụng cv2.imencode + tofile thay vì cv2.imwrite trực tiếp.

    Args:
        filepath (str): Đường dẫn lưu ảnh.
        img (numpy.ndarray): Ma trận ảnh cần lưu.
        params (list): Tham số mã hóa (tùy chọn).

    Returns:
        bool: True nếu lưu thành công, False nếu thất bại.
    """
    try:
        # Tạo thư mục cha nếu chưa tồn tại
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Lấy phần mở rộng file
        ext = os.path.splitext(filepath)[1]
        if not ext:
            ext = '.jpg'

        # Mã hóa ảnh theo định dạng tương ứng
        if params is not None:
            result, encoded = cv2.imencode(ext, img, params)
        else:
            result, encoded = cv2.imencode(ext, img)

        if result:
            encoded.tofile(filepath)
            return True
        else:
            print(f"[Lỗi] Không thể mã hóa ảnh: {filepath}")
            return False
    except Exception as e:
        print(f"[Lỗi] Không thể ghi ảnh: {filepath} - {e}")
        return False


# ==============================================================================
# CHUYỂN ĐỔI VÀ XỬ LÝ ẢNH
# ==============================================================================

def to_grayscale(img):
    """
    Chuyển ảnh sang grayscale.

    Args:
        img (numpy.ndarray): Ảnh đầu vào (BGR hoặc grayscale).

    Returns:
        numpy.ndarray: Ảnh grayscale.
    """
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def bgr_to_rgb(img):
    """Chuyển ảnh BGR (OpenCV) sang RGB (matplotlib)."""
    if img is None:
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ==============================================================================
# TÍNH TOÁN HISTOGRAM
# ==============================================================================

def compute_histogram(img, channel=0, bins=256):
    """
    Tính histogram cho một kênh của ảnh.

    Args:
        img (numpy.ndarray): Ảnh đầu vào.
        channel (int): Chỉ số kênh (0 cho grayscale).
        bins (int): Số lượng bin.

    Returns:
        numpy.ndarray: Mảng histogram.
    """
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
    else:
        hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
    return hist.flatten()


def compute_cdf(hist):
    """
    Tính hàm phân phối tích lũy (CDF) từ histogram.

    Args:
        hist (numpy.ndarray): Mảng histogram.

    Returns:
        numpy.ndarray: CDF đã chuẩn hóa về [0, 1].
    """
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Chuẩn hóa về [0, 1]
    return cdf_normalized


# ==============================================================================
# TẢI DANH SÁCH ẢNH TỪ THƯ MỤC
# ==============================================================================

def load_images_from_folder(folder_path, max_images=None):
    """
    Tải tất cả ảnh từ thư mục.

    Args:
        folder_path (str): Đường dẫn thư mục chứa ảnh.
        max_images (int): Số lượng ảnh tối đa cần tải (None = tất cả).

    Returns:
        list[tuple]: Danh sách (tên_file, ảnh_BGR).
    """
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = []

    if not os.path.isdir(folder_path):
        print(f"[Lỗi] Thư mục không tồn tại: {folder_path}")
        return images

    files = sorted(os.listdir(folder_path))
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_ext:
            continue

        fpath = os.path.join(folder_path, fname)
        img = imread_unicode(fpath)
        if img is not None:
            images.append((fname, img))

        if max_images is not None and len(images) >= max_images:
            break

    return images


# ==============================================================================
# TẠO THƯ MỤC OUTPUT
# ==============================================================================

def create_output_dirs(base_output):
    """
    Tạo cấu trúc thư mục output.

    Args:
        base_output (str): Đường dẫn thư mục output gốc.

    Returns:
        dict: Từ điển chứa đường dẫn các thư mục con.
    """
    dirs = {
        'base': base_output,
        'global': os.path.join(base_output, 'global'),
        'local': os.path.join(base_output, 'local'),
        'comparison': os.path.join(base_output, 'comparison'),
        'histograms': os.path.join(base_output, 'histograms'),
        'metrics': os.path.join(base_output, 'metrics'),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


# ==============================================================================
# HIỂN THỊ ẢNH VỚI MATPLOTLIB
# ==============================================================================

def show_images_row(images_titles, figsize=None, save_path=None, dpi=150):
    """
    Hiển thị một hàng ảnh với tiêu đề.

    Args:
        images_titles (list[tuple]): Danh sách (ảnh, tiêu_đề).
        figsize (tuple): Kích thước figure (tùy chọn).
        save_path (str): Đường dẫn lưu ảnh (tùy chọn).
        dpi (int): Độ phân giải khi lưu.

    Returns:
        tuple: (fig, axes)
    """
    n = len(images_titles)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, images_titles):
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(bgr_to_rgb(img))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig, axes
