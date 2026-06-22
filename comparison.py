"""
comparison.py - Module so sánh và đánh giá kết quả tăng cường ảnh.

Metrics:
- Mean: Độ sáng trung bình
- Std: Độ tương phản (độ lệch chuẩn)
- Entropy: Lượng thông tin trong ảnh
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (tự cài đặt, không cần skimage)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import compute_histogram, compute_cdf, bgr_to_rgb, to_grayscale


# ==============================================================================
# TÍNH CÁC METRIC ĐÁNH GIÁ
# ==============================================================================

def compute_mean(img):
    """Tính độ sáng trung bình của ảnh grayscale."""
    gray = to_grayscale(img)
    return float(np.mean(gray))


def compute_std(img):
    """Tính độ lệch chuẩn (đại diện cho độ tương phản) của ảnh grayscale."""
    gray = to_grayscale(img)
    return float(np.std(gray))


def compute_entropy(img):
    """
    Tính entropy (Shannon) của ảnh grayscale.

    Entropy = -sum(p(k) * log2(p(k))) với p(k) là xác suất mức xám k.
    Entropy cao → ảnh chứa nhiều thông tin, histogram phân bố đều.
    Entropy thấp → ảnh ít thông tin, histogram tập trung.

    Args:
        img (numpy.ndarray): Ảnh đầu vào.

    Returns:
        float: Giá trị entropy.
    """
    gray = to_grayscale(img)
    hist = compute_histogram(gray)
    total = gray.shape[0] * gray.shape[1]

    # Tính xác suất
    prob = hist / total
    # Loại bỏ giá trị 0 (log(0) không xác định)
    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def compute_psnr(original, enhanced):
    """
    Tính PSNR (Peak Signal-to-Noise Ratio) giữa ảnh gốc và ảnh tăng cường.

    PSNR = 10 * log10(MAX^2 / MSE)

    PSNR cao → ảnh tăng cường giữ chất lượng tốt so với gốc.
    Thường PSNR > 30dB được coi là chất lượng tốt.

    Args:
        original (numpy.ndarray): Ảnh gốc.
        enhanced (numpy.ndarray): Ảnh sau tăng cường.

    Returns:
        float: Giá trị PSNR (dB). inf nếu hai ảnh giống hệt.
    """
    gray_orig = to_grayscale(original)
    gray_enh = to_grayscale(enhanced)

    mse = np.mean((gray_orig.astype(np.float64) - gray_enh.astype(np.float64)) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return float(psnr)


def compute_ssim(original, enhanced, window_size=11, C1=6.5025, C2=58.5225):
    """
    Tính SSIM (Structural Similarity Index) - tự cài đặt.

    SSIM đánh giá mức độ tương đồng cấu trúc giữa hai ảnh.
    SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

    SSIM ∈ [-1, 1], giá trị 1 = hai ảnh giống hệt.

    Args:
        original (numpy.ndarray): Ảnh gốc.
        enhanced (numpy.ndarray): Ảnh sau tăng cường.
        window_size (int): Kích thước cửa sổ Gaussian.
        C1 (float): Hằng số ổn định 1 (mặc định (0.01*255)^2).
        C2 (float): Hằng số ổn định 2 (mặc định (0.03*255)^2).

    Returns:
        float: Giá trị SSIM trung bình.
    """
    gray_orig = to_grayscale(original).astype(np.float64)
    gray_enh = to_grayscale(enhanced).astype(np.float64)

    # Tạo kernel Gaussian
    sigma = 1.5
    gauss = cv2.getGaussianKernel(window_size, sigma)
    window = np.outer(gauss, gauss.transpose())

    # Tính mean, variance, covariance sử dụng filter
    mu1 = cv2.filter2D(gray_orig, -1, window)
    mu2 = cv2.filter2D(gray_enh, -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(gray_orig ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(gray_enh ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(gray_orig * gray_enh, -1, window) - mu1_mu2

    # Công thức SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


# ==============================================================================
# TÍNH TẤT CẢ METRICS CHO MỘT ẢNH
# ==============================================================================

def compute_all_metrics(original, enhanced):
    """
    Tính tất cả metrics đánh giá cho một cặp ảnh gốc-tăng cường.

    Args:
        original (numpy.ndarray): Ảnh gốc.
        enhanced (numpy.ndarray): Ảnh sau tăng cường.

    Returns:
        dict: Từ điển chứa tất cả metrics.
    """
    metrics = {
        'mean_original': compute_mean(original),
        'mean_enhanced': compute_mean(enhanced),
        'std_original': compute_std(original),
        'std_enhanced': compute_std(enhanced),
        'entropy_original': compute_entropy(original),
        'entropy_enhanced': compute_entropy(enhanced),
        'psnr': compute_psnr(original, enhanced),
        'ssim': compute_ssim(original, enhanced),
    }
    return metrics


# ==============================================================================
# VẼ SO SÁNH HISTOGRAM
# ==============================================================================

def plot_histogram_comparison(original, global_he, local_clahe,
                              title_prefix="", save_path=None):
    """
    Vẽ so sánh histogram: Gốc vs Global HE vs CLAHE.

    Layout 2 hàng x 3 cột:
        Hàng 1: Ảnh gốc, ảnh Global HE, ảnh CLAHE
        Hàng 2: Histogram tương ứng

    Args:
        original: Ảnh grayscale gốc
        global_he: Ảnh sau Global HE
        local_clahe: Ảnh sau CLAHE
        title_prefix: Tiền tố tiêu đề
        save_path: Đường dẫn lưu ảnh (tùy chọn)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    images = [original, global_he, local_clahe]
    titles = [
        f'{title_prefix}Ảnh gốc',
        f'{title_prefix}Global HE',
        f'{title_prefix}CLAHE (Cục bộ)'
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Hàng 1: Hiển thị ảnh
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(title, fontsize=13, fontweight='bold')
        axes[0, i].axis('off')

    # Hàng 2: Histogram
    for i, (img, title, color) in enumerate(zip(images, titles, colors)):
        hist = compute_histogram(img)
        axes[1, i].bar(range(256), hist, color=color, alpha=0.7, width=1)
        axes[1, i].set_title(f'Histogram - {title}', fontsize=11)
        axes[1, i].set_xlabel('Mức xám', fontsize=10)
        axes[1, i].set_ylabel('Số pixel', fontsize=10)
        axes[1, i].set_xlim([0, 255])

        # Thêm CDF
        ax2 = axes[1, i].twinx()
        cdf = compute_cdf(hist)
        ax2.plot(range(256), cdf, color='orange', linewidth=2, label='CDF')
        ax2.set_ylabel('CDF', fontsize=10, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# ==============================================================================
# VẼ SO SÁNH ẢNH SIDE-BY-SIDE
# ==============================================================================

def plot_comparison_sidebyside(original_bgr, global_he_bgr, local_clahe_bgr,
                                title_prefix="", save_path=None):
    """
    Vẽ so sánh ảnh màu side-by-side: Gốc vs Global HE vs CLAHE.

    Args:
        original_bgr: Ảnh màu BGR gốc
        global_he_bgr: Ảnh màu BGR sau Global HE
        local_clahe_bgr: Ảnh màu BGR sau CLAHE
        title_prefix: Tiền tố tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    images = [original_bgr, global_he_bgr, local_clahe_bgr]
    titles = [
        f'{title_prefix}Ảnh gốc',
        f'{title_prefix}Global HE',
        f'{title_prefix}CLAHE (Cục bộ)'
    ]

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(bgr_to_rgb(img))
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# ==============================================================================
# VẼ BIỂU ĐỒ SO SÁNH METRICS
# ==============================================================================

def plot_metrics_comparison(metrics_global, metrics_local, image_names=None,
                            save_path=None):
    """
    Vẽ biểu đồ cột so sánh metrics giữa Global HE và CLAHE.

    Args:
        metrics_global (list[dict]): Danh sách metrics cho Global HE.
        metrics_local (list[dict]): Danh sách metrics cho CLAHE.
        image_names (list[str]): Tên các ảnh (tùy chọn).
        save_path (str): Đường dẫn lưu ảnh.
    """
    n = len(metrics_global)
    if image_names is None:
        image_names = [f'Ảnh {i+1}' for i in range(n)]

    # Trích xuất metrics
    metric_keys = ['entropy_enhanced', 'std_enhanced', 'psnr', 'ssim']
    metric_labels = ['Entropy', 'Std Dev', 'PSNR (dB)', 'SSIM']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    x = np.arange(n)
    width = 0.35

    for idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]

        vals_global = [m[key] for m in metrics_global]
        vals_local = [m[key] for m in metrics_local]

        bars1 = ax.bar(x - width/2, vals_global, width,
                       label='Global HE', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, vals_local, width,
                       label='CLAHE', color='#2ecc71', alpha=0.8)

        ax.set_xlabel('Ảnh', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'So sánh {label}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        # Rút gọn tên ảnh
        short_names = [f'#{i+1}' for i in range(n)]
        ax.set_xticklabels(short_names, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('So sánh Metrics: Global HE vs CLAHE',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# ==============================================================================
# VẼ SO SÁNH NHIỀU THAM SỐ CLAHE
# ==============================================================================

def plot_clahe_params_comparison(original, results, save_path=None):
    """
    Vẽ so sánh kết quả CLAHE với nhiều bộ tham số.

    Args:
        original: Ảnh grayscale gốc.
        results (list[dict]): Kết quả từ experiment_clahe_params().
        save_path: Đường dẫn lưu ảnh.
    """
    n = len(results) + 1  # +1 cho ảnh gốc
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # Ảnh gốc
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Ảnh gốc', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Các kết quả CLAHE
    for i, res in enumerate(results):
        axes[i + 1].imshow(res['result'], cmap='gray')
        axes[i + 1].set_title(res['label'], fontsize=10, fontweight='bold')
        axes[i + 1].axis('off')

    # Ẩn các ô trống
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle('So sánh CLAHE với nhiều tham số',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig
