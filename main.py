"""
main.py - Script chính cho Đồ Án Cuối Kỳ:
    Tăng Cường Ảnh Cục Bộ Dựa Trên Histogram

Chủ đề 6: Tăng cường ảnh cục bộ dựa trên histogram
Mục tiêu: Cải thiện chi tiết cục bộ trong ảnh có độ sáng
           hoặc độ tương phản không đồng đều.
Yêu cầu: So sánh phương pháp tăng cường cục bộ (CLAHE)
          với phương pháp toàn cục (Global HE),
          nhận xét ưu điểm và hạn chế.

Pipeline:
    1. Đọc ảnh đầu vào từ thư mục "Tăng Cường Ảnh"
    2. Áp dụng Global Histogram Equalization
    3. Áp dụng CLAHE (Local Enhancement)
    4. Tính metrics đánh giá (Mean, Std, Entropy, PSNR, SSIM)
    5. Tạo ảnh so sánh, histogram, biểu đồ
    6. Xuất báo cáo tổng hợp
"""

import os
import sys
import time
import numpy as np

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_images_from_folder, create_output_dirs,
    to_grayscale, imwrite_unicode
)
from global_enhancement import apply_global_enhancement
from local_enhancement import apply_local_enhancement, experiment_clahe_params
from comparison import (
    compute_all_metrics,
    plot_histogram_comparison,
    plot_comparison_sidebyside,
    plot_metrics_comparison,
    plot_clahe_params_comparison
)
from report_generator import (
    create_metrics_table_image,
    create_summary_chart,
    generate_text_report
)


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tự động tìm thư mục ảnh đầu vào (xử lý Unicode)
INPUT_DIR = None
for d in os.listdir(BASE_DIR):
    p = os.path.join(BASE_DIR, d)
    if os.path.isdir(p) and d != 'output' and d != '__pycache__':
        INPUT_DIR = p
        break

if INPUT_DIR is None:
    print("[Lỗi] Không tìm thấy thư mục ảnh đầu vào!")
    sys.exit(1)

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Tham số CLAHE mặc định
DEFAULT_CLIP_LIMIT = 3.0
DEFAULT_TILE_SIZE = (8, 8)

# Số ảnh tối đa xử lý (None = tất cả)
MAX_IMAGES = None


# ==============================================================================
# HÀM CHÍNH
# ==============================================================================

def main():
    """Hàm chính - chạy toàn bộ pipeline."""

    print("=" * 70)
    print("  ĐỒ ÁN CUỐI KỲ: TĂNG CƯỜNG ẢNH CỤC BỘ DỰA TRÊN HISTOGRAM")
    print("  Chủ đề 6 - Xử Lý Ảnh Số")
    print("=" * 70)
    print()

    start_time = time.time()

    # ----- BƯỚC 1: TẢI ẢNH -----
    print("[Bước 1] Tải ảnh đầu vào...")
    print(f"  Thư mục: {INPUT_DIR}")
    images = load_images_from_folder(INPUT_DIR, max_images=MAX_IMAGES)
    print(f"  Đã tải: {len(images)} ảnh")

    if len(images) == 0:
        print("[Lỗi] Không tìm thấy ảnh nào!")
        return

    # ----- BƯỚC 2: TẠO THƯ MỤC OUTPUT -----
    print("\n[Bước 2] Tạo thư mục output...")
    dirs = create_output_dirs(OUTPUT_DIR)
    print(f"  Output: {OUTPUT_DIR}")

    # ----- BƯỚC 3: XỬ LÝ TỪNG ẢNH -----
    print(f"\n[Bước 3] Xử lý tăng cường ảnh...")
    print(f"  CLAHE params: clipLimit={DEFAULT_CLIP_LIMIT}, "
          f"tileGridSize={DEFAULT_TILE_SIZE}")
    print()

    all_metrics = []
    image_names = []
    metrics_global_list = []
    metrics_local_list = []

    for i, (fname, img) in enumerate(images):
        short_name = f"img_{i+1:02d}"
        image_names.append(short_name)

        print(f"  [{i+1}/{len(images)}] Xử lý: {fname}")

        gray = to_grayscale(img)

        # --- 3a: Global HE ---
        global_result = apply_global_enhancement(img, method='opencv')
        gray_global = global_result['gray_enhanced']

        # Lưu ảnh global
        if 'color_enhanced' in global_result:
            save_global = global_result['color_enhanced']
        else:
            save_global = gray_global

        imwrite_unicode(
            os.path.join(dirs['global'], f'{short_name}_global_he.jpg'),
            save_global
        )

        # --- 3b: CLAHE (Local) ---
        local_result = apply_local_enhancement(
            img,
            clip_limit=DEFAULT_CLIP_LIMIT,
            tile_grid_size=DEFAULT_TILE_SIZE
        )
        gray_local = local_result['gray_enhanced']

        # Lưu ảnh local
        if 'color_enhanced' in local_result:
            save_local = local_result['color_enhanced']
        else:
            save_local = gray_local

        imwrite_unicode(
            os.path.join(dirs['local'], f'{short_name}_clahe.jpg'),
            save_local
        )

        # --- 3c: Tính metrics ---
        metrics_g = compute_all_metrics(img, save_global)
        metrics_l = compute_all_metrics(img, save_local)

        metrics_global_list.append(metrics_g)
        metrics_local_list.append(metrics_l)
        all_metrics.append({
            'name': short_name,
            'global': metrics_g,
            'local': metrics_l
        })

        print(f"    Global HE → PSNR: {metrics_g['psnr']:.2f} dB, "
              f"SSIM: {metrics_g['ssim']:.4f}")
        print(f"    CLAHE     → PSNR: {metrics_l['psnr']:.2f} dB, "
              f"SSIM: {metrics_l['ssim']:.4f}")

        # --- 3d: Tạo ảnh so sánh ---
        # So sánh histogram (grayscale)
        plot_histogram_comparison(
            gray, gray_global, gray_local,
            title_prefix=f"{short_name}: ",
            save_path=os.path.join(dirs['histograms'],
                                   f'{short_name}_histogram.png')
        )

        # So sánh side-by-side (màu)
        if len(img.shape) == 3:
            plot_comparison_sidebyside(
                img, save_global, save_local,
                title_prefix=f"{short_name}: ",
                save_path=os.path.join(dirs['comparison'],
                                       f'{short_name}_comparison.png')
            )
        else:
            plot_comparison_sidebyside(
                gray, gray_global, gray_local,
                title_prefix=f"{short_name}: ",
                save_path=os.path.join(dirs['comparison'],
                                       f'{short_name}_comparison.png')
            )

    # ----- BƯỚC 4: THỬ NGHIỆM THAM SỐ CLAHE -----
    print(f"\n[Bước 4] Thử nghiệm CLAHE với nhiều tham số...")

    # Chọn ảnh đầu tiên để thử nghiệm
    sample_img = to_grayscale(images[0][1])
    clahe_results = experiment_clahe_params(
        sample_img,
        clip_limits=[2.0, 3.0, 5.0],
        tile_sizes=[(4, 4), (8, 8), (16, 16)]
    )

    plot_clahe_params_comparison(
        sample_img, clahe_results,
        save_path=os.path.join(dirs['comparison'],
                               'clahe_params_experiment.png')
    )
    print(f"  Đã thử {len(clahe_results)} bộ tham số")

    # ----- BƯỚC 5: TẠO BIỂU ĐỒ SO SÁNH TỔNG HỢP -----
    print(f"\n[Bước 5] Tạo biểu đồ so sánh tổng hợp...")

    # Biểu đồ metrics
    plot_metrics_comparison(
        metrics_global_list, metrics_local_list, image_names,
        save_path=os.path.join(dirs['metrics'], 'metrics_comparison.png')
    )

    # Bảng metrics dạng ảnh
    create_metrics_table_image(
        all_metrics, image_names,
        save_path=os.path.join(dirs['metrics'], 'metrics_table.png')
    )

    # Biểu đồ tổng hợp
    create_summary_chart(
        all_metrics,
        save_path=os.path.join(dirs['metrics'], 'summary_chart.png')
    )

    # ----- BƯỚC 6: XUẤT BÁO CÁO -----
    print(f"\n[Bước 6] Xuất báo cáo...")

    generate_text_report(
        all_metrics, image_names,
        save_path=os.path.join(dirs['base'], 'bao_cao_ket_qua.txt')
    )

    # ----- TỔNG KẾT -----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  HOÀN TẤT! Thời gian: {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  ├── global/       - {len(images)} ảnh sau Global HE")
    print(f"  ├── local/        - {len(images)} ảnh sau CLAHE")
    print(f"  ├── comparison/   - {len(images)} ảnh so sánh + thử nghiệm tham số")
    print(f"  ├── histograms/   - {len(images)} biểu đồ histogram")
    print(f"  ├── metrics/      - Biểu đồ & bảng metrics")
    print(f"  └── bao_cao_ket_qua.txt - Báo cáo chi tiết")

    # In tóm tắt metrics trung bình
    avg_g_psnr = np.mean([m['global']['psnr'] for m in all_metrics])
    avg_l_psnr = np.mean([m['local']['psnr'] for m in all_metrics])
    avg_g_ssim = np.mean([m['global']['ssim'] for m in all_metrics])
    avg_l_ssim = np.mean([m['local']['ssim'] for m in all_metrics])

    print(f"\n  === TÓM TẮT METRICS (TRUNG BÌNH) ===")
    print(f"  {'Metric':<10} {'Global HE':>12} {'CLAHE':>12}")
    print(f"  {'-'*34}")
    print(f"  {'PSNR':<10} {avg_g_psnr:>12.2f} {avg_l_psnr:>12.2f}")
    print(f"  {'SSIM':<10} {avg_g_ssim:>12.4f} {avg_l_ssim:>12.4f}")
    print()


if __name__ == '__main__':
    main()
