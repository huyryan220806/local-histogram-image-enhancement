"""
report_generator.py - Tạo báo cáo tổng hợp kết quả tăng cường ảnh.

Bao gồm:
- Bảng tổng hợp metrics dạng ảnh
- Nhận xét ưu điểm / hạn chế
- Báo cáo dạng text file
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# TẠO BẢNG METRICS DẠNG ẢNH
# ==============================================================================

def create_metrics_table_image(all_metrics, image_names, save_path=None):
    """
    Tạo bảng tổng hợp metrics dưới dạng ảnh.

    Args:
        all_metrics (list[dict]): Danh sách dict chứa metrics mỗi ảnh.
            Mỗi dict có keys: 'name', 'global', 'local' (chứa metrics).
        image_names (list[str]): Tên các ảnh.
        save_path (str): Đường dẫn lưu ảnh.
    """
    # Headers
    col_labels = [
        'Ảnh', 'PP',
        'Mean', 'Std', 'Entropy',
        'PSNR', 'SSIM'
    ]

    # Build table data
    cell_data = []
    cell_colors = []

    for i, m in enumerate(all_metrics):
        name = f'#{i+1}'
        g = m['global']
        l = m['local']

        # Dòng Original
        cell_data.append([
            name, 'Gốc',
            f"{g['mean_original']:.1f}", f"{g['std_original']:.1f}",
            f"{g['entropy_original']:.2f}",
            '-', '-'
        ])
        cell_colors.append(['#f0f0f0'] * 7)

        # Dòng Global HE
        cell_data.append([
            '', 'Global HE',
            f"{g['mean_enhanced']:.1f}", f"{g['std_enhanced']:.1f}",
            f"{g['entropy_enhanced']:.2f}",
            f"{g['psnr']:.2f}", f"{g['ssim']:.4f}"
        ])
        cell_colors.append(['#ffe0e0'] * 7)

        # Dòng CLAHE
        cell_data.append([
            '', 'CLAHE',
            f"{l['mean_enhanced']:.1f}", f"{l['std_enhanced']:.1f}",
            f"{l['entropy_enhanced']:.2f}",
            f"{l['psnr']:.2f}", f"{l['ssim']:.4f}"
        ])
        cell_colors.append(['#e0ffe0'] * 7)

    # Tạo figure
    fig_height = max(4, 0.5 * len(cell_data) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#4a90d9'] * len(col_labels),
        loc='center',
        cellLoc='center'
    )

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)
        table[0, j].set_height(0.06)

    # Style cells
    for i in range(len(cell_data)):
        for j in range(len(col_labels)):
            table[i + 1, j].set_height(0.04)
            table[i + 1, j].set_fontsize(9)

    table.auto_set_font_size(False)
    table.scale(1, 1.5)

    plt.title('Bảng Tổng Hợp Metrics - So Sánh Global HE vs CLAHE',
              fontsize=14, fontweight='bold', pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# ==============================================================================
# TẠO BIỂU ĐỒ TỔNG HỢP
# ==============================================================================

def create_summary_chart(all_metrics, save_path=None):
    """
    Tạo biểu đồ tổng hợp trung bình metrics.

    Args:
        all_metrics (list[dict]): Danh sách metrics.
        save_path (str): Đường dẫn lưu.
    """
    # Tính trung bình
    avg_global = {
        'mean': np.mean([m['global']['mean_enhanced'] for m in all_metrics]),
        'std': np.mean([m['global']['std_enhanced'] for m in all_metrics]),
        'entropy': np.mean([m['global']['entropy_enhanced'] for m in all_metrics]),
        'psnr': np.mean([m['global']['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['global']['ssim'] for m in all_metrics]),
    }

    avg_local = {
        'mean': np.mean([m['local']['mean_enhanced'] for m in all_metrics]),
        'std': np.mean([m['local']['std_enhanced'] for m in all_metrics]),
        'entropy': np.mean([m['local']['entropy_enhanced'] for m in all_metrics]),
        'psnr': np.mean([m['local']['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['local']['ssim'] for m in all_metrics]),
    }

    avg_original = {
        'mean': np.mean([m['global']['mean_original'] for m in all_metrics]),
        'std': np.mean([m['global']['std_original'] for m in all_metrics]),
        'entropy': np.mean([m['global']['entropy_original'] for m in all_metrics]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Biểu đồ 1: Mean ---
    ax = axes[0, 0]
    categories = ['Gốc', 'Global HE', 'CLAHE']
    values = [avg_original['mean'], avg_global['mean'], avg_local['mean']]
    bars = ax.bar(categories, values,
                  color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_title('Mean (Độ sáng TB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # --- Biểu đồ 2: Std ---
    ax = axes[0, 1]
    values = [avg_original['std'], avg_global['std'], avg_local['std']]
    bars = ax.bar(categories, values,
                  color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_title('Std Dev (Độ tương phản)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # --- Biểu đồ 3: Entropy ---
    ax = axes[0, 2]
    values = [avg_original['entropy'], avg_global['entropy'], avg_local['entropy']]
    bars = ax.bar(categories, values,
                  color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_title('Entropy (Lượng thông tin)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị (bits)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # --- Biểu đồ 4: PSNR ---
    ax = axes[1, 0]
    categories_2 = ['Global HE', 'CLAHE']
    values = [avg_global['psnr'], avg_local['psnr']]
    bars = ax.bar(categories_2, values,
                  color=['#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_title('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('dB')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # --- Biểu đồ 5: SSIM ---
    ax = axes[1, 1]
    values = [avg_global['ssim'], avg_local['ssim']]
    bars = ax.bar(categories_2, values,
                  color=['#e74c3c', '#2ecc71'], alpha=0.85)
    ax.set_title('SSIM', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị')
    ax.set_ylim([0, 1.1])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

    # --- Biểu đồ 6: Radar/Kết luận ---
    ax = axes[1, 2]
    ax.axis('off')
    conclusion_text = (
        "KẾT LUẬN TỔNG QUÁT\n\n"
        "• CLAHE bảo toàn cấu trúc tốt hơn\n"
        "  (SSIM cao hơn)\n\n"
        "• Global HE thay đổi mạnh histogram\n"
        "  (Mean thay đổi nhiều)\n\n"
        "• CLAHE tăng cường chi tiết cục bộ\n"
        "  hiệu quả hơn Global HE\n\n"
        "• CLAHE ít gây mất thông tin\n"
        "  (PSNR cao hơn)"
    )
    ax.text(0.5, 0.5, conclusion_text,
            transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8',
                      edgecolor='#2980b9', linewidth=2))

    plt.suptitle('TỔNG HỢP KẾT QUẢ - So sánh Global HE vs CLAHE',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# ==============================================================================
# TẠO BÁO CÁO TEXT
# ==============================================================================

def generate_text_report(all_metrics, image_names, save_path):
    """
    Tạo báo cáo text chi tiết.

    Args:
        all_metrics: Danh sách metrics cho tất cả ảnh.
        image_names: Tên các ảnh.
        save_path: Đường dẫn lưu file text.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  BÁO CÁO KẾT QUẢ - TĂNG CƯỜNG ẢNH CỤC BỘ DỰA TRÊN HISTOGRAM\n")
        f.write("  Chủ đề 6: Tăng cường ảnh cục bộ dựa trên histogram\n")
        f.write("=" * 80 + "\n\n")

        # ---- PHẦN 1: CƠ SỞ LÝ THUYẾT ----
        f.write("1. CƠ SỞ LÝ THUYẾT\n")
        f.write("-" * 40 + "\n\n")

        f.write("1.1. Histogram Equalization toàn cục (Global HE)\n")
        f.write("  - Tính histogram h(k) cho toàn bộ ảnh\n")
        f.write("  - Tính CDF: C(k) = sum(h(i)) for i=0..k\n")
        f.write("  - Ánh xạ: s(k) = round((C(k) - C_min) / (M*N - C_min) * (L-1))\n")
        f.write("  - Kết quả: histogram đầu ra phân bố đều hơn\n\n")

        f.write("1.2. CLAHE (Contrast Limited Adaptive Histogram Equalization)\n")
        f.write("  - Chia ảnh thành các tile (block) nhỏ\n")
        f.write("  - Tính histogram riêng cho từng tile\n")
        f.write("  - Clip histogram: giới hạn giá trị tối đa mỗi bin\n")
        f.write("  - Phân phối lại phần bị clip cho tất cả bin\n")
        f.write("  - Nội suy bilinear tại biên tile → tránh block artifact\n")
        f.write("  - Tham số chính: clipLimit, tileGridSize\n\n")

        # ---- PHẦN 2: KẾT QUẢ METRICS ----
        f.write("2. KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG\n")
        f.write("-" * 40 + "\n\n")

        f.write(f"{'Ảnh':<8} {'PP':<12} {'Mean':>8} {'Std':>8} "
                f"{'Entropy':>10} {'PSNR':>10} {'SSIM':>10}\n")
        f.write("-" * 68 + "\n")

        for i, m in enumerate(all_metrics):
            name = f"#{i+1}"
            g = m['global']
            l = m['local']

            f.write(f"{name:<8} {'Gốc':<12} {g['mean_original']:>8.1f} "
                    f"{g['std_original']:>8.1f} {g['entropy_original']:>10.2f} "
                    f"{'---':>10} {'---':>10}\n")
            f.write(f"{'':8} {'Global HE':<12} {g['mean_enhanced']:>8.1f} "
                    f"{g['std_enhanced']:>8.1f} {g['entropy_enhanced']:>10.2f} "
                    f"{g['psnr']:>10.2f} {g['ssim']:>10.4f}\n")
            f.write(f"{'':8} {'CLAHE':<12} {l['mean_enhanced']:>8.1f} "
                    f"{l['std_enhanced']:>8.1f} {l['entropy_enhanced']:>10.2f} "
                    f"{l['psnr']:>10.2f} {l['ssim']:>10.4f}\n")
            f.write("\n")

        # ---- PHẦN 3: TRUNG BÌNH ----
        f.write("\n3. METRICS TRUNG BÌNH TRÊN TOÀN BỘ ẢNH\n")
        f.write("-" * 40 + "\n\n")

        avg_g_psnr = np.mean([m['global']['psnr'] for m in all_metrics])
        avg_l_psnr = np.mean([m['local']['psnr'] for m in all_metrics])
        avg_g_ssim = np.mean([m['global']['ssim'] for m in all_metrics])
        avg_l_ssim = np.mean([m['local']['ssim'] for m in all_metrics])
        avg_g_entropy = np.mean([m['global']['entropy_enhanced'] for m in all_metrics])
        avg_l_entropy = np.mean([m['local']['entropy_enhanced'] for m in all_metrics])
        avg_g_std = np.mean([m['global']['std_enhanced'] for m in all_metrics])
        avg_l_std = np.mean([m['local']['std_enhanced'] for m in all_metrics])

        f.write(f"  {'Metric':<15} {'Global HE':>12} {'CLAHE':>12} {'Tốt hơn':>12}\n")
        f.write(f"  {'-'*51}\n")

        better_psnr = "CLAHE" if avg_l_psnr > avg_g_psnr else "Global HE"
        better_ssim = "CLAHE" if avg_l_ssim > avg_g_ssim else "Global HE"
        better_entropy = "CLAHE" if avg_l_entropy > avg_g_entropy else "Global HE"

        f.write(f"  {'PSNR (dB)':<15} {avg_g_psnr:>12.2f} {avg_l_psnr:>12.2f} {better_psnr:>12}\n")
        f.write(f"  {'SSIM':<15} {avg_g_ssim:>12.4f} {avg_l_ssim:>12.4f} {better_ssim:>12}\n")
        f.write(f"  {'Entropy':<15} {avg_g_entropy:>12.2f} {avg_l_entropy:>12.2f} {better_entropy:>12}\n")
        f.write(f"  {'Std Dev':<15} {avg_g_std:>12.1f} {avg_l_std:>12.1f} {'---':>12}\n")

        # ---- PHẦN 4: NHẬN XÉT ----
        f.write("\n\n4. NHẬN XÉT ƯU ĐIỂM VÀ HẠN CHẾ\n")
        f.write("-" * 40 + "\n\n")

        f.write("4.1. Global Histogram Equalization\n\n")
        f.write("  ƯU ĐIỂM:\n")
        f.write("  + Đơn giản, dễ cài đặt\n")
        f.write("  + Tốc độ xử lý nhanh (tính toán trên toàn bộ ảnh 1 lần)\n")
        f.write("  + Hiệu quả với ảnh có độ tương phản thấp đồng đều\n")
        f.write("  + Không có tham số cần điều chỉnh\n\n")

        f.write("  HẠN CHẾ:\n")
        f.write("  - Không hiệu quả với ảnh có độ sáng không đồng đều\n")
        f.write("  - Có thể làm mất chi tiết ở vùng sáng hoặc tối\n")
        f.write("  - Histogram sau xử lý không thực sự phân bố đều\n")
        f.write("  - Dễ khuếch đại nhiễu nếu ảnh có vùng đồng nhất lớn\n")
        f.write("  - Không bảo toàn tốt cấu trúc cục bộ (SSIM thấp hơn CLAHE)\n\n")

        f.write("4.2. CLAHE (Contrast Limited Adaptive Histogram Equalization)\n\n")
        f.write("  ƯU ĐIỂM:\n")
        f.write("  + Cải thiện chi tiết cục bộ hiệu quả\n")
        f.write("  + Bảo toàn cấu trúc tổng thể tốt hơn (SSIM cao hơn)\n")
        f.write("  + Tránh khuếch đại nhiễu nhờ clip limit\n")
        f.write("  + Linh hoạt: điều chỉnh clipLimit và tileGridSize\n")
        f.write("  + Xử lý tốt ảnh có độ sáng không đồng đều\n")
        f.write("  + Tránh hiệu ứng khối nhờ nội suy bilinear\n\n")

        f.write("  HẠN CHẾ:\n")
        f.write("  - Phức tạp hơn Global HE\n")
        f.write("  - Tốc độ xử lý chậm hơn (xử lý nhiều tile)\n")
        f.write("  - Cần chọn tham số phù hợp (clipLimit, tileGridSize)\n")
        f.write("  - Với tileGridSize quá nhỏ → hiệu quả gần Global HE\n")
        f.write("  - Với tileGridSize quá lớn → quá cục bộ, có thể gây artifact\n\n")

        f.write("4.3. So sánh trực tiếp\n\n")

        if avg_l_ssim > avg_g_ssim:
            f.write(f"  → CLAHE bảo toàn cấu trúc tốt hơn "
                    f"(SSIM: {avg_l_ssim:.4f} > {avg_g_ssim:.4f})\n")
        else:
            f.write(f"  → Global HE bảo toàn cấu trúc tốt hơn "
                    f"(SSIM: {avg_g_ssim:.4f} > {avg_l_ssim:.4f})\n")

        if avg_l_psnr > avg_g_psnr:
            f.write(f"  → CLAHE có PSNR cao hơn "
                    f"({avg_l_psnr:.2f} > {avg_g_psnr:.2f} dB)\n")
        else:
            f.write(f"  → Global HE có PSNR cao hơn "
                    f"({avg_g_psnr:.2f} > {avg_l_psnr:.2f} dB)\n")

        f.write(f"\n  → CLAHE phù hợp hơn cho ảnh có độ sáng/tương phản không đồng đều\n")
        f.write(f"  → Global HE phù hợp cho ảnh có độ tương phản thấp đồng đều\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  HẾT BÁO CÁO\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Đã lưu báo cáo: {save_path}")
