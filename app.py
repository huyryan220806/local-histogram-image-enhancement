"""
app.py - Giao diện web Streamlit cho đồ án Tăng cường ảnh cục bộ dựa trên Histogram.

Chạy: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# Import modules từ đồ án
from utils import to_grayscale, bgr_to_rgb, compute_histogram, compute_cdf
from global_enhancement import (
    histogram_equalization_manual,
    histogram_equalization_opencv,
    histogram_equalization_color,
)
from local_enhancement import (
    clahe_opencv,
    clahe_color,
    experiment_clahe_params,
)
from comparison import (
    compute_all_metrics,
    compute_mean,
    compute_std,
    compute_entropy,
)


# ==============================================================================
# CẤU HÌNH TRANG
# ==============================================================================

st.set_page_config(
    page_title="Tăng Cường Ảnh - Histogram Enhancement",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
/* ===== Import Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== Global ===== */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* ===== Header banner ===== */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(48, 43, 99, 0.4);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: shimmer 6s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { transform: translateX(-30%) translateY(-30%); }
    50% { transform: translateX(10%) translateY(10%); }
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a8edea, #fed6e3, #a8edea);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-text 4s ease infinite;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}
@keyframes gradient-text {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}
.hero-banner p {
    color: rgba(255, 255, 255, 0.75);
    font-size: 1rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
    margin: 0;
}

/* ===== Section headings ===== */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #a8edea, #fed6e3) 1;
    display: inline-block;
}

/* ===== Metric cards ===== */
.metric-card {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid rgba(168, 237, 234, 0.15);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 237, 234, 0.4);
    box-shadow: 0 8px 24px rgba(168, 237, 234, 0.15);
}
.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(168, 237, 234, 0.8);
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
}
.metric-card .metric-delta {
    font-size: 0.7rem;
    font-weight: 500;
    margin-top: 0.3rem;
}
.metric-delta.good { color: #2ecc71; }
.metric-delta.bad { color: #e74c3c; }
.metric-delta.neutral { color: #95a5a6; }

/* ===== Image comparison cards ===== */
.img-card {
    background: #1a1a2e;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.3s ease;
}
.img-card:hover {
    border-color: rgba(168, 237, 234, 0.3);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.img-card-header {
    padding: 0.7rem 1rem;
    font-weight: 600;
    font-size: 0.85rem;
    text-align: center;
    letter-spacing: 0.5px;
}
.img-card-header.original {
    background: linear-gradient(90deg, #3498db, #2980b9);
    color: white;
}
.img-card-header.global-he {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    color: white;
}
.img-card-header.clahe {
    background: linear-gradient(90deg, #2ecc71, #27ae60);
    color: white;
}

/* ===== Sidebar styling ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1a2e);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a8edea;
}

/* ===== Info box ===== */
.info-box {
    background: linear-gradient(135deg, rgba(168, 237, 234, 0.1), rgba(254, 214, 227, 0.1));
    border: 1px solid rgba(168, 237, 234, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ===== Download button ===== */
.stDownloadButton > button {
    background: linear-gradient(135deg, #a8edea, #fed6e3) !important;
    color: #1a1a2e !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(168, 237, 234, 0.4) !important;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: rgba(255,255,255,0.35);
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HÀM TIỆN ÍCH
# ==============================================================================

def load_uploaded_image(uploaded_file):
    """Đọc ảnh từ file upload của Streamlit."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset file pointer
    return img


def encode_image_to_bytes(img, fmt='.png'):
    """Chuyển ảnh numpy sang bytes để tải xuống."""
    success, encoded = cv2.imencode(fmt, img)
    if success:
        return encoded.tobytes()
    return None


def render_metric_card(label, value, delta_text="", delta_class="neutral"):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {delta_class}">{delta_text}</div>
    </div>
    """


# ==============================================================================
# HERO BANNER
# ==============================================================================

st.markdown("""
<div class="hero-banner">
    <h1>Tăng Cường Ảnh Cục Bộ Dựa Trên Histogram</h1>
    <p>Local Image Enhancement using CLAHE · So sánh Global HE vs CLAHE · Đồ án Nhập Môn Xử Lý Ảnh Số</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR - CẤU HÌNH THAM SỐ
# ==============================================================================

with st.sidebar:
    st.markdown("## Cấu hình tham số")
    st.markdown("---")

    st.markdown("### Tải ảnh lên")
    uploaded_file = st.file_uploader(
        "Chọn ảnh từ máy tính",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        help="Hỗ trợ định dạng: JPG, PNG, BMP, TIFF"
    )

    st.markdown("---")
    st.markdown("### Tham số CLAHE")

    clip_limit = st.slider(
        "Clip Limit",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Ngưỡng giới hạn tương phản. Giá trị nhỏ (1-2): tăng cường nhẹ. Giá trị lớn (4-10): tăng cường mạnh."
    )

    tile_size = st.select_slider(
        "Tile Grid Size",
        options=[4, 8, 16, 32],
        value=8,
        help="Kích thước lưới chia tile. Nhỏ → vùng xử lý lớn. Lớn → vùng xử lý nhỏ, cục bộ hơn."
    )

    st.markdown("---")
    st.markdown("### Phương pháp Global HE")

    global_method = st.radio(
        "Chọn phương pháp:",
        options=['OpenCV (cv2.equalizeHist)', 'Thủ công (Manual)'],
        index=0,
        help="Chọn thuật toán Global HE: OpenCV (nhanh) hoặc Thủ công (minh họa từng bước)."
    )

    st.markdown("---")
    st.markdown("### Hiển thị")

    show_color = st.checkbox("Hiển thị ảnh màu", value=True, help="Hiển thị kết quả trên ảnh màu (nếu ảnh gốc là ảnh màu)")
    show_histogram = st.checkbox("Hiển thị Histogram", value=True, help="Hiển thị biểu đồ histogram và CDF")
    show_metrics = st.checkbox("Hiển thị Metrics", value=True, help="Hiển thị các chỉ số PSNR, SSIM, Entropy")
    show_experiment = st.checkbox("Thử nghiệm tham số CLAHE", value=False, help="Thử CLAHE với nhiều bộ tham số khác nhau")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color: rgba(255,255,255,0.4); font-size: 0.75rem; padding: 1rem 0;">
        <b>Nguyễn Đình Huy</b><br>
        MSSV: 2474802010140<br>
        Đại học Văn Lang
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# NỘI DUNG CHÍNH
# ==============================================================================

if uploaded_file is not None:
    # --- Đọc ảnh ---
    img_bgr = load_uploaded_image(uploaded_file)

    if img_bgr is None:
        st.error("Không thể đọc ảnh. Vui lòng chọn file ảnh khác.")
        st.stop()

    img_gray = to_grayscale(img_bgr)
    is_color = len(img_bgr.shape) == 3

    # --- Xử lý Global HE ---
    method = 'opencv' if 'OpenCV' in global_method else 'manual'
    if method == 'manual':
        global_gray = histogram_equalization_manual(img_gray)
    else:
        global_gray = histogram_equalization_opencv(img_gray)

    if is_color:
        global_color = histogram_equalization_color(img_bgr)
    else:
        global_color = global_gray

    # --- Xử lý CLAHE ---
    tile_grid = (tile_size, tile_size)
    clahe_gray = clahe_opencv(img_gray, clip_limit=clip_limit, tile_grid_size=tile_grid)

    if is_color:
        clahe_color_img = clahe_color(img_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid)
    else:
        clahe_color_img = clahe_gray

    # --- Thông tin ảnh ---
    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if is_color else 1

    st.markdown("""
    <div class="info-box">
        <b>{filename}</b> &nbsp;·&nbsp; {w} × {h} px &nbsp;·&nbsp; {ch} kênh màu &nbsp;·&nbsp;
        CLAHE: clip={cl}, tile={ts}×{ts}
    </div>
    """.format(
        filename=uploaded_file.name,
        w=w, h=h,
        ch=channels,
        cl=clip_limit,
        ts=tile_size
    ), unsafe_allow_html=True)

    # ====================================================================
    # TAB LAYOUT
    # ====================================================================
    tab_compare, tab_hist, tab_metrics, tab_experiment, tab_download = st.tabs([
        "So sánh ảnh",
        "Histogram",
        "Metrics",
        "Thử nghiệm",
        "Tải xuống",
    ])

    # ------------------------------------------------------------------
    # TAB 1: SO SÁNH ẢNH
    # ------------------------------------------------------------------
    with tab_compare:
        st.markdown('<div class="section-title">So sánh kết quả: Ảnh gốc vs Global HE vs CLAHE</div>', unsafe_allow_html=True)

        if show_color and is_color:
            display_original = bgr_to_rgb(img_bgr)
            display_global = bgr_to_rgb(global_color)
            display_clahe = bgr_to_rgb(clahe_color_img)
            mode_label = "Ảnh màu"
        else:
            display_original = img_gray
            display_global = global_gray
            display_clahe = clahe_gray
            mode_label = "Ảnh xám (Grayscale)"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="img-card"><div class="img-card-header original">Ảnh gốc</div></div>', unsafe_allow_html=True)
            st.image(display_original, use_container_width=True)

        with col2:
            st.markdown('<div class="img-card"><div class="img-card-header global-he">Global HE</div></div>', unsafe_allow_html=True)
            st.image(display_global, use_container_width=True)

        with col3:
            st.markdown('<div class="img-card"><div class="img-card-header clahe">CLAHE (Cục bộ)</div></div>', unsafe_allow_html=True)
            st.image(display_clahe, use_container_width=True)

        st.caption(f"Chế độ hiển thị: **{mode_label}** · CLAHE params: clipLimit={clip_limit}, tileGridSize={tile_size}×{tile_size}")

    # ------------------------------------------------------------------
    # TAB 2: HISTOGRAM
    # ------------------------------------------------------------------
    with tab_hist:
        if show_histogram:
            st.markdown('<div class="section-title">Biểu đồ Histogram & CDF</div>', unsafe_allow_html=True)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.patch.set_facecolor('#0e1117')

            images_list = [img_gray, global_gray, clahe_gray]
            titles = ['Ảnh gốc', 'Global HE', 'CLAHE (Cục bộ)']
            colors = ['#3498db', '#e74c3c', '#2ecc71']

            for i, (img, title, color) in enumerate(zip(images_list, titles, colors)):
                # Hàng 1: Ảnh
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
                axes[0, i].axis('off')

                # Hàng 2: Histogram + CDF
                hist = compute_histogram(img)
                cdf = compute_cdf(hist)

                axes[1, i].bar(range(256), hist, color=color, alpha=0.7, width=1)
                axes[1, i].set_title(f'Histogram - {title}', fontsize=12, fontweight='bold', color='white')
                axes[1, i].set_xlabel('Mức xám', fontsize=10, color='white')
                axes[1, i].set_ylabel('Số pixel', fontsize=10, color=color)
                axes[1, i].set_xlim([0, 255])
                axes[1, i].set_facecolor('#1a1a2e')
                axes[1, i].tick_params(colors='white')
                axes[1, i].spines['bottom'].set_color('#555')
                axes[1, i].spines['left'].set_color('#555')
                axes[1, i].spines['top'].set_visible(False)
                axes[1, i].spines['right'].set_visible(False)

                ax2 = axes[1, i].twinx()
                ax2.plot(range(256), cdf, color='#f39c12', linewidth=2.5, label='CDF')
                ax2.set_ylabel('CDF', fontsize=10, color='#f39c12')
                ax2.tick_params(axis='y', labelcolor='#f39c12')
                ax2.set_ylim([0, 1.05])
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_color('#f39c12')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Bật tùy chọn **Hiển thị Histogram** trong sidebar để xem biểu đồ.")

    # ------------------------------------------------------------------
    # TAB 3: METRICS
    # ------------------------------------------------------------------
    with tab_metrics:
        if show_metrics:
            st.markdown('<div class="section-title">Chỉ số đánh giá chất lượng ảnh</div>', unsafe_allow_html=True)

            # Compute metrics
            metrics_global = compute_all_metrics(img_bgr, global_color if is_color else global_gray)
            metrics_clahe = compute_all_metrics(img_bgr, clahe_color_img if is_color else clahe_gray)

            # --- PSNR & SSIM cards ---
            st.markdown("#### So sánh PSNR & SSIM")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                psnr_g = metrics_global['psnr']
                psnr_label = f"{psnr_g:.2f} dB" if psnr_g != float('inf') else "∞"
                st.markdown(render_metric_card("PSNR · Global HE", psnr_label, "So với ảnh gốc", "bad"), unsafe_allow_html=True)

            with c2:
                psnr_c = metrics_clahe['psnr']
                psnr_label_c = f"{psnr_c:.2f} dB" if psnr_c != float('inf') else "∞"
                delta_class = "good" if psnr_c > psnr_g else "bad"
                st.markdown(render_metric_card("PSNR · CLAHE", psnr_label_c, f"{'↑' if psnr_c > psnr_g else '↓'} vs Global HE", delta_class), unsafe_allow_html=True)

            with c3:
                ssim_g = metrics_global['ssim']
                st.markdown(render_metric_card("SSIM · Global HE", f"{ssim_g:.4f}", "So với ảnh gốc", "bad"), unsafe_allow_html=True)

            with c4:
                ssim_c = metrics_clahe['ssim']
                delta_class = "good" if ssim_c > ssim_g else "bad"
                st.markdown(render_metric_card("SSIM · CLAHE", f"{ssim_c:.4f}", f"{'↑' if ssim_c > ssim_g else '↓'} vs Global HE", delta_class), unsafe_allow_html=True)

            st.markdown("")

            # --- Entropy & Std cards ---
            st.markdown("#### So sánh Entropy & Độ lệch chuẩn")
            d1, d2, d3 = st.columns(3)

            with d1:
                ent_orig = metrics_global['entropy_original']
                std_orig = metrics_global['std_original']
                st.markdown(render_metric_card("Entropy · Ảnh gốc", f"{ent_orig:.4f}", f"Std: {std_orig:.2f}", "neutral"), unsafe_allow_html=True)

            with d2:
                ent_g = metrics_global['entropy_enhanced']
                std_g = metrics_global['std_enhanced']
                st.markdown(render_metric_card("Entropy · Global HE", f"{ent_g:.4f}", f"Std: {std_g:.2f}", "neutral"), unsafe_allow_html=True)

            with d3:
                ent_c = metrics_clahe['entropy_enhanced']
                std_c = metrics_clahe['std_enhanced']
                delta_class = "good" if ent_c > ent_g else "neutral"
                st.markdown(render_metric_card("Entropy · CLAHE", f"{ent_c:.4f}", f"Std: {std_c:.2f}", delta_class), unsafe_allow_html=True)

            # --- Summary table ---
            st.markdown("")
            st.markdown("#### Bảng tổng hợp")

            table_data = {
                "Chỉ số": ["Mean (Độ sáng TB)", "Std (Độ tương phản)", "Entropy", "PSNR (dB)", "SSIM"],
                "Ảnh gốc": [
                    f"{metrics_global['mean_original']:.2f}",
                    f"{metrics_global['std_original']:.2f}",
                    f"{metrics_global['entropy_original']:.4f}",
                    "—",
                    "1.0000"
                ],
                "Global HE": [
                    f"{metrics_global['mean_enhanced']:.2f}",
                    f"{metrics_global['std_enhanced']:.2f}",
                    f"{metrics_global['entropy_enhanced']:.4f}",
                    f"{metrics_global['psnr']:.2f}" if metrics_global['psnr'] != float('inf') else "∞",
                    f"{metrics_global['ssim']:.4f}"
                ],
                "CLAHE": [
                    f"{metrics_clahe['mean_enhanced']:.2f}",
                    f"{metrics_clahe['std_enhanced']:.2f}",
                    f"{metrics_clahe['entropy_enhanced']:.4f}",
                    f"{metrics_clahe['psnr']:.2f}" if metrics_clahe['psnr'] != float('inf') else "∞",
                    f"{metrics_clahe['ssim']:.4f}"
                ],
            }

            st.table(table_data)

            # --- Analysis ---
            winner_psnr = "CLAHE" if psnr_c > psnr_g else "Global HE"
            winner_ssim = "CLAHE" if ssim_c > ssim_g else "Global HE"
            winner_ent = "CLAHE" if ent_c > ent_g else "Global HE"

            st.markdown(f"""
            <div class="info-box">
                <b>Nhận xét tự động:</b><br>
                • <b>PSNR</b>: <b>{winner_psnr}</b> giữ độ trung thực tốt hơn (PSNR cao hơn = ít sai lệch so với ảnh gốc).<br>
                • <b>SSIM</b>: <b>{winner_ssim}</b> bảo toàn cấu trúc tốt hơn (SSIM càng gần 1.0 càng tốt).<br>
                • <b>Entropy</b>: <b>{winner_ent}</b> chứa nhiều thông tin hơn (histogram phân bố đều hơn).<br><br>
                ⇒ Kết luận: <b>CLAHE</b> thường vượt trội hơn Global HE nhờ xử lý cục bộ thích ứng, đặc biệt trên ảnh có độ sáng không đồng đều.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("Bật tùy chọn **Hiển thị Metrics** trong sidebar để xem chỉ số.")

    # ------------------------------------------------------------------
    # TAB 4: THỬ NGHIỆM THAM SỐ
    # ------------------------------------------------------------------
    with tab_experiment:
        if show_experiment:
            st.markdown('<div class="section-title">Thử nghiệm CLAHE với nhiều bộ tham số</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                Trang này cho phép bạn quan sát ảnh hưởng của các tham số <b>clipLimit</b> và <b>tileGridSize</b> lên kết quả CLAHE.<br>
                • <b>clipLimit nhỏ (1-2)</b>: tăng cường nhẹ, ít nhiễu.<br>
                • <b>clipLimit lớn (4-8)</b>: tăng cường mạnh, có thể khuếch đại nhiễu.<br>
                • <b>tileGrid nhỏ (4×4)</b>: xử lý vùng lớn, gần Global HE.<br>
                • <b>tileGrid lớn (16×16)</b>: xử lý cục bộ rất mạnh.
            </div>
            """, unsafe_allow_html=True)

            clip_limits = [1.0, 2.0, 3.0, 5.0]
            tile_sizes_exp = [(4, 4), (8, 8), (16, 16)]

            results = experiment_clahe_params(img_gray, clip_limits=clip_limits, tile_sizes=tile_sizes_exp)

            # Display original
            st.markdown("##### Ảnh gốc (Grayscale)")
            st.image(img_gray, width=300)

            # Grid of results
            st.markdown("##### Kết quả thử nghiệm")

            for cl_idx, cl in enumerate(clip_limits):
                st.markdown(f"**Clip Limit = {cl}**")
                cols = st.columns(len(tile_sizes_exp))
                for ts_idx, ts in enumerate(tile_sizes_exp):
                    res_idx = cl_idx * len(tile_sizes_exp) + ts_idx
                    with cols[ts_idx]:
                        st.image(results[res_idx]['result'], caption=f"tile={ts[0]}×{ts[1]}", use_container_width=True)
                st.markdown("")

        else:
            st.info("Bật tùy chọn **Thử nghiệm tham số CLAHE** trong sidebar để khám phá các tham số.")

    # ------------------------------------------------------------------
    # TAB 5: TẢI XUỐNG
    # ------------------------------------------------------------------
    with tab_download:
        st.markdown('<div class="section-title">Tải xuống kết quả</div>', unsafe_allow_html=True)

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.markdown("##### Ảnh gốc")
            original_bytes = encode_image_to_bytes(img_bgr)
            if original_bytes:
                st.download_button(
                    label="Tải ảnh gốc",
                    data=original_bytes,
                    file_name=f"original_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True,
                )

        with col_d2:
            st.markdown("##### Global HE")
            global_dl = global_color if (is_color and show_color) else global_gray
            global_bytes = encode_image_to_bytes(global_dl)
            if global_bytes:
                st.download_button(
                    label="Tải Global HE",
                    data=global_bytes,
                    file_name=f"global_he_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with col_d3:
            st.markdown("##### CLAHE")
            clahe_dl = clahe_color_img if (is_color and show_color) else clahe_gray
            clahe_bytes = encode_image_to_bytes(clahe_dl)
            if clahe_bytes:
                st.download_button(
                    label="Tải CLAHE",
                    data=clahe_bytes,
                    file_name=f"clahe_{uploaded_file.name.rsplit('.', 1)[0]}_cl{clip_limit}_ts{tile_size}.png",
                    mime="image/png",
                    use_container_width=True,
                )

        st.markdown("")
        st.markdown("##### Tải biểu đồ Histogram")

        # Generate histogram figure for download
        fig_dl, axes_dl = plt.subplots(2, 3, figsize=(18, 10))
        fig_dl.patch.set_facecolor('white')
        images_dl = [img_gray, global_gray, clahe_gray]
        titles_dl = ['Ảnh gốc', 'Global HE', 'CLAHE (Cục bộ)']
        colors_dl = ['#3498db', '#e74c3c', '#2ecc71']

        for i, (img, title, color) in enumerate(zip(images_dl, titles_dl, colors_dl)):
            axes_dl[0, i].imshow(img, cmap='gray')
            axes_dl[0, i].set_title(title, fontsize=14, fontweight='bold')
            axes_dl[0, i].axis('off')

            hist = compute_histogram(img)
            cdf = compute_cdf(hist)
            axes_dl[1, i].bar(range(256), hist, color=color, alpha=0.7, width=1)
            axes_dl[1, i].set_title(f'Histogram - {title}', fontsize=12)
            axes_dl[1, i].set_xlabel('Mức xám')
            axes_dl[1, i].set_ylabel('Số pixel')
            axes_dl[1, i].set_xlim([0, 255])

            ax2_dl = axes_dl[1, i].twinx()
            ax2_dl.plot(range(256), cdf, color='orange', linewidth=2, label='CDF')
            ax2_dl.set_ylabel('CDF', color='orange')
            ax2_dl.set_ylim([0, 1.05])

        plt.tight_layout()
        buf = BytesIO()
        fig_dl.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig_dl)

        st.download_button(
            label="Tải biểu đồ Histogram (PNG)",
            data=buf,
            file_name=f"histogram_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    # --- Trang chào mừng khi chưa upload ảnh ---
    st.markdown("")

    col_welcome_l, col_welcome_c, col_welcome_r = st.columns([1, 2, 1])
    with col_welcome_c:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem; background: linear-gradient(90deg, #a8edea, #fed6e3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IMAGE ENHANCEMENT</div>
            <h2 style="color: #a8edea; font-weight: 700; margin-bottom: 0.5rem;">Chào mừng đến với công cụ Tăng Cường Ảnh!</h2>
            <p style="color: rgba(255,255,255,0.6); font-size: 1.05rem; line-height: 1.8; max-width: 600px; margin: 0 auto;">
                Hãy <b>tải một ảnh lên</b> từ thanh bên trái để bắt đầu.<br>
                Công cụ sẽ tự động áp dụng <b>Global Histogram Equalization</b>
                và <b>CLAHE</b> (Contrast Limited Adaptive Histogram Equalization),
                cho phép bạn so sánh trực quan và định lượng giữa hai phương pháp.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="text-align:left;">
            <b>Tính năng chính:</b><br><br>
            &bull; &nbsp;<b>So sánh ảnh</b> — Xem trực tiếp Ảnh gốc · Global HE · CLAHE cạnh nhau<br>
            &bull; &nbsp;<b>Histogram & CDF</b> — Biểu đồ phân bố mức xám và hàm phân phối tích lũy<br>
            &bull; &nbsp;<b>Metrics</b> — PSNR, SSIM, Entropy, Mean, Std Dev<br>
            &bull; &nbsp;<b>Thử nghiệm</b> — Khám phá ảnh hưởng của tham số clipLimit và tileGridSize<br>
            &bull; &nbsp;<b>Tải xuống</b> — Lưu ảnh kết quả và biểu đồ về máy
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("""
<div class="footer">
    Đồ án Nhập Môn Xử Lý Ảnh Số · Trường Đại học Văn Lang <br>
    Nguyễn Đình Huy · MSSV 2474802010140 | Lê Quyết Tiến · MSSV 2474802010386
</div>
""", unsafe_allow_html=True)
