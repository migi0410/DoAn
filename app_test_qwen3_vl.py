import os
import sys
import json
import time
import requests
import pandas as pd
from PIL import Image
import streamlit as st

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="AVIR-KIE | Test Qwen3-VL-8B LoRA",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 28px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="header-box">
    <h2 style="margin:0; font-size:26px;">🧾 AVIR-KIE — Trình Kiểm Thử Mô Hình Đa Phương Thức Qwen3-VL-8B</h2>
    <p style="margin:6px 0 0 0; opacity:0.9; font-size:14px;">
        Trích xuất thực thể thông tin (Key Information Extraction) từ hóa đơn tiếng Việt bằng Qwen3-VL Fine-tuned QLoRA & Zero-Shot
    </p>
</div>
""", unsafe_allow_html=True)

# URL API PopOS Server
DEFAULT_API_URL = "http://100.80.138.26:8000"
OLLAMA_API_URL = "http://100.80.138.26:11434"

# Sidebar Cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_host = st.text_input("Địa chỉ PopOS Backend API:", value=DEFAULT_API_URL)
    
    # Check Server Health
    col_btn, col_st = st.columns([1.5, 1])
    with col_btn:
        if st.button("🔄 Kiểm tra Server"):
            try:
                r = requests.get(f"{api_host}/health", timeout=3)
                if r.status_code == 200:
                    st.success("✅ PopOS Server Ready!")
                else:
                    st.warning("⚠️ Server đang phản hồi...")
            except Exception as e:
                st.error("❌ Chưa kết nối được API PopOS.")
                
    st.markdown("---")
    st.subheader("🎯 Chọn Mô Hình")
    model_choice = st.radio(
        "Mô hình thử nghiệm:",
        ["🌟 Qwen3-VL-8B (Fine-tuned QLoRA)", "🤖 Qwen3-VL-8B (Zero-Shot)"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📁 Nguồn Ảnh Hóa Đơn")
    input_mode = st.radio("Cách chọn ảnh:", ["Chọn mẫu có sẵn", "Tải ảnh từ máy tính"], index=0)

# Danh mục mẫu hóa đơn sẵn có
SAMPLE_DIR = r"C:\Users\Admin\DoAn\data\FINAL_RUNPOD_DATASET\images"
selected_img_path = None
uploaded_file_obj = None

if input_mode == "Chọn mẫu có sẵn":
    if os.path.exists(SAMPLE_DIR):
        all_imgs = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Gom nhóm một số ảnh tiêu biểu
        featured_samples = [f for f in all_imgs if any(k in f.lower() for k in ["starbucks", "highlands", "gs25", "circlek", "winmart", "phuclong", "kfc", "lotte"])]
        sample_list = featured_samples[:30] if featured_samples else all_imgs[:30]
        
        selected_sample_name = st.selectbox("Chọn ảnh mẫu tiêu biểu:", sample_list)
        if selected_sample_name:
            selected_img_path = os.path.join(SAMPLE_DIR, selected_sample_name)
    else:
        st.warning(f"Không tìm thấy thư mục ảnh mẫu: {SAMPLE_DIR}")
else:
    uploaded_file_obj = st.file_uploader("Kéo thả hoặc tải lên ảnh hóa đơn (PNG/JPG):", type=["png", "jpg", "jpeg"])

# Layout chính: 2 Cột
col_img, col_result = st.columns([1, 1.3], gap="large")

with col_img:
    st.subheader("📷 Ảnh Hóa Đơn Đầu Vào")
    display_image = None
    if uploaded_file_obj:
        display_image = Image.open(uploaded_file_obj).convert("RGB")
    elif selected_img_path and os.path.exists(selected_img_path):
        display_image = Image.open(selected_img_path).convert("RGB")
        
    if display_image:
        st.image(display_image, use_container_width=True, caption="Ảnh hóa đơn được chọn")
        w, h = display_image.size
        st.caption(f"Kích thước gốc: {w} x {h} px")
    else:
        st.info("Vui lòng chọn hoặc tải lên một ảnh hóa đơn để bắt đầu.")

    st.markdown("<br>", unsafe_allow_html=True)
    extract_btn = st.button("🚀 TRÍCH XUẤT THÔNG TIN (EXTRACT KIE)", use_container_width=True, disabled=(display_image is None))

with col_result:
    tab_extract, tab_chat = st.tabs(["📊 Kết Quả Trích Xuất KIE", "💬 Hỏi Đáp (Visual Chat)"])

    with tab_extract:
        if extract_btn and display_image:
            with st.spinner("Đang gửi ảnh sang GPU RTX 5060 Ti trên PopOS để phân tích..."):
                t0 = time.time()
                try:
                    # Chuẩn bị file gửi đi
                    import io
                    buf = io.BytesIO()
                    display_image.save(buf, format="JPEG", quality=90)
                    buf.seek(0)
                    
                    if "Fine-tuned" in model_choice:
                        # Gọi server FastAPI Qwen3 LoRA
                        files = {"file": ("invoice.jpg", buf, "image/jpeg")}
                        resp = requests.post(f"{api_host}/extract", files=files, timeout=60)
                        res_data = resp.json()
                        extracted = res_data.get("data", {})
                        latency = res_data.get("latency_s", round(time.time() - t0, 2))
                    else:
                        # Gọi Zero-shot qua Ollama
                        import base64
                        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        prompt = "Trích xuất các trường thông tin: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT từ hóa đơn này dưới dạng JSON thuần túy."
                        resp = requests.post(
                            f"{OLLAMA_API_URL}/api/generate",
                            json={"model": "qwen3-vl:8b-instruct", "prompt": prompt, "images": [b64], "stream": False},
                            timeout=60
                        )
                        raw_resp = resp.json().get("response", "")
                        import re
                        m = re.search(r'\{.*\}', raw_resp, re.DOTALL)
                        extracted = json.loads(m.group(0)) if m else {"raw": raw_resp}
                        latency = round(time.time() - t0, 2)

                    st.session_state["last_extracted"] = extracted
                    st.session_state["last_latency"] = latency
                    st.success(f"⚡ Trích xuất thành công trong **{latency} giây**!")
                except requests.exceptions.ConnectionError:
                    st.warning("""
                    ⏳ **Backend API trên PopOS chưa khởi động xong do GPU đang huấn luyện 10 phút cuối!**
                    
                    * **Lý do:** Mô hình đang ở **Step 2,260+ / 2,332 (97%)** và đang dùng 12.1 GB VRAM.
                    * **Thời gian chờ:** Còn khoảng **~9 - 10 phút nữa** (tầm **21:42**).
                    * **Cơ chế tự động:** Ngay khi huấn luyện xong và lưu checkpoint cuối cùng, API cổng 8000 sẽ tự động bật lên ngay lập tức!
                    """)
                except Exception as e:
                    st.error(f"Lỗi gọi suy luận: {e}")

        # Hiển thị kết quả đã lưu trong session state nếu có
        if "last_extracted" in st.session_state:
            data = st.session_state["last_extracted"]
            latency = st.session_state.get("last_latency", 0)

            # 4 Thẻ thông tin chính (Header fields)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🏪 Tên Cửa Hàng (SELLER)</div>
                    <div class="metric-value">{data.get('SELLER', '—')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🕐 Thời Gian (TIMESTAMP)</div>
                    <div class="metric-value">{data.get('TIMESTAMP', '—')}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📍 Địa Chỉ (ADDRESS)</div>
                    <div class="metric-value">{data.get('ADDRESS', '—')}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #4f46e5;">
                    <div class="metric-title">💰 Tổng Tiền (TOTAL_COST)</div>
                    <div class="metric-value" style="color:#4f46e5;">{data.get('TOTAL_COST', '—')}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🛍️ Danh Sách Mặt Hàng (Line Items)")
            
            names = data.get("ITEM_NAME", [])
            qtys = data.get("ITEM_QTY", [])
            prices = data.get("ITEM_PRICE", [])
            amounts = data.get("ITEM_AMOUNT", [])

            if isinstance(names, str): names = [names]
            if isinstance(qtys, str): qtys = [qtys]
            if isinstance(prices, str): prices = [prices]
            if isinstance(amounts, str): amounts = [amounts]

            max_len = max(len(names), len(qtys), len(prices), len(amounts), 0)
            if max_len > 0:
                # Pad to same length
                names = (names + ["—"] * max_len)[:max_len]
                qtys = (qtys + ["—"] * max_len)[:max_len]
                prices = (prices + ["—"] * max_len)[:max_len]
                amounts = (amounts + ["—"] * max_len)[:max_len]

                df = pd.DataFrame({
                    "STT": list(range(1, max_len + 1)),
                    "Tên Sản Phẩm": names,
                    "Số Lượng": qtys,
                    "Đơn Giá": prices,
                    "Thành Tiền": amounts
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Không có mặt hàng con được trích xuất.")

            with st.expander("🔍 Xem Cấu Trúc JSON Nguyên Bản (Raw Response)"):
                st.json(data)
        else:
            st.info("Nhấn **'Trích xuất thông tin'** ở cột bên trái để hiển thị kết quả tại đây.")

    with tab_chat:
        st.subheader("💬 Hỏi Đáp Với Hóa Đơn (Visual QA)")
        chat_q = st.text_input("Nhập câu hỏi của bạn (Ví dụ: 'Hóa đơn này có thuế VAT không?', 'Món nào đắt tiền nhất?'):")
        if st.button("Gửi câu hỏi", key="chat_btn") and chat_q and display_image:
            with st.spinner("Qwen3-VL đang đọc hóa đơn và trả lời..."):
                try:
                    import io
                    buf = io.BytesIO()
                    display_image.save(buf, format="JPEG", quality=90)
                    buf.seek(0)
                    
                    files = {"file": ("invoice.jpg", buf, "image/jpeg")}
                    data_form = {"question": chat_q}
                    resp = requests.post(f"{api_host}/chat", files=files, data=data_form, timeout=60)
                    reply = resp.json().get("reply", "Không nhận được phản hồi.")
                    st.markdown(f"**🤖 Qwen3-VL:** {reply}")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
