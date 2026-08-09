"""
AVIR-KIE Demo — Streamlit inference demo với 4 models
Chạy: streamlit run demo.py
API server phải chạy trước: python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os, io, json, requests, time
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_URL = os.environ.get("AVIR_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AVIR-KIE | Demo Trích xuất Hóa đơn",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero h1 { margin:0; font-size:2.2rem; font-weight:800; letter-spacing:-1px; }
.hero p  { margin:10px 0 0; font-size:1rem; opacity:0.85; }

.model-badge {
    display:inline-flex; align-items:center; gap:6px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 4px 14px; font-size:0.78rem; font-weight:600;
    color:white; margin-right:8px; margin-top:8px;
}

.card {
    background: white;
    border-radius: 14px;
    border: 1px solid #e8edf2;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

.field-row {
    display:flex; justify-content:space-between; align-items:center;
    padding: 10px 0; border-bottom: 1px solid #f0f0f0;
}
.field-row:last-child { border-bottom: none; }
.field-label { font-size:0.8rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:.5px; }
.field-value { font-size:0.95rem; font-weight:500; color:#111827; max-width:65%; text-align:right; }

.item-card {
    background: #f8fafc; border-radius: 10px; padding: 12px 16px;
    border-left: 3px solid #6366f1; margin-bottom: 10px;
}
.item-name { font-weight:600; color:#1e293b; margin-bottom:6px; }
.item-meta { display:flex; gap:16px; font-size:0.82rem; color:#64748b; }

.model-tag {
    display:inline-block; font-size:0.7rem; font-weight:700; padding:2px 10px;
    border-radius:12px; margin-bottom:12px; text-transform:uppercase; letter-spacing:.8px;
}
.tag-phobert    { background:#dbeafe; color:#1d4ed8; }
.tag-layoutlm   { background:#d1fae5; color:#065f46; }
.tag-qwen       { background:#fef3c7; color:#92400e; }
.tag-minicpm    { background:#fce7f3; color:#9d174d; }
.tag-rule       { background:#ede9fe; color:#5b21b6; }

.stat-box {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-radius: 12px; padding: 16px; color:white; text-align:center;
}
.stat-box .num { font-size:1.8rem; font-weight:800; }
.stat-box .lbl { font-size:0.75rem; opacity:0.85; margin-top:2px; }

.compare-header {
    background: linear-gradient(90deg,#667eea,#764ba2);
    color:white; border-radius:10px; padding:12px 20px;
    font-weight:700; font-size:1rem; margin-bottom:16px;
}

.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white; border:none; border-radius:10px;
    font-weight:600; padding:12px 28px;
    transition: all 0.3s; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}
.stButton>button:hover { transform:translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.45); }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
MODEL_INFO = {
    "phobert":    {"label": "PhoBERT + PaddleOCR",  "icon": "🤖", "tag": "tag-phobert",  "desc": "NER trên token tiếng Việt"},
    "layoutlmv3": {"label": "LayoutLMv3 + OCR",     "icon": "📄", "tag": "tag-layoutlm", "desc": "Kết hợp layout + text"},
    "qwen2_vl":   {"label": "Qwen2-VL LoRA",        "icon": "👁", "tag": "tag-qwen",     "desc": "Vision-Language Model"},
    "minicpm_v":  {"label": "MiniCPM-V LoRA",       "icon": "⚡", "tag": "tag-minicpm",  "desc": "Compact VLM, tốc độ cao"},
    "rule_based": {"label": "Rule-Based Baseline",  "icon": "📏", "tag": "tag-rule",     "desc": "Regex + heuristic rules"},
}

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    st.markdown("**API Server:**")
    api_url = st.text_input("URL", value=API_URL, label_visibility="collapsed")

    # Check server
    try:
        r = requests.get(f"{api_url}/", timeout=3)
        st.success("✅ API Server đang chạy")
    except:
        st.error("❌ API Server chưa khởi động")

    st.markdown("---")
    st.markdown("**Chọn Models để chạy:**")
    selected_models = []
    for key, info in MODEL_INFO.items():
        default = key in ["phobert", "layoutlmv3", "qwen2_vl", "minicpm_v"]
        if st.checkbox(f"{info['icon']} {info['label']}", value=default, key=f"chk_{key}"):
            selected_models.append(key)

    st.markdown("---")
    show_boxes = st.toggle("Hiển thị Bounding Box", value=True)
    compare_mode = st.toggle("So sánh song song", value=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧾 AVIR-KIE — Invoice KIE Demo</h1>
  <p>Trích xuất thông tin hóa đơn với 4 mô hình AI: PhoBERT · LayoutLMv3 · Qwen2-VL · MiniCPM-V</p>
  <div style="margin-top:14px">
    <span class="model-badge">🤖 PhoBERT</span>
    <span class="model-badge">📄 LayoutLMv3</span>
    <span class="model-badge">👁 Qwen2-VL</span>
    <span class="model-badge">⚡ MiniCPM-V</span>
    <span class="model-badge">📏 Rule-Based</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── UPLOAD ───────────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1])
with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Tải lên hóa đơn")
    uploaded = st.file_uploader(
        "Chọn ảnh hóa đơn (JPG/PNG)", type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption=f"📸 {uploaded.name}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_preview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Thống kê")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="num">{len(selected_models)}</div><div class="lbl">Models</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="num">8</div><div class="lbl">Fields</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="num">✓</div><div class="lbl">Ready</div></div>', unsafe_allow_html=True)

    if not selected_models:
        st.warning("⚠️ Chưa chọn model nào. Chọn ít nhất 1 model ở sidebar.")
    else:
        st.markdown("**Models đã chọn:**")
        for m in selected_models:
            info = MODEL_INFO[m]
            st.markdown(f"- {info['icon']} **{info['label']}** — _{info['desc']}_")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── RUN ──────────────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Chạy Trích xuất", use_container_width=True, disabled=(not uploaded or not selected_models))

def call_api(img_bytes, filename, model_key, api_url):
    files = {"file": (filename, img_bytes, "image/jpeg")}
    data  = {"baseline": model_key, "preprocess": "false"}
    t0 = time.time()
    try:
        r = requests.post(f"{api_url}/api/predict", files=files, data=data, timeout=120)
        elapsed = time.time() - t0
        if r.status_code == 200:
            return r.json(), elapsed
        return {"error": r.text}, elapsed
    except Exception as e:
        return {"error": str(e)}, time.time() - t0

def render_result(result, model_key, elapsed, img_bytes, show_boxes):
    info = MODEL_INFO[model_key]
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<span class="model-tag {info["tag"]}">{info["icon"]} {info["label"]}</span>', unsafe_allow_html=True)
    st.markdown(f'<small style="color:#9ca3af">⏱ {elapsed:.2f}s</small>', unsafe_allow_html=True)

    if "error" in result or not result.get("success"):
        err = result.get("error", result.get("traceback", "Unknown error"))[:300]
        st.error(f"❌ {err}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    extraction = result.get("extraction", {})

    # Header fields
    HEADER_LABELS = {
        "SELLER":     ("🏪", "Người bán"),
        "ADDRESS":    ("📍", "Địa chỉ"),
        "TIMESTAMP":  ("🕐", "Thời gian"),
        "TOTAL_COST": ("💰", "Tổng tiền"),
    }
    st.markdown('<div style="margin:12px 0">', unsafe_allow_html=True)
    for key, (icon, label) in HEADER_LABELS.items():
        val = extraction.get(key, "")
        if val:
            st.markdown(f"""
            <div class="field-row">
              <span class="field-label">{icon} {label}</span>
              <span class="field-value">{val}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Items
    items = extraction.get("ITEMS", [])
    if items:
        st.markdown(f"**🛒 Sản phẩm ({len(items)} mặt hàng):**")
        for item in items[:8]:
            name   = item.get("ITEM_NAME", item.get("item_name", ""))
            qty    = item.get("ITEM_QTY", item.get("ITEM_QUANTITY", item.get("qty", "")))
            price  = item.get("ITEM_PRICE", item.get("item_price", ""))
            amount = item.get("ITEM_AMOUNT", item.get("ITEM_TOTAL", item.get("total", "")))
            if name or amount:
                st.markdown(f"""
                <div class="item-card">
                  <div class="item-name">{name or "—"}</div>
                  <div class="item-meta">
                    <span>SL: {qty or "—"}</span>
                    <span>Giá: {price or "—"}</span>
                    <span>Thành tiền: {amount or "—"}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

    # Bbox image
    if show_boxes and result.get("image_url"):
        try:
            img_r = requests.get(f"{api_url}{result['image_url']}", timeout=10)
            if img_r.status_code == 200:
                st.image(img_r.content, caption="📦 Bounding Boxes", use_container_width=True)
        except:
            pass

    st.markdown('</div>', unsafe_allow_html=True)

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if run_btn and uploaded and selected_models:
    st.markdown("---")
    st.markdown("## 📊 Kết quả Trích xuất")

    img_bytes = uploaded.getvalue()
    results = {}

    with st.spinner(f"🔄 Đang chạy {len(selected_models)} model(s)..."):
        prog = st.progress(0)
        for i, model_key in enumerate(selected_models):
            prog.progress((i+1)/len(selected_models), text=f"Running {MODEL_INFO[model_key]['label']}...")
            result, elapsed = call_api(img_bytes, uploaded.name, model_key, api_url)
            results[model_key] = (result, elapsed)

    if compare_mode and len(selected_models) > 1:
        # Grid layout
        n_cols = min(len(selected_models), 3)
        cols = st.columns(n_cols)
        for i, model_key in enumerate(selected_models):
            result, elapsed = results[model_key]
            with cols[i % n_cols]:
                render_result(result, model_key, elapsed, img_bytes, show_boxes)
    else:
        for model_key in selected_models:
            result, elapsed = results[model_key]
            render_result(result, model_key, elapsed, img_bytes, show_boxes)

    # Timing comparison
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### ⏱ So sánh tốc độ")
        timing_cols = st.columns(len(results))
        for i, (mk, (res, elapsed)) in enumerate(results.items()):
            info = MODEL_INFO[mk]
            ok = res.get("success", False)
            with timing_cols[i]:
                color = "#10b981" if ok else "#ef4444"
                st.markdown(f"""
                <div style="text-align:center;padding:16px;border-radius:12px;
                            border:2px solid {color};margin-bottom:8px">
                  <div style="font-size:1.6rem;font-weight:800;color:{color}">{elapsed:.2f}s</div>
                  <div style="font-size:0.75rem;color:#6b7280;margin-top:4px">{info['icon']} {info['label']}</div>
                  <div style="font-size:0.7rem;color:{'#10b981' if ok else '#ef4444'}">{'✅ OK' if ok else '❌ Lỗi'}</div>
                </div>""", unsafe_allow_html=True)

elif not uploaded:
    st.info("📎 Tải lên ảnh hóa đơn để bắt đầu.")
