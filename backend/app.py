import os
import json
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from synthetic_generator import generate_random_invoice_data, generate_invoice_assets, render_html_with_data

# Set page config for professional appearance
st.set_page_config(
    page_title="AVIR-KIE | Hóa đơn giả lập & Trích xuất Tọa độ",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0056b3 0%, #002244 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 16px;
        opacity: 0.9;
    }
    
    .section-card {
        background-color: white;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 86, 179, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 86, 179, 0.3);
    }
    
    .tag-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #0056b3;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to draw bounding boxes
def draw_bboxes(image_path, annotations):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Bảng màu cho từng loại nhãn để dễ phân biệt
    color_map = {
        "SELLER": "#1f77b4",     # Xanh biển
        "ADDRESS": "#ff7f0e",    # Cam
        "TIMESTAMP": "#2ca02c",  # Xanh lá
        "TOTAL_COST": "#d62728", # Đỏ
        "ITEM_NAME": "#9467bd",  # Tím
        "ITEM_PRICE": "#8c564b", # Nâu
        "ITEM_QTY": "#e377c2",   # Hồng
        "ITEM_AMOUNT": "#7f7f7f" # Xám
    }
    
    for ann in annotations:
        box = ann["box"] # [xmin, ymin, xmax, ymax]
        label = ann["label"]
        color = color_map.get(label, "#17becf") # Mặc định xanh lơ
        
        # Chỉ vẽ khung viền (bounding box) với màu tương ứng
        draw.rectangle(box, outline=color, width=2)
        
        # Vẽ lớp phủ mờ (transparent overlay) bên trong box cho đẹp (tùy chọn)
        # Chuyển đổi mã hex sang RGB
        h = color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        fill_color = rgb + (30,) # Độ trong suốt (alpha = 30)
        draw.rectangle(box, fill=fill_color)
        
    return img

# Initialize session state for invoice data
if "invoice_data" not in st.session_state:
    st.session_state["invoice_data"] = generate_random_invoice_data()

# ---------------------------------------------------------
# RENDER APP INTERFACE
# ---------------------------------------------------------

# Title Banner
st.markdown("""
<div class="main-header">
    <h1>🧾 AVIR-KIE | Công Cụ Tạo Hóa Đơn Giả Lập & Gán Nhãn Tự Động</h1>
    <p>Giải pháp Synthetic Data Pipeline dành cho huấn luyện các mô hình Layout Analysis & LayoutLM (KIE)</p>
</div>
""", unsafe_allow_html=True)

# Main columns
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🎨 Cấu hình Layout & Template")
    TEMPLATES_OPTIONS = [
        ("Hóa đơn nhiệt truyền thống (Classic)", "thermal_classic"),
        ("Siêu thị WinMart (Đỏ)", "supermarket_winmart"),
        ("Siêu thị LOTTE Mart", "supermarket_lotte"),
        ("Bách Hóa Xanh (Xanh lá)", "supermarket_bachhoaxanh"),
        ("Cửa hàng tiện lợi Circle K", "convenience_circlek"),
        ("Cửa hàng tiện lợi GS25", "convenience_gs25"),
        ("Cửa hàng tiện lợi 7-Eleven", "convenience_7eleven"),
        ("Highlands Coffee (Nâu)", "cafe_highlands"),
        ("Phúc Long Tea & Coffee", "cafe_phuclong"),
        ("Starbucks Coffee (Tối giản)", "cafe_starbucks"),
        ("Cửa hàng gà rán KFC", "restaurant_kfc"),
        ("Cửa hàng thức ăn nhanh Jollibee", "restaurant_jollibee"),
        ("Hóa đơn điện tử Viettel (A4)", "einvoice_viettel"),
        ("Hóa đơn điện tử VNPT (A4)", "einvoice_vnpt"),
        ("Hóa đơn giao hàng ShopeeFood (Cam)", "delivery_shopeefood")
    ]
    
    # Track last template to trigger brand-aligned regeneration
    if "last_template_key" not in st.session_state:
        st.session_state["last_template_key"] = None
        
    template_type = st.selectbox("Chọn mẫu hóa đơn:", TEMPLATES_OPTIONS, format_func=lambda x: x[0])
    
    # Auto-generate brand data when template changes
    if st.session_state["last_template_key"] != template_type[1]:
        st.session_state["invoice_data"] = generate_random_invoice_data(template_type[1])
        st.session_state["last_template_key"] = template_type[1]
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🛠️ Cấu hình dữ liệu Hóa đơn (LLM Output Mock)")
    
    # Generate new random data button
    if st.button("🎲 Sinh dữ liệu Ngẫu nhiên", use_container_width=True):
        st.session_state["invoice_data"] = generate_random_invoice_data(template_type[1])
        
    # Editable invoice properties
    data = st.session_state["invoice_data"]
    
    with st.expander("📷 Mô phỏng điều kiện chụp thực tế (Augmentation)", expanded=True):
        perspective = st.slider("Độ méo hình (Chụp nghiêng)", min_value=0.00, max_value=0.08, value=0.00, step=0.01, help="Mô phỏng góc chụp nghiêng của camera.")
        shadow = st.slider("Cường độ Bóng mờ (Shadow)", min_value=0.00, max_value=0.60, value=0.00, step=0.05, help="Mô phỏng bóng tay/vật thể che khuất ánh sáng.")
        flash = st.slider("Cường độ Đèn Flash", min_value=0.00, max_value=0.60, value=0.00, step=0.05, help="Mô phỏng lóa sáng do đèn flash điện thoại.")
        fold = st.slider("Biên độ Nếp gấp giấy (Folds)", min_value=0.0, max_value=8.0, value=0.0, step=0.5, help="Mô phỏng nếp nhăn và nếp gấp trên giấy hóa đơn.")
        fade = st.slider("Độ phai màu mực in nhiệt", min_value=0.00, max_value=0.60, value=0.00, step=0.05, help="Mô phỏng vết loang phai màu/mất chữ do hóa đơn cũ hoặc bị tác động nhiệt.")
        streak = st.slider("Độ vệt xước in nhiệt (Streaks)", min_value=0.00, max_value=0.60, value=0.00, step=0.05, help="Mô phỏng các vệt xước trắng dọc do lỗi đầu in nhiệt của máy in.")
        
    with st.expander("📝 Thông tin cửa hàng & Hóa đơn", expanded=True):
        store_name = st.text_input("Tên Cửa hàng", value=data["store_name"])
        address = st.text_input("Địa chỉ", value=data["address"])
        phone = st.text_input("Số điện thoại", value=data["phone"])
        tax_code = st.text_input("Mã số thuế", value=data["tax_code"])
        invoice_no = st.text_input("Số hóa đơn", value=data["invoice_no"])
        date = st.text_input("Thời gian lập", value=data["date"])
        cashier = st.text_input("Nhân viên thu ngân", value=data["cashier"])
        payment_method = st.selectbox("Phương thức thanh toán", ["Tiền mặt", "Chuyển khoản (VietQR)", "Ví MoMo", "Thẻ Napas"], index=["Tiền mặt", "Chuyển khoản (VietQR)", "Ví MoMo", "Thẻ Napas"].index(data["payment_method"]))
        
        # Cafe specific input fields
        table_no = data.get("table_no", "Bàn 01")
        discount_rate = data.get("discount_rate", 0)
        is_cafe_or_restaurant = template_type[1].startswith("cafe_") or template_type[1].startswith("restaurant_")
        if data.get("store_type") == "cafe" or is_cafe_or_restaurant:
            table_no = st.text_input("Số bàn (Chỉ áp dụng Nhà hàng/Cafe)", value=table_no)
            discount_rate = st.number_input("Chiết khấu / Khuyến mại (%)", value=int(discount_rate), min_value=0, max_value=50, step=5)
        else:
            discount_rate = 0
        
    with st.expander("🛍️ Danh sách mặt hàng (Items)", expanded=True):
        items_list = data["items"]
        updated_items = []
        for idx, item in enumerate(items_list):
            st.markdown(f"**Sản phẩm #{item['no']}**")
            sub_col1, sub_col2, sub_col3 = st.columns([2, 1, 1.2])
            with sub_col1:
                name = st.text_input(f"Tên hàng", value=item["name"], key=f"name_{idx}")
            with sub_col2:
                qty = st.number_input(f"SL", value=int(item["qty"]), min_value=1, step=1, key=f"qty_{idx}")
            with sub_col3:
                price = st.number_input(f"Đơn giá (đ)", value=int(item["price"]), min_value=0, step=1000, key=f"price_{idx}")
                
            amount = qty * price
            updated_items.append({
                "no": item["no"],
                "name": name,
                "qty": qty,
                "unit": item["unit"],
                "price": price,
                "amount": amount
            })
            
    # Calculate totals
    subtotal = sum(item["amount"] for item in updated_items)
    discount_amount = int(subtotal * discount_rate / 100)
    vat_rate = st.number_input("Thuế suất VAT (%)", value=data["vat_rate"], min_value=0, max_value=20, step=1)
    vat_amount = int((subtotal - discount_amount) * vat_rate / 100)
    total_amount = subtotal - discount_amount + vat_amount
    
    # Save edits to session state
    st.session_state["invoice_data"] = {
        "store_name": store_name,
        "store_type": data["store_type"],
        "address": address,
        "phone": phone,
        "tax_code": tax_code,
        "date": date,
        "invoice_no": invoice_no,
        "cashier": cashier,
        "items": updated_items,
        "subtotal": subtotal,
        "table_no": table_no,
        "discount_rate": discount_rate,
        "discount_amount": discount_amount,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "cash_received": ((total_amount + 9999) // 10000) * 10000 if payment_method == "Tiền mặt" else total_amount,
        "cash_returned": (((total_amount + 9999) // 10000) * 10000 - total_amount) if payment_method == "Tiền mặt" else 0
    }
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    tab_preview, tab_output, tab_json = st.tabs(["👁️ Xem trước HTML", "📸 Ảnh Bounding Box (Nhãn LayoutLM)", "📄 Nhãn JSON Tọa độ"])
    
    # Path settings
    output_dir = "streamlit_output"
    prefix = f"invoice_{st.session_state['invoice_data']['invoice_no']}"
    
    # Render HTML in Preview Tab
    with tab_preview:
        html_rendered = render_html_with_data(st.session_state["invoice_data"], template_type[1])
        st.markdown("**Xem trước bố cục HTML:**")
        
        # Display rendered HTML inside iframe
        is_a4 = template_type[1].startswith("einvoice_")
        height = 900 if is_a4 else 700
        st.components.v1.html(html_rendered, height=height, scrolling=True)
        
    with tab_output:
        st.markdown("**Trích xuất ảnh hóa đơn cùng Bounding Box tự động:**")
        
        # Action button to trigger generation
        if st.button("🚀 Render Hóa đơn & Trích xuất Tọa độ", use_container_width=True):
            with st.spinner("Đang chạy Playwright để vẽ hóa đơn và tính toán tọa độ..."):
                try:
                    # Create temporary JSON file for invoice data
                    os.makedirs(output_dir, exist_ok=True)
                    temp_json_path = os.path.join(output_dir, f"temp_{prefix}.json")
                    with open(temp_json_path, "w", encoding="utf-8") as f:
                        json.dump(st.session_state["invoice_data"], f, ensure_ascii=False, indent=2)
                        
                    # Execute synthetic_generator.py in a clean subprocess to avoid thread/asyncio event loop conflicts in Streamlit
                    import subprocess
                    import sys
                    
                    cmd = [
                        sys.executable,
                        "synthetic_generator.py",
                        "--data_json", temp_json_path,
                        "--template", template_type[1],
                        "--output_dir", output_dir,
                        "--prefix", prefix,
                        "--perspective", str(perspective),
                        "--shadow", str(shadow),
                        "--flash", str(flash),
                        "--fold", str(fold),
                        "--fade", str(fade),
                        "--streak", str(streak)
                    ]
                    
                    # Set UTF-8 encoding in the environment
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
                    
                    if result.returncode == 0 and "SUCCESS" in result.stdout:
                        # Extract the JSON block from output
                        stdout_lines = result.stdout.strip().split("\n")
                        json_str = next(line for line in stdout_lines if line.startswith("{"))
                        assets = json.loads(json_str)
                        
                        st.session_state["assets"] = assets
                        st.success("Tạo Hóa đơn và Trích xuất Tọa độ thành công!")
                    else:
                        st.error(f"Lỗi khi chạy bộ sinh dữ liệu: {result.stderr or result.stdout}")
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {str(e)}")
                    
        # If assets have been generated, display them
        if "assets" in st.session_state:
            assets = st.session_state["assets"]
            
            # Generate annotated image showing bounding boxes
            with st.spinner("Đang gán nhãn Bounding Box lên ảnh..."):
                bbox_image = draw_bboxes(assets["png_path"], assets["label_data"]["annotations"])
                
            st.image(bbox_image, caption="Ảnh gán nhãn Bounding Box tự động (Nhiều màu, không text)", use_container_width=True)
            
            # Khung hiển thị các giá trị trích xuất được
            st.markdown("### 📋 Kết quả trích xuất thông tin (Entities)")
            
            extracted_data = []
            for ann in assets["label_data"]["annotations"]:
                extracted_data.append({
                    "Nhãn (Label)": ann["label"],
                    "Giá trị Text": ann["text"]
                })
            
            if extracted_data:
                st.dataframe(extracted_data, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Download actions
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                # PDF Download
                with open(assets["pdf_path"], "rb") as f:
                    st.download_button(
                        label="📥 Tải xuống file PDF Hóa đơn",
                        data=f.read(),
                        file_name=f"{prefix}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with dl_col2:
                # PNG Download
                with open(assets["png_path"], "rb") as f:
                    st.download_button(
                        label="📥 Tải xuống ảnh PNG Hóa đơn",
                        data=f.read(),
                        file_name=f"{prefix}.png",
                        mime="image/png",
                        use_container_width=True
                    )
        else:
            st.info("Nhấp nút **Render Hóa đơn & Trích xuất Tọa độ** phía trên để tạo file PDF, Ảnh và Tọa độ gán nhãn.")
            
    with tab_json:
        st.markdown("**File nhãn Bounding Box chuẩn JSON:**")
        if "assets" in st.session_state:
            st.json(st.session_state["assets"]["label_data"])
            
            # Download JSON Button
            st.download_button(
                label="📥 Tải xuống file nhãn JSON (LayoutLM Annotation)",
                data=json.dumps(st.session_state["assets"]["label_data"], ensure_ascii=False, indent=2),
                file_name=f"{prefix}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Chưa có nhãn JSON. Nhấp nút **Render Hóa đơn & Trích xuất Tọa độ** ở tab Ảnh Bounding Box.")

    st.markdown('</div>', unsafe_allow_html=True)
    
# Footer with KIE pipeline architecture
st.markdown('<div class="section-card" style="text-align: center;">', unsafe_allow_html=True)
st.markdown("""
### 🧠 Phân tích Quy trình Trích xuất Thông tin & Tạo Hóa đơn (AVIR-KIE)
Sử dụng **Synthetic Data Pipeline** (Dữ liệu giả lập) giúp giải quyết các rào cản về dữ liệu trong huấn luyện LayoutLM:
""")

# Build standard architecture grid for the user to understand
arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
with arch_col1:
    st.markdown("""
    **1. LLM Data Generator**
    - Sử dụng mô hình ngôn ngữ (Gemini/ChatGPT) để sinh dữ liệu thô.
    - Dữ liệu thô chứa thông tin thực tế tiếng Việt, phong phú, bảo mật.
    """)
with arch_col2:
    st.markdown("""
    **2. Bố cục HTML Templates**
    - Dữ liệu thô được gắn vào các file HTML thông qua cú pháp render.
    - Cấu trúc CSS giúp định hình vị trí, font chữ, độ rộng hóa đơn.
    """)
with arch_col3:
    st.markdown("""
    **3. Render & Extract ngầm**
    - Trình duyệt ngầm (Playwright) mở trang HTML và in xuất PDF/PNG.
    - Đo tọa độ biên DOM `getBoundingClientRect()` chuẩn xác 100%.
    """)
with arch_col4:
    st.markdown("""
    **4. LayoutLM Training**
    - File ảnh hóa đơn (PNG) + Nhãn tọa độ (JSON) được gom lại thành bộ dữ liệu.
    - Tự động hóa hoàn toàn khâu gán nhãn thủ công (Auto-Labeling).
    """)
st.markdown('</div>', unsafe_allow_html=True)
