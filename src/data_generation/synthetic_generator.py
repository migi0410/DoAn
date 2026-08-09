# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE (Trích xuất thông tin hóa đơn tiếng Việt)
Mô tả: Script sinh dữ liệu giả lập sử dụng Playwright và tích hợp bộ tăng cường ảnh.
Tác giả: Dũng (Leader)
"""

import os
import sys
import json
import re
import random
from datetime import datetime, timedelta
from jinja2 import Environment
from playwright.sync_api import sync_playwright
from invoice_templates import TEMPLATES_MAP

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thiết lập môi trường để tránh lỗi protobuf với PaddlePaddle
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ---------------------------------------------------------
# MOCK LLM DATA GENERATOR (Vietnamese Invoices)
# ---------------------------------------------------------
CITIES = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Bình Dương", "Đồng Nai"]
DISTRICTS = {
    "Hà Nội": ["Quận Cầu Giấy", "Quận Đống Đa", "Quận Ba Đình", "Quận Hai Bà Trưng", "Quận Hoàn Kiếm", "Quận Thanh Xuân"],
    "TP. Hồ Chí Minh": ["Quận 1", "Quận 3", "Quận 7", "Quận 10", "Quận Bình Thạnh", "Quận Tân Bình", "Thành phố Thủ Đức"],
    "Đà Nẵng": ["Quận Hải Châu", "Quận Thanh Khê", "Quận Sơn Trà", "Quận Ngũ Hành Sơn"]
}
STREETS = ["Lê Lợi", "Nguyễn Huệ", "Trần Hưng Đạo", "Cách Mạng Tháng Tám", "Lý Thường Kiệt", "Hai Bà Trưng", "Bùi Thị Xuân", "Nguyễn Trãi"]

TEMPLATE_PROFILES = {
    "supermarket_winmart": {
        "store_name": ["VinCommerce", "VM+QNH CẨM PHẢ", "VM+QNH HẠ LONG", "VM+HN TRẦN DUY HƯNG"],
        "store_type": "supermarket",
        "items": [
            ("Mì tôm Hảo Hảo chua cay", 3500, "gói"),
            ("Sữa tươi TH True Milk 1L", 38000, "hộp"),
            ("Nước rửa chén Sunlight 750ml", 29000, "chai"),
            ("Coca Cola Lon 320ml", 10000, "lon"),
            ("Bánh Custas Orion 6P", 34000, "hộp"),
            ("Khăn giấy Pulppy 100 tờ", 18000, "gói"),
            ("Dầu ăn Simply 1L", 56000, "chai")
        ]
    },
    "supermarket_lotte": {
        "store_name": ["Co.opmart Cống Quỳnh", "Co.op Food Nguyễn Đình Chiểu", "Saigon Co.op", "Co.opmart Lý Thường Kiệt"],
        "store_type": "supermarket",
        "items": [
            ("Hộp dâu tây Đà Lạt 250g", 65000, "hộp"),
            ("Nước cam ép Twister 1L", 28000, "hộp"),
            ("Bột giặt Ariel 3.2kg", 175000, "túi"),
            ("Kem đánh răng Colgate", 32000, "hộp"),
            ("Bánh quy Cosy Kinh Đô", 22000, "gói"),
            ("Khăn ướt Bobby 80 tờ", 27000, "gói"),
            ("Bia Heineken Lon 330ml", 19500, "lon")
        ]
    },
    "supermarket_bachhoaxanh": {
        "store_name": ["BÁCH HÓA XANH", "Bách Hóa Xanh Nguyễn Trãi", "Bách Hóa Xanh Lê Đức Thọ", "Bách Hóa Xanh Quận 12"],
        "store_type": "supermarket",
        "items": [
            ("Rau muống nước 500g", 8000, "bó"),
            ("Thịt ba rọi heo 500g", 75000, "khay"),
            ("Trứng gà tươi hộp 10 quả", 32000, "hộp"),
            ("Hành lá 100g", 3000, "gói"),
            ("Đường cát trắng Biên Hòa 1kg", 26000, "gói"),
            ("Nước mắm Nam Ngư 750ml", 42000, "chai"),
            ("Thịt đùi heo 500g", 68000, "khay")
        ]
    },
    "convenience_circlek": {
        "store_name": ["CIRCLE K", "Circle K Hai Bà Trưng", "Circle K Lê Lợi", "Circle K Phạm Ngũ Lão"],
        "store_type": "convenience",
        "items": [
            ("Mì Ly ăn liền Modern", 9000, "ly"),
            ("Nước uống Aquafina 500ml", 6000, "chai"),
            ("Bánh bao trứng muối", 16000, "cái"),
            ("Xúc xích tiệt trùng Ponnie", 8000, "cái"),
            ("Trà xanh Không Độ 500ml", 10000, "chai"),
            ("Kẹo cao su Cool Air hũ", 22000, "hũ"),
            ("Băng cá nhân Urgo hộp 10 miếng", 15000, "hộp")
        ]
    },
    "convenience_gs25": {
        "store_name": ["GS25 LANDMARK 81", "GS25 NGUYỄN HUỆ", "GS25 TÔN THẤT THUYẾT", "GS25 THỦ ĐỨC"],
        "store_type": "convenience",
        "items": [
            ("Cơm nắm cá hồi sốt Mayo", 15000, "cái"),
            ("Mì trộn Indomie đặc biệt", 12000, "hộp"),
            ("Cafe sữa đá GS25", 22000, "ly"),
            ("Xúc xích Đức ăn liền", 10000, "cái"),
            ("Trà sữa Kirin Latte 345ml", 16000, "chai"),
            ("Bánh giò thịt băm trứng cút", 14000, "cái"),
            ("Sữa chua uống Proby", 8000, "chai")
        ]
    },
    "convenience_7eleven": {
        "store_name": ["7-ELEVEN BÙI VIỆN", "7-Eleven Nguyễn Du", "7-Eleven Saigon Trade Center"],
        "store_type": "convenience",
        "items": [
            ("Sandwich xá xíu phô mai", 25000, "cái"),
            ("Mì xào tương đen", 20000, "hộp"),
            ("Nước khoáng Dasani 500ml", 5000, "chai"),
            ("Snack khoai tây Lays 95g", 19000, "gói"),
            ("Trà đào sả Slurpee", 18000, "ly"),
            ("Kẹo dẻo Haribo Goldbears", 15000, "gói")
        ]
    },
    "cafe_highlands": {
        "store_name": ["HIGHLANDS COFFEE", "Highlands Coffee Lê Lợi", "Highlands Coffee Trần Hưng Đạo", "Highlands Coffee Landmark"],
        "store_type": "cafe",
        "items": [
            ("Phin Sữa Đá Size L", 39000, "ly"),
            ("Trà Sen Vàng Size M", 45000, "ly"),
            ("Freeze Trà Xanh Size M", 55000, "ly"),
            ("Bánh Mì Que Gà Xé", 19000, "cái"),
            ("Cà Phê Đen Đá Size M", 29000, "ly"),
            ("Trà Thạch Đào Size L", 55000, "ly"),
            ("Bánh Tiramisu", 35000, "cái")
        ]
    },
    "cafe_phuclong": {
        "store_name": ["PHÚC LONG TEA & COFFEE", "Phúc Long Landmark", "Phúc Long Nguyễn Văn Cừ", "Phúc Long Lê Lai"],
        "store_type": "cafe",
        "items": [
            ("Trà sữa Phúc Long L", 50000, "ly"),
            ("Hồng Trà Sữa M", 45000, "ly"),
            ("Trà Lài Đác Thơm L", 55000, "ly"),
            ("Trà Đào Phúc Long L", 55000, "ly"),
            ("Bánh Croissant Bơ Pháp", 25000, "cái"),
            ("Cà phê Latte", 45000, "ly")
        ]
    },
    "cafe_starbucks": {
        "store_name": ["STARBUCKS REX HOTEL", "STARBUCKS NEW WORLD", "STARBUCKS HÀN THUYÊN"],
        "store_type": "cafe",
        "items": [
            ("Caramel Macchiato L", 85000, "ly"),
            ("Green Tea Latte M", 75000, "ly"),
            ("Cold Brew Coffee M", 65000, "ly"),
            ("Butter Croissant", 40000, "cái"),
            ("Chocolate Muffin", 45000, "cái"),
            ("Caffe Americano L", 60000, "ly")
        ]
    },
    "restaurant_kfc": {
        "store_name": ["KFC VIỆT NAM", "KFC Nguyễn Thái Học", "KFC Bà Hom", "KFC Nguyễn Ảnh Thủ"],
        "store_type": "restaurant",
        "items": [
            ("Gà Giòn Cay 1 miếng", 39000, "miếng"),
            ("Burger Tôm", 42000, "cái"),
            ("Khoai Tây Chiên L", 28000, "phần"),
            ("Nước ngọt Pepsi L", 19000, "ly"),
            ("Bắp Cải Trộn L", 22000, "hộp"),
            ("Bánh Trứng Egg Tart", 18000, "cái")
        ]
    },
    "restaurant_jollibee": {
        "store_name": ["JOLLIBEE VINCOM", "Jollibee Pasteur", "Jollibee Coopmart"],
        "store_type": "restaurant",
        "items": [
            ("Gà Giòn Vui Vẻ 1 miếng", 35000, "miếng"),
            ("Mì Ý Sốt Bò Bằm", 38000, "đĩa"),
            ("Khoai Tây Lắc Phô Mai", 25000, "phần"),
            ("Bánh Khoai Môn", 15000, "cái"),
            ("Nước ngọt 7Up L", 17000, "ly")
        ]
    },
    "einvoice_viettel": {
        "store_name": ["CÔNG TY CỔ PHẦN CÔNG NGHỆ SAO NAM", "CÔNG TY TNHH THƯƠNG MẠI & DỊCH VỤ KHÁNH AN", "CÔNG TY CỔ PHẦN ĐẦU TƯ & PHÁT TRIỂN HƯNG PHÁT"],
        "store_type": "einvoice",
        "items": [
            ("Máy in HP LaserJet Pro", 3850000, "cái"),
            ("Giấy Double A A4 70gsm", 65000, "ram"),
            ("Mực in Canon Cartridge", 1250000, "hộp"),
            ("Bút bi Thiên Long FO-03", 4000, "cây"),
            ("Dịch vụ Bảo trì Hệ thống mạng", 1500000, "tháng")
        ]
    },
    "einvoice_vnpt": {
        "store_name": ["CÔNG TY TNHH PHÁT TRIỂN CÔNG NGHỆ HOÀNG PHÁT", "TỔNG CÔNG TY DỊCH VỤ VIỄN THÔNG VNPT", "CÔNG TY CỔ PHẦN THIẾT BỊ VĂN PHÒNG HOÀNG MINH"],
        "store_type": "einvoice",
        "items": [
            ("Bộ phát Wifi TP-Link Archer", 850000, "cái"),
            ("Cáp mạng Cat6 UTP 305m", 1850000, "cuộn"),
            ("Dịch vụ Lắp đặt camera giám sát", 2500000, "gói"),
            ("Chuột không dây Logitech", 290000, "cái"),
            ("Bàn phím cơ Dareu EK87", 450000, "cái")
        ]
    },
    "delivery_shopeefood": {
        "store_name": ["BÚN ĐẬU MẮM TÔM CÔ BÔNG", "CƠM TẤM PHÚC LỘC THỌ", "TRÀ SỮA GONG CHA", "BÁNH MÌ HUỲNH HOA"],
        "store_type": "delivery",
        "items": [
            ("Mẹt Bún Đậu Đầy Đủ", 55000, "mẹt"),
            ("Cơm Tấm Sườn Bì Chả", 45000, "dĩa"),
            ("Trà Sữa Trân Châu Đen L", 49000, "ly"),
            ("Bánh Mì Huỳnh Hoa Đặc Biệt", 58000, "ổ"),
            ("Trà Đào Đá M", 35000, "ly")
        ]
    },
    "minimart_anan": {
        "store_name": ["MINIMART ANAN", "ANAN CONVENIENCE STORE", "CỬA HÀNG TIỆN LỢI AN AN"],
        "store_type": "supermarket",
        "items": [
            ("Mì Ly ăn liền Modern", 9000, "ly"),
            ("Nước uống Aquafina 500ml", 6000, "chai"),
            ("Bánh bao trứng muối", 16000, "cái"),
            ("Xúc xích tiệt trùng Ponnie", 8000, "cái"),
            ("Trà xanh Không Độ 500ml", 10000, "chai"),
            ("Kẹo cao su Cool Air hũ", 22000, "hũ"),
            ("Bột giặt Ariel 3.2kg", 175000, "túi"),
            ("Kem đánh răng Colgate", 32000, "hộp")
        ]
    },
    "nhasach_campha": {
        "store_name": ["NHÀ SÁCH GD-TC CẨM PHẢ", "NHÀ SÁCH CẨM PHẢ", "NHÀ SÁCH GIÁO DỤC CẨM PHẢ"],
        "store_type": "bookstore",
        "items": [
            ("Sách Giáo Khoa Toán Lớp 5", 15000, "cuốn"),
            ("Vở ô ly Hồng Hà 48 trang", 7000, "cuốn"),
            ("Bút bi Thiên Long xanh", 4000, "cái"),
            ("Hộp bút sáp màu 12 màu", 18000, "hộp"),
            ("Thước kẻ nhựa 30cm", 5000, "cái"),
            ("Tẩy gôm Pentel Nhật Bản", 8000, "cái"),
            ("Bút chì 2B Hồng Hà", 3500, "cái"),
            ("Sổ tay A5 bìa da", 45000, "cuốn")
        ]
    },
    "cuahang_namoanh": {
        "store_name": ["CỬA HÀNG NĂM OÁNH", "TẠP HÓA NĂM OÁNH", "ĐẠI LÝ BÁN LẺ NĂM OÁNH"],
        "store_type": "supermarket",
        "items": [
            ("Mì tôm Hảo Hảo chua cay", 3500, "gói"),
            ("Sữa tươi TH True Milk 1L", 38000, "hộp"),
            ("Nước rửa chén Sunlight 750ml", 29000, "chai"),
            ("Coca Cola Lon 320ml", 10000, "lon"),
            ("Bánh Custas Orion 6P", 34000, "hộp"),
            ("Dầu ăn Simply 1L", 56000, "chai")
        ]
    },
    "pos_slip": {
        "store_name": ["VIETCOMBANK POS SLIP", "BIDV CARD PAYMENT SYSTEM", "TECHCOMBANK MERCHANT SLIP"],
        "store_type": "pos",
        "items": [
            ("Thanh toán dịch vụ ăn uống", 120000, "lần"),
            ("Mua sắm hàng hóa tiêu dùng", 350000, "lần"),
            ("Thanh toán cước hóa đơn", 450000, "lần")
        ]
    },
    "toll_ticket": {
        "store_name": ["TRẠM THU PHÍ BOT PHÁP VÂN", "TRẠM THU PHÍ BOT XUÂN MAI", "TRẠM THU PHÍ BOT CẦU GIẼ"],
        "store_type": "toll",
        "items": [
            ("Phí dịch vụ đường bộ lượt", 35000, "lượt"),
            ("Phí qua trạm tiêu chuẩn", 40000, "lượt"),
            ("Vé cầu đường quốc lộ 1A", 45000, "lượt")
        ]
    },
    "receipt_c45_bb": {
        "store_name": ["ỦY BAN NHÂN DÂN QUẬN CẦU GIẤY", "TRUNG TÂM Y TẾ HUYỆN GIA LÂM", "TRƯỜNG ĐẠI HỌC BÁCH KHOA HÀ NỘI"],
        "store_type": "receipt",
        "items": [
            ("Lệ phí trước bạ xe máy", 150000, "lượt"),
            ("Viện phí khám bệnh theo yêu cầu", 200000, "lượt"),
            ("Học học phí kỳ I năm học", 1500000, "lượt"),
            ("Phí dịch vụ hành chính công", 50000, "lượt")
        ]
    }
}

# ---------------------------------------------------------
# HELPER FUNCTIONS FOR VIETNAMESE NUMBERS AND CURRENCY
# ---------------------------------------------------------

def num_to_vietnamese_words(num):
    """Converts a number to Vietnamese words for receipts like C45-BB."""
    units = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    if num == 0:
        return "không đồng"
    
    def read_three_digits(n, show_zero=False):
        hundreds = n // 100
        tens = (n % 100) // 10
        ones = n % 10
        res = []
        if hundreds > 0 or show_zero:
            res.append(units[hundreds] + " trăm")
        if tens > 0:
            if tens == 1:
                res.append("mười")
            else:
                res.append(units[tens] + " mươi")
        elif (hundreds > 0 or show_zero) and ones > 0:
            res.append("lẻ")
        
        if ones > 0:
            if ones == 1 and tens > 1:
                res.append("mốt")
            elif ones == 5 and tens > 0:
                res.append("lăm")
            else:
                res.append(units[ones])
        return " ".join(res)
    
    groups = []
    temp = num
    while temp > 0:
        groups.append(temp % 1000)
        temp //= 1000
        
    group_names = ["", "nghìn", "triệu", "tỷ"]
    words = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        show_zero = (i < len(groups) - 1)
        g_words = read_three_digits(g, show_zero)
        if g_words:
            words.append(g_words)
            if group_names[i]:
                words.append(group_names[i])
                
    result_str = " ".join(words) + " đồng"
    result_str = result_str.strip()
    if result_str:
        result_str = result_str[0].upper() + result_str[1:]
    return result_str

def format_currency(val, template_type):
    """Formats currency dynamically based on template branding requirements."""
    val = float(val)
    if template_type in ["supermarket_winmart", "minimart_anan", "nhasach_campha", "cuahang_namoanh", "toll_ticket", "receipt_c45_bb"]:
        # Dot separation, no decimals, no currency sign (for WinMart / MC-OCR compatibility)
        s = f"{val:,.0f}"
        return s.replace(",", ".")
    elif template_type == "supermarket_lotte": # Saigon Co.op
        # Comma separation with .00 decimals
        return f"{val:,.2f}"
    else:
        # Standard: comma separation
        return f"{val:,.0f}"

# ---------------------------------------------------------
# MOCK LLM DATA GENERATOR
# ---------------------------------------------------------

def generate_random_invoice_data(template_type=None):
    """Simulates LLM generating a highly realistic structured Vietnamese invoice JSON."""
    if template_type and template_type in TEMPLATE_PROFILES:
        profile = TEMPLATE_PROFILES[template_type]
    else:
        profile = random.choice(list(TEMPLATE_PROFILES.values()))
        template_type = [k for k, v in TEMPLATE_PROFILES.items() if v == profile][0]
        
    store_name = random.choice(profile["store_name"])
    store_type = profile["store_type"]
    items_source = profile["items"]
    
    city = random.choice(CITIES)
    district = random.choice(DISTRICTS.get(city, ["Quận 1"]))
    street = random.choice(STREETS)
    address = f"Số {random.randint(1, 450)} {street}, {district}, {city}"
    phone = f"0{random.choice([24, 28, 236, 225])}.{random.randint(3000, 9999)}.{random.randint(1000, 9999)}"
    
    # Date in recent 30 days
    days_ago = random.randint(0, 30)
    hour = random.randint(7, 22)
    minute = random.randint(0, 59)
    invoice_date = datetime.now() - timedelta(days=days_ago)
    invoice_date = invoice_date.replace(hour=hour, minute=minute, second=0)
    
    invoice_no = f"HD{random.randint(100000, 999999)}"
    tax_code = f"{random.randint(1000000000, 9999999999)}"
    cashier = random.choice(["Nguyễn Văn A", "Trần Thị B", "Lê Hoàng C", "Phạm Minh D", "Nguyễn Mỹ H"])
    
    # Items selection
    num_items = random.randint(1, 4) if template_type in ["pos_slip", "toll_ticket", "receipt_c45_bb"] else random.randint(2, 6)
    selected_items = random.sample(items_source, k=min(num_items, len(items_source)))
    
    items_list = []
    subtotal = 0
    for idx, (name, price, unit) in enumerate(selected_items):
        qty = 1 if template_type in ["pos_slip", "toll_ticket", "receipt_c45_bb"] else random.randint(1, 4)
        amount = price * qty
        subtotal += amount
        items_list.append({
            "no": idx + 1,
            "name": name,
            "qty": qty,
            "unit": unit,
            "price": price,
            "amount": amount
        })
        
    # Cafe specific fields
    table_no = f"Bàn {random.randint(1, 20):02d}"
    discount_rate = random.choice([0, 5, 10]) if store_type == "cafe" else 0
    discount_amount = int(subtotal * discount_rate / 100)
    
    vat_rate = random.choice([8, 10])
    vat_amount = int((subtotal - discount_amount) * vat_rate / 100)
    total_amount = subtotal - discount_amount + vat_amount
    
    payment_method = random.choice(["Tiền mặt", "Chuyển khoản (VietQR)", "Ví MoMo", "Thẻ Napas"])
    cash_received = ((total_amount + 9999) // 10000) * 10000
    if payment_method != "Tiền mặt" or template_type in ["pos_slip", "toll_ticket"]:
        cash_received = total_amount
    cash_returned = cash_received - total_amount
    
    # Custom dates formats
    if template_type == "supermarket_lotte": # Saigon Co.op
        date_str = invoice_date.strftime("%d/%m/%Y %H:%M:%S")
    elif template_type == "supermarket_winmart":
        date_str = invoice_date.strftime("%d/%m/%Y %H:%M")
    elif template_type == "receipt_c45_bb":
        date_str = f"Ngày {invoice_date.day:02d} tháng {invoice_date.month:02d} năm {invoice_date.year}"
    else:
        date_str = invoice_date.strftime("%d/%m/%Y %H:%M")
        
    # Vietnamese words for C45-BB
    total_words = num_to_vietnamese_words(total_amount)
    
    return {
        "store_name": store_name,
        "store_type": store_type,
        "address": address,
        "phone": phone,
        "tax_code": tax_code,
        "date": date_str,
        "invoice_no": invoice_no,
        "cashier": cashier,
        "items": items_list,
        "subtotal": subtotal,
        "table_no": table_no,
        "discount_rate": discount_rate,
        "discount_amount": discount_amount,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "cash_received": cash_received,
        "cash_returned": cash_returned,
        "total_words": total_words
    }

# ---------------------------------------------------------
# CORE RENDERING & COORDINATE EXTRACTION ENGINE
# ---------------------------------------------------------

def render_html_with_data(data, template_type="supermarket_winmart"):
    """Renders invoice HTML using Jinja2 template engine with custom currency filters."""
    template_str = TEMPLATES_MAP.get(template_type, TEMPLATES_MAP["supermarket_winmart"])
    
    # Preprocess the template string to replace old python string format syntax with a Jinja2 filter.
    # Matches {{ "{:,.0f}".format(X) }} or {{ "{:,.2f}".format(X) }} or {{ '{:,.0f}'.format(X) }}
    processed = re.sub(
        r'\{\{\s*["\']\{:,\.[02]f\}["\']\.format\((.*?)\)\s*\}\}',
        r'{{\1 | format_price}}',
        template_str
    )
    
    thermal_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Inconsolata:wght@400;700&display=swap');
        body {
            font-family: 'Inconsolata', monospace !important;
            letter-spacing: 0px !important;
            color: #000 !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
            font-size: 15px !important;
        }
        table, th, td, span, .store-name {
            font-size: 15px !important;
        }
        .store-name {
            font-size: 18px !important;
            text-transform: uppercase;
        }
        .divider {
            border-top: 2px dashed #000 !important;
            margin: 8px 0 !important;
        }
        * {
            -webkit-font-smoothing: none !important;
        }
    </style>
    </head>
    """
    processed = processed.replace("</head>", thermal_css)
    
    # Create Jinja2 environment and register our custom format_price filter
    env = Environment()
    env.filters['format_price'] = lambda val: format_currency(val, template_type)
    
    # Compile and render
    template = env.from_string(processed)
    return template.render(**data)

def generate_invoice_assets(data, template_type="supermarket_winmart", output_dir="output", prefix="invoice",
                            perspective_level=0.0, shadow_level=0.0, flash_level=0.0, fold_level=0.0,
                            fade_level=0.0, streak_level=0.0, bg_type="white"):
    """
    Renders the invoice HTML with playwright, saves as PDF, PNG,
    and extracts bounding box labels in JSON format. Supports image augmentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pdfs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    html_content = render_html_with_data(data, template_type)
    
    # Save raw HTML first
    html_path = os.path.join(output_dir, f"{prefix}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    png_path = os.path.join(output_dir, "images", f"{prefix}.png")
    pdf_path = os.path.join(output_dir, "pdfs", f"{prefix}.pdf")
    json_path = os.path.join(output_dir, "labels", f"{prefix}.json")
    
    is_a4 = template_type.startswith("einvoice_")
    viewport_width = 794 if is_a4 else 380
    viewport_height = 1123 if is_a4 else 800
    
    # Playwright rendering
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1.0
        )
        page = context.new_page()
        
        # Load HTML
        page.set_content(html_content)
        page.wait_for_timeout(500) # Give it half a second to load webfonts
        
        # 1. Take a screenshot of the page
        page.screenshot(path=png_path, full_page=True)
        
        # 2. Print page to PDF
        if is_a4:
            page.pdf(path=pdf_path, format="A4", margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"})
        else:
            page.pdf(path=pdf_path, width="80mm", height="200mm", margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"})
            
        # 3. RUN DOM EVALUATION TO EXTRACT EXACT BOUNDING BOXES FOR EACH FIELD
        box_data = page.evaluate("""() => {
            const elements = document.querySelectorAll('.kie-field');
            const result = [];
            elements.forEach(el => {
                const rect = el.getBoundingClientRect();
                result.push({
                    label: el.getAttribute('data-label'),
                    text: el.innerText.trim(),
                    box: [
                        Math.round(rect.left),
                        Math.round(rect.top),
                        Math.round(rect.right),
                        Math.round(rect.bottom)
                    ]
                });
            });
            return result;
        }""")
        
        # 4. Áp dụng Tăng cường ảnh (Augmentation) nếu có yêu cầu
        if perspective_level > 0 or shadow_level > 0 or flash_level > 0 or fold_level > 0 or fade_level > 0 or streak_level > 0 or bg_type != "white":
            try:
                import cv2
                from receipt_augmenter import ReceiptAugmenter
                img_augmented, box_data_augmented = ReceiptAugmenter.apply_pipeline(
                    png_path, 
                    box_data, 
                    perspective_level=perspective_level,
                    shadow_level=shadow_level,
                    flash_level=flash_level,
                    fold_level=fold_level,
                    fade_level=fade_level,
                    streak_level=streak_level,
                    bg_type=bg_type
                )
                # Ghi đè ảnh sạch bằng ảnh đã tăng cường
                cv2.imwrite(png_path, img_augmented)
                box_data = box_data_augmented
            except Exception as e:
                print(f"[WARNING] Quá trình tăng cường ảnh bị lỗi: {str(e)}")
        
        # Add metadata and save as JSON
        output_label = {
            "file_name": f"{prefix}.png",
            "image_width": viewport_width,
            "image_height": page.evaluate("() => document.body.scrollHeight"),
            "annotations": box_data
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_label, f, ensure_ascii=False, indent=2)
            
        browser.close()
        
    return {
        "html_path": html_path,
        "png_path": png_path,
        "pdf_path": pdf_path,
        "json_path": json_path,
        "label_data": output_label
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", help="Path to input invoice data JSON file")
    parser.add_argument("--template", default="thermal_classic", help="Template type from TEMPLATES_MAP")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--prefix", default="invoice", help="File prefix")
    
    # Đối số tăng cường ảnh
    parser.add_argument("--perspective", type=float, default=0.0, help="Mức độ méo hình phối cảnh (chụp nghiêng, ví dụ 0.03)")
    parser.add_argument("--shadow", type=float, default=0.0, help="Cường độ bóng mờ (ví dụ 0.3)")
    parser.add_argument("--flash", type=float, default=0.0, help="Cường độ đèn flash (ví dụ 0.3)")
    parser.add_argument("--fold", type=float, default=0.0, help="Biên độ nếp gấp giấy (ví dụ 2.5)")
    parser.add_argument("--fade", type=float, default=0.0, help="Mức độ phai màu mực (ví dụ 0.3)")
    parser.add_argument("--streak", type=float, default=0.0, help="Cường độ vệt lỗi đầu in nhiệt (ví dụ 0.3)")
    
    args = parser.parse_args()
    
    if args.data_json:
        # Load data from file
        with open(args.data_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Generate assets
        assets = generate_invoice_assets(
            data, args.template, args.output_dir, args.prefix,
            perspective_level=args.perspective,
            shadow_level=args.shadow,
            flash_level=args.flash,
            fold_level=args.fold,
            fade_level=args.fade,
            streak_level=args.streak
        )
        print("SUCCESS")
        print(json.dumps(assets, ensure_ascii=False))
    else:
        print("Testing Synthetic Data Generation...")
        data = generate_random_invoice_data()
        print("Generated invoice data mock:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        
        print("\nRendering to Thermal Receipt with test Augmentations...")
        assets_thermal = generate_invoice_assets(
            data, "supermarket_winmart", "test_output", "thermal_test",
            perspective_level=0.03,
            shadow_level=0.25,
            flash_level=0.2,
            fold_level=3.0,
            fade_level=0.35,
            streak_level=0.3
        )
        print(f"Generated Thermal assets: {assets_thermal['png_path']}, {assets_thermal['pdf_path']}, {assets_thermal['json_path']}")
        
        print("\nRendering to e-Invoice...")
        assets_einvoice = generate_invoice_assets(data, "einvoice_viettel", "test_output", "einvoice_test")
        print(f"Generated e-Invoice assets: {assets_einvoice['png_path']}, {assets_einvoice['pdf_path']}, {assets_einvoice['json_path']}")
        print("Done!")
