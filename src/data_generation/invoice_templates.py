# -*- coding: utf-8 -*-
"""
Dự án: AVIR-KIE
Mô tả: Định nghĩa 21 template HTML hóa đơn tiếng Việt khác nhau (15 mẫu cũ chuẩn hóa đen trắng + 6 mẫu mới).
      Tất cả các template sử dụng chung hệ thống biến và nhãn .kie-field để trích xuất tọa độ tự động.
      Mọi màu sắc màu mè được đưa về thang xám (grayscale) để tăng tính thực tế cho hóa đơn in nhiệt và in kim.
"""

# Template 1: Classic Thermal Receipt (Default black/white)
THERMAL_CLASSIC = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px;
            font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px;
            background-color: #ffffff; color: #000000;
            font-size: 13px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000; margin: 8px 0; }
        .store-name { font-size: 18px; font-weight: bold; display: block; }
        .info-row { display: flex; justify-content: space-between; font-size: 11px; }
        table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px; }
        th, td { text-align: left; padding: 3px 0; vertical-align: top; }
        .item-name { width: 50%; }
        .item-qty-price { width: 25%; text-align: right; }
        .item-amount { width: 25%; text-align: right; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="store-name KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
        <span class="KIE-address kie-field" data-label="address" style="font-size:11px; display:block;">{{address}}</span>
        <span style="font-size:11px;">ĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></span><br>
        <span style="font-size:11px;">MST: <span class="KIE-tax_code kie-field" data-label="tax_code">{{tax_code}}</span></span>
        <div class="divider"></div>
        <span class="bold">HOÁ ĐƠN BÁN LẺ</span><br>
        <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">Số HĐ: {{invoice_no}}</span>
    </div>
    <div class="divider"></div>
    <div class="info-row">
        <span>Ngày lập:</span>
        <span class="KIE-date kie-field" data-label="date">{{date}}</span>
    </div>
    <div class="info-row">
        <span>Thu ngân:</span>
        <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr>
                <th class="item-name">Tên hàng</th>
                <th class="item-qty-price">SLxĐG</th>
                <th class="item-amount">T.Tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td class="item-name"><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td class="item-qty-price"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>x<span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td class="item-amount"><span class="bold KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div class="info-row" style="font-size:13px;">
        <span>Cộng tiền hàng:</span>
        <span class="bold">{{ "{:,.0f}".format(subtotal) }}đ</span>
    </div>
    <div class="info-row" style="font-size:13px;">
        <span>Thuế VAT ({{vat_rate}}%):</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span>
    </div>
    <div class="info-row" style="font-size:15px; margin-top:5px;">
        <span class="bold">TỔNG CỘNG:</span>
        <span class="bold KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
    </div>
    <div class="divider"></div>
    <div class="info-row">
        <span>Thanh toán:</span>
        <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top:20px; font-size:11px;">
        <p>Cảm ơn Quý khách. Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 2: WinMart Supermarket Style (Grayscale standardized)
SUPERMARKET_WINMART = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .gray-header { background-color: #333333; color: white; padding: 6px; font-weight: bold; font-size: 16px; margin-bottom: 8px; text-transform: uppercase; }
        .divider { border-top: 1px dashed #333333; margin: 8px 0; }
        .bold { font-weight: bold; }
        table { width: 100%; font-size: 11px; margin: 8px 0; }
        th { border-bottom: 1px solid #000; padding: 4px 0; }
        td { padding: 4px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="gray-header text-center">
        <span class="KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
    </div>
    <div class="text-center" style="font-size: 11px;">
        <span class="KIE-address kie-field" data-label="address">{{address}}</span><br>
        ĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span> - MST: <span class="KIE-tax_code kie-field" data-label="tax_code">{{tax_code}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 13px;">HÓA ĐƠN THANH TOÁN</div>
    <div class="text-center"><span class="KIE-invoice_no kie-field" data-label="invoice_no">HĐ: {{invoice_no}}</span></div>
    <div style="font-size: 10px; margin-top: 8px;">
        <div>Ngày lập: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Thu ngân: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    <div class="divider"></div>
    <table cellspacing="0">
        <thead>
            <tr>
                <th align="left" style="width: 50%;">Tên sản phẩm</th>
                <th align="right" style="width: 20%;">SL</th>
                <th align="right" style="width: 30%;">T.Tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right"><span class="bold KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between;">
        <span>Tổng tiền hàng:</span>
        <span class="bold">{{ "{:,.0f}".format(subtotal) }}</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span>Thuế VAT:</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:13px; margin-top:4px;" class="bold">
        <span>TỔNG TIỀN PHẢI T.TOÁN:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="divider"></div>
    <div style="font-size: 11px;">
        Thanh toán: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 20px; font-size:10px;">
        <p>Quý khách vui lòng kiểm tra hóa đơn trước khi ra về!</p>
    </div>
</body>
</html>
"""

# Template 3: Saigon Co.op / Lotte Style (Grayscale standardized)
SUPERMARKET_LOTTE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #333;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .coop-header { color: #000000; font-weight: 800; font-size: 20px; text-transform: uppercase; }
        .divider { border-top: 2px solid #333; margin: 8px 0; }
        .sub-divider { border-top: 1px solid #ddd; margin: 6px 0; }
        .bold { font-weight: bold; }
        table { width: 100%; font-size: 11px; margin: 6px 0; }
        td, th { padding: 3px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="coop-header KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size:11px;" data-label="address">{{address}}</span><br>
        ĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 14px; letter-spacing:1px;">PHIẾU THANH TOÁN</div>
    <div class="text-center" style="font-size: 10px;">Số HĐ: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="sub-divider"></div>
    <table style="font-size: 10px;">
        <tr>
            <td>Ngày: <span class="KIE-date kie-field" data-label="date">{{date}}</span></td>
            <td align="right">Thu ngân: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></td>
        </tr>
    </table>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #333;">
                <th align="left">Tên hàng hóa</th>
                <th align="right">Đơn giá</th>
                <th align="right">SL</th>
                <th align="right">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td colspan="4"><span class="bold KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
            </tr>
            <tr style="color: #666;">
                <td></td>
                <td align="right"><span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.2f}".format(item.price) }}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span></td>
                <td align="right" style="color:#000;"><span class="bold KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.2f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <table style="font-size:12px;">
        <tr>
            <td>Cộng tiền hàng:</td>
            <td align="right">{{ "{:,.2f}".format(subtotal) }}</td>
        </tr>
        <tr>
            <td>Thuế VAT ({{vat_rate}}%):</td>
            <td align="right"><span class="KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.2f}".format(vat_amount) }}</span></td>
        </tr>
        <tr style="font-size:14px; font-weight:bold;">
            <td>Tong so tien thanh toan:</td>
            <td align="right" style="color: #000;"><span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.2f}".format(total_amount) }}</span></td>
        </tr>
    </table>
    <div class="sub-divider"></div>
    <div style="font-size: 11px;">
        Hình thức: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top:20px; font-size:10px; color:#777;">
        <p>Cảm ơn Quý khách đã mua sắm!</p>
    </div>
</body>
</html>
"""

# Template 4: Bach Hoa Xanh Style (Grayscale standardized)
SUPERMARKET_BACHHOAXANH = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000000;
            font-size: 12px; line-height: 1.4; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px solid #000000; margin: 8px 0; }
        .dashed-divider { border-top: 1px dashed #555555; margin: 6px 0; }
        .brand-logo { font-size: 16px; font-weight: 800; color: #000; text-transform: uppercase; }
        .store-info { font-size: 11px; display: block; }
        .title { font-size: 14px; font-weight: bold; color: #000; margin: 8px 0 4px 0; display: block; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="brand-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="store-info">www.bachhoaxanh.com</span>
        <span class="store-info KIE-address kie-field" data-label="address">{{address}}</span>
    </div>
    
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 14px; letter-spacing: 0.5px;">PHIẾU THANH TOÁN</div>
    
    <div style="font-size: 11px; margin-top: 8px; line-height: 1.5;">
        <div>Số CT: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
        <div>Ngày CT: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Nhân viên: &nbsp;&nbsp;&nbsp;&nbsp;<span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    
    <div class="divider"></div>
    
    <!-- Table Header (flexbox to match no product header style) -->
    <div style="display: flex; font-size: 11px; font-weight: bold; border-bottom: 1px solid #000000; padding-bottom: 4px;">
        <div style="width: 30%; text-align: left; padding-left: 20px;">SL</div>
        <div style="width: 35%; text-align: right;">Giá bán</div>
        <div style="width: 35%; text-align: right;">T.Tiền</div>
    </div>
    
    <!-- Items Loop -->
    <div style="margin: 6px 0;">
        {% for item in items %}
        <div style="margin-bottom: 8px;">
            <!-- Row 1: Product Name -->
            <div style="font-size: 11px;" class="bold">
                <span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span>
            </div>
            <!-- Row 2: Quantities & Cost -->
            <div style="display: flex; font-size: 11px; margin-top: 2px;">
                <div style="width: 30%; text-align: left; padding-left: 20px;">
                    <span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>
                </div>
                <div style="width: 35%; text-align: right;">
                    <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span>
                </div>
                <div style="width: 35%; text-align: right;" class="bold">
                    <span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="divider"></div>
    
    <!-- Totals -->
    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
        <span>Tổng tiền:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 2px;" class="bold">
        <span>Thanh toán:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div style="text-align: left; font-size: 10px; font-style: italic; color: #555; margin-left: 15px; margin-bottom: 6px;">
        (Đã làm tròn)
    </div>
    
    <div class="dashed-divider"></div>
    
    <div style="font-size: 11px; line-height: 1.5;">
        <div style="display: flex; justify-content: space-between;">
            <span>Tiền mặt:</span>
            <span>{{ "{:,.0f}".format(total_amount) }}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Tiền thối lại:</span>
            <span>0</span>
        </div>
        <div style="text-align: center; font-size: 10px; font-style: italic; margin-top: 4px;">
            (Giá trên đã bao gồm thuế GTGT)
        </div>
    </div>
    
    <div class="divider"></div>
    
    <!-- CSS Barcode Simulation -->
    <div style="display: flex; justify-content: center; margin: 12px 0;">
        <div style="width: 250px; height: 35px; background: repeating-linear-gradient(90deg, #000, #000 2px, #fff 2px, #fff 5px, #000 5px, #000 7px, #fff 7px, #fff 9px); opacity: 0.85;"></div>
    </div>
    
    <div class="divider"></div>
    
    <div style="font-size: 10px; line-height: 1.4; color: #333; text-align: left; padding: 0 5px;">
        <div class="bold" style="text-align: center; font-size: 11px; margin-bottom: 4px;">Tổng đài góp ý/khiếu nại: 1800 1067</div>
        <span>Lưu ý: Bách Hóa Xanh chỉ xuất hóa đơn trong ngày, Quý khách vui lòng liên hệ thu ngân để được hỗ trợ. Quý khách có thể in bản sao hóa đơn VAT tại trang web https://hddt.bachhoaxanh.com.</span><br><br>
        <span>Quý khách vui lòng xem chi tiết Chính sách đổi-trả hàng được niêm yết tại cửa hàng BHX. Xin cảm ơn quý khách!</span>
    </div>
</body>
</html>
"""

# Template 5: Circle K Convenience Store Style (Grayscale)
CONVENIENCE_CIRCLEK = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 12px; line-height: 1.3;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .logo { font-size: 24px; font-weight: 900; letter-spacing: -1px; margin-bottom: 2px; }
        .logo-box { border: 2px solid #000; display: inline-block; padding: 2px 8px; margin-bottom: 5px; }
        .divider { border-top: 1px dashed #000; margin: 6px 0; }
        .bold { font-weight: bold; }
        table { width: 100%; font-size: 11px; }
        td, th { padding: 2px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <div class="logo-box">
            <span class="logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
        </div><br>
        <span class="KIE-address kie-field" style="font-size: 10px;" data-label="address">{{address}}</span><br>
        SĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold">PHIẾU THANH TOÁN</div>
    <div class="text-center" style="font-size:10px;">Số HĐ: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="divider"></div>
    <div style="font-size: 10px;">
        <div>Thời gian: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Thu ngân: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr>
                <th align="left" style="width: 50%;">Tên hàng</th>
                <th align="right" style="width: 25%;">SLxĐG</th>
                <th align="right" style="width: 25%;">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>x<span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right"><span class="bold KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between;">
        <span>Cộng tiền hàng:</span>
        <span class="bold">{{ "{:,.0f}".format(subtotal) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span>Thuế VAT:</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:13px; margin-top:2px;" class="bold">
        <span>TỔNG THANH TOÁN:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
    </div>
    <div class="divider"></div>
    <div style="font-size: 11px;">
        Thanh toán: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 15px; font-size:10px;">
        <p>Cảm ơn & Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 6: GS25 Convenience Store Style (Grayscale standardized)
CONVENIENCE_GS25 = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #333;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .gs-bold { color: #000000; font-weight: 800; font-size: 20px; }
        .divider { border-top: 1px solid #333333; margin: 8px 0; }
        .dashed-divider { border-top: 1px dashed #ccc; margin: 6px 0; }
        .bold { font-weight: bold; }
        table { width: 100%; font-size: 11px; margin: 6px 0; }
        td, th { padding: 3px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="gs-bold KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 10px; color:#555;" data-label="address">{{address}}</span><br>
        SĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 13px;">HÓA ĐƠN BÁN LẺ</div>
    <div class="text-center" style="font-size: 10px; color:#666;">HĐ số: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="dashed-divider"></div>
    <div style="font-size: 10px; color:#555;">
        <div>Ngày mua: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Nhân viên: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #eee;">
                <th align="left">Sản phẩm</th>
                <th align="right">SL</th>
                <th align="right">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="bold KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right"><span class="bold KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between;">
        <span>Tiền hàng:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span>Thuế VAT:</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:14px; color: #000; margin-top:4px;" class="bold">
        <span>TỔNG TIỀN:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
    </div>
    <div class="dashed-divider"></div>
    <div style="font-size: 11px;">
        Hình thức: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 15px; font-size:9px; color:#888;">
        <p>Cảm ơn & Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 7: 7-Eleven Convenience Store Style (Grayscale)
CONVENIENCE_7ELEVEN = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000000;
            font-size: 12px; line-height: 1.4; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .brand-logo { font-size: 24px; font-weight: bold; color: #000000; letter-spacing: -0.5px; }
        .divider { border-top: 1px solid #000000; margin: 8px 0; }
        table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 11px; }
        td { padding: 4px 0; vertical-align: top; }
        .reference-box { border: 1px solid #ff3300; border-radius: 8px; padding: 10px; margin: 10px 0; font-family: sans-serif; text-align: center; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <!-- Brand Logo -->
        <span class="brand-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="store-info" style="font-size: 11px;">1095 - Cobi Tower II D7</span><br>
        <span style="font-size: 11px;">Ngay <span class="KIE-date kie-field" data-label="date">{{date}}</span></span>
    </div>
    
    <div class="text-center bold" style="font-size: 13px; margin-top: 6px;">HÓA ĐƠN BÁN HÀNG</div>
    
    <table>
        <tbody>
            {% for item in items %}
            <tr>
                <td style="width: 8%;">{{item.qty}}</td>
                <td style="width: 62%;"><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td style="width: 30%; text-align: right;" {% if loop.last %}style="border-bottom: 1px solid #000000; padding-bottom: 2px;"{% endif %}><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div style="font-size: 11px; line-height: 1.6; margin-top: 5px;">
        {% if discount_amount and discount_amount > 0 %}
        <div style="display: flex; justify-content: space-between;">
            <span>GIẢM GIÁ</span>
            <span>{{ "{:,.0f}".format(discount_amount) }}</span>
        </div>
        {% endif %}
        <div style="display: flex; justify-content: space-between;" class="bold">
            <span>TỔNG CỘNG</span>
            <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>TIỀN MẶT</span>
            <span>{{ "{:,.0f}".format(total_amount) }}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>7REWARDS ***Tuan</span>
            <span>+6 Diem</span>
        </div>
        <div style="font-size: 10px;">Tai 7REWARDS de doi diem lay voucher.</div>
    </div>
    
    <div class="divider"></div>
    
    <div style="font-size: 10px; line-height: 1.4;">
        <span class="bold">Cac chuong trinh giam gia da ap dung:</span><br>
        <div style="display: flex; justify-content: space-between;">
            <span>Mon Thai + Sprite PET 390</span>
            <span>-8,000</span>
        </div>
        <span>(Songkran Combo)</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="text-center" style="font-size: 11px;">
        <span>Co gia tri xuat HD VAT trong ngay.</span><br>
        <span class="bold" style="font-size: 12px; display: block; margin-top: 4px;">1095010-0117-2100-{{invoice_no[2:6]}}</span>
    </div>
    
    <!-- Reference Box -->
    <div class="reference-box">
        <div style="font-size: 12px; font-weight: bold; color: #000;">Ma tham chieu:</div>
        <div style="font-size: 14px; font-weight: bold; color: #000; margin-top: 4px; font-family: monospace;">1095001172{{invoice_no[2:10]}}</div>
    </div>
    
    <div style="display: flex; align-items: flex-start; font-size: 10px; font-family: sans-serif; margin-top: 8px;">
        <!-- Mock QR Code -->
        <div style="width: 50px; height: 50px; background: repeating-conic-gradient(#000 0% 25%, #fff 0% 50%) 50% / 5px 5px; border: 1px solid #000; margin-right: 10px; flex-shrink: 0;"></div>
        <div>
            <div class="bold" style="font-size: 11px;">Ho tro truc tuyen</div>
            <span>Quet QR nay hoac den 7now.vn/hotro de:</span>
            <ul style="margin: 3px 0 0 15px; padding: 0;">
                <li>Xuat hoa don VAT</li>
                <li>Chi tiet don hang & gui phan hoi</li>
                <li>Tai app tich diem & giao hang</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Template 8: Highlands Coffee Style (Grayscale standardized)
CAFE_HIGHLANDS = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000000;
            font-size: 12px; line-height: 1.4; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000000; margin: 8px 0; }
        .brand-logo { font-size: 18px; font-weight: bold; color: #000; text-transform: uppercase; }
        .store-info { font-size: 11px; display: block; }
        .title { font-size: 14px; font-weight: bold; color: #000; margin: 8px 0 4px 0; display: block; }
        table { width: 100%; border-collapse: collapse; margin: 8px 0; }
        td { padding: 3px 0; vertical-align: top; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="brand-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="store-info KIE-address kie-field" data-label="address">{{address}}</span>
        <span class="store-info">SDT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></span>
        <span class="store-info">ShopID: 30403</span>
        <span class="title">Hoa Don Thanh Toan</span>
    </div>
    
    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
        <span class="bold">In Store</span>
        <span class="bold">Pager: 01</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>Check: <span class="KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no[2:8]}}</span></span>
        <span>POS01</span>
    </div>
    <div>Ngay : <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
    <div>Thu ngan: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    
    <div class="divider"></div>
    
    <table>
        <tbody>
            {% for item in items %}
            <tr>
                <td style="width: 8%;"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span></td>
                <td style="width: 62%;"><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td style="width: 30%; text-align: right;"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            <!-- Giả lập dòng chiết khấu / giảm giá của món nếu có -->
            {% if item.discount_amount and item.discount_amount > 0 %}
            <tr style="font-size: 11px;">
                <td></td>
                <td>-MER.MEGASALE_Dis5</td>
                <td style="text-align: right;">-{{ "{:,.0f}".format(item.discount_amount) }}</td>
            </tr>
            {% endif %}
            {% endfor %}
        </tbody>
    </table>
    
    <div class="divider"></div>
    
    <div style="display: flex; justify-content: space-between;">
        <span>Thanh tien:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 16px;" class="bold">
        <span>Tong tien:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>  <span class="KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span></span>
        <span>{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="text-center" style="font-size: 11px; margin-top: 15px;">
        <span>* Chia se y kien cua ban voi chung toi*</span><br>
        <span>customerservice@highlandscoffee.com.vn</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="text-center" style="font-size: 11px;">
        <div>Phương thức: <span class="KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span></div>
        <p style="margin-top: 15px; font-style: italic; color: #555;">Cám ơn Quý khách - Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 9: Phuc Long Tea Style (Grayscale standardized)
CAFE_PHUCLONG = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000000;
            font-size: 12px; line-height: 1.4; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000000; margin: 8px 0; }
        .double-divider { border-top: 2px double #000000; margin: 8px 0; }
        .pl-logo { font-size: 16px; font-weight: bold; color: #000; text-transform: uppercase; display: block; margin-bottom: 2px; }
        .store-info { font-size: 11px; display: block; }
        .title { font-size: 14px; font-weight: bold; color: #000; margin: 8px 0; display: block; }
        table { width: 100%; border-collapse: collapse; margin: 6px 0; font-size: 11px; }
        td, th { padding: 3px 0; vertical-align: top; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <!-- Logo Text -->
        <span class="pl-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
        <span class="store-info KIE-address kie-field" data-label="address">{{address}}</span>
        <div class="divider"></div>
        <span class="title">PHIEU THANH TOAN</span>
        <span class="bold">TAG: Q#2</span>
    </div>
    
    <div style="font-size: 11px; margin-top: 6px;">
        <div style="display: flex; justify-content: space-between;">
            <span>Trans#: <span class="KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no[2:8]}}</span></span>
            <span>Serv: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span><span class="KIE-date kie-field" data-label="date">{{date}}</span></span>
            <span>#Cust: 1</span>
        </div>
    </div>
    
    <div class="double-divider"></div>
    
    <table>
        <thead>
            <tr>
                <th align="left" style="width: 10%;">Quan</th>
                <th align="left" style="width: 60%;">Description</th>
                <th align="right" style="width: 30%;">Cost</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td colspan="3"><div style="border-top: 1px dashed #000; margin: 3px 0;"></div></td>
            </tr>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span></td>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="double-divider"></div>
    
    <div style="display: flex; justify-content: space-between; font-size: 12px;">
        <span></span>
        <span>Total: {{ "{:,.0f}".format(subtotal) }}</span>
    </div>
    
    <div class="double-divider"></div>
    
    <div style="display: flex; justify-content: space-between; font-size: 14px;" class="bold">
        <span>TOTAL:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    
    <div class="double-divider"></div>
    
    <div style="font-size: 11px;">
        <div style="display: flex; justify-content: space-between;">
            <span>CASH</span>
            <span>{{ "{:,.0f}".format(total_amount) }}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Change</span>
            <span>0</span>
        </div>
    </div>
    
    <div class="text-center bold" style="margin-top: 10px;">01</div>
    
    <div class="text-center" style="font-size: 11px; margin-top: 8px;">
        <span>ROSE TEA</span><br>
        <span>Tea Of Love</span>
    </div>
    
    <div class="text-center" style="font-size: 11px; margin-top: 8px;">
        <span>-----***-----</span><br>
        <span>Thank you, see you again</span><br>
        <span>Please ask our staff for VAT Invoice today</span><br>
        <span>Quy khach can hoa don tai chinh,</span><br>
        <span>vui long cung cap thong tin trong ngay.</span><br>
        <span>-----***-----</span>
    </div>
    
    <div class="text-center" style="font-size: 11px; margin-top: 8px;">
        <span>Free wifi access code:</span><br>
        <span class="bold">phuclong.com.vn</span>
    </div>
</body>
</html>
"""


# Template 10: Starbucks Style (Black/White minimalist theme - Already grayscale)
CAFE_STARBUCKS = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000000;
            font-size: 12px; line-height: 1.4; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000000; margin: 8px 0; }
        .brand-logo { font-size: 15px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }
        table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 11px; }
        th { padding: 5px 0; text-align: left; text-transform: uppercase; }
        td { padding: 6px 0; }
        .summary-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 11px; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <!-- Circular Logo Simulator -->
    <div style="display: flex; justify-content: center; margin-bottom: 8px;">
        <div style="width: 55px; height: 55px; border-radius: 50%; border: 2.5px solid #000000; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: 900; font-size: 7px; text-align: center; line-height: 1.1; letter-spacing: 0.5px;">
            <span>STARBUCKS</span>
            <span style="font-size: 10px; margin: 1px 0;">★</span>
            <span>COFFEE</span>
        </div>
    </div>

    <div class="text-center" style="font-size: 11px;">
        <span class="brand-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="KIE-address kie-field" data-label="address">{{address}}</span><br>
        Tel:<span class="KIE-phone kie-field" data-label="phone">{{phone}}</span>
    </div>
    
    <div class="divider"></div>
    
    <div style="font-size: 11px; line-height: 1.5;">
        <div>Date:<span class="KIE-date kie-field" data-label="date">{{date.split(' ')[0] if ' ' in date else date}}</span> &nbsp;Time:<span>{{date.split(' ')[1] if ' ' in date else '07:38'}}</span></div>
        <div>Terminal: &nbsp;0427101</div>
        <div>Partner: &nbsp;&nbsp;<span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
        <div>Cust.Name:</div>
    </div>
    
    <table>
        <thead>
            <tr style="border-top: 1px dashed #000000; border-bottom: 1px dashed #000000;">
                <th style="width: 60%;">ITEM NAME</th>
                <th style="width: 15%; text-align: center;">QTY</th>
                <th style="width: 25%; text-align: right;">AMOUNT</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="center"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>.00</td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div style="border-top: 1px dashed #000000; margin: 4px 0;"></div>
    
    <div class="summary-row">
        <span>Subtotal:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}</span>
    </div>
    <div class="summary-row">
        <span>Total tax:</span>
        <span class="KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}</span>
    </div>
    <div class="summary-row bold" style="font-size: 13px;">
        <span>Total:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="summary-row">
        <span>Cash</span>
        <span>{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="summary-row">
        <span>Change back (Cash)</span>
        <span>0</span>
    </div>
    
    <div class="text-center" style="margin-top: 15px; font-size: 11px; line-height: 1.4;">
        <span>Thank you!</span><br>
        <span>Please ask our staff for VAT invoice today</span><br>
        <span>Quy khach can hoa don tai chinh,</span><br>
        <span>vui long cung cap thong tin trong ngay.</span>
    </div>
    
    <div class="divider"></div>
    
    <div class="text-center" style="font-size: 11px;">
        <span>Free wifi access code:</span><br>
        <span class="bold">1577152971</span><br>
        <span>Valid for one hour only.</span>
    </div>
</body>
</html>
"""

# Template 11: KFC Fast Food Restaurant Style (Grayscale standardized)
RESTAURANT_KFC = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Montserrat', sans-serif; font-weight: bold;
            margin: 0; padding: 15px; background-color: #ffffff; color: #111;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .kfc-logo { font-size: 26px; font-weight: 900; color: #000; letter-spacing: 1px; margin-bottom: 2px; }
        .divider { border-top: 3px double #111; margin: 8px 0; }
        .sub-divider { border-top: 1px dashed #ccc; margin: 6px 0; }
        table { width: 100%; font-size: 11px; font-family: 'Inter', sans-serif; }
        td, th { padding: 4px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="kfc-logo KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-family:Arial; font-size:10px; color:#555;" data-label="address">{{address}}</span><br>
        <span style="font-family:Arial; font-size:10px;">ĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 14px;">PHIẾU THU TIỀN</div>
    <div class="text-center" style="font-family:Arial; font-size: 10px;">Hóa đơn số: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="sub-divider"></div>
    <table style="font-family:Arial; font-size: 10px; color:#444;">
        <tr>
            <td>Thời gian: <span class="KIE-date kie-field" data-label="date">{{date}}</span></td>
            <td align="right">Thu ngân: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></td>
        </tr>
    </table>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #111;">
                <th align="left">Món ăn</th>
                <th align="right">SL</th>
                <th align="right">Đơn giá</th>
                <th align="right">Tổng cộng</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="bold KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span></td>
                <td align="right"><span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <table style="font-size:12px;">
        <tr>
            <td>Tiền hàng:</td>
            <td align="right">{{ "{:,.0f}".format(subtotal) }}đ</td>
        </tr>
        <tr>
            <td>Thuế VAT ({{vat_rate}}%):</td>
            <td align="right"><span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span></td>
        </tr>
        <tr style="font-size:15px; font-weight:bold; color:#000;">
            <td>TỔNG CỘNG TIỀN:</td>
            <td align="right"><span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>đ</td>
        </tr>
    </table>
    <div class="sub-divider"></div>
    <div style="font-family:Arial; font-size:11px;">
        Thanh toán: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 15px; font-family:Arial; font-size:10px; color:#555;">
        <p>Chúc Quý khách ngon miệng!</p>
    </div>
</body>
</html>
"""

# Template 12: Jollibee Restaurant Style (Grayscale standardized)
RESTAURANT_JOLLIBEE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Montserrat', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #333;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .jb-header { color: #000; font-weight: bold; font-size: 22px; text-transform: uppercase; }
        .divider { border-top: 2px solid #333; margin: 8px 0; }
        .dashed-divider { border-top: 1px dashed #ccc; margin: 6px 0; }
        .bold { font-weight: bold; }
        table { width: 100%; font-size: 11px; margin: 6px 0; }
        td, th { padding: 4px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="jb-header KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 10px; color:#666;" data-label="address">{{address}}</span><br>
        ĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 13px;">HÓA ĐƠN BÁN HÀNG</div>
    <div class="text-center" style="font-size: 10px;">Số phiếu: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="dashed-divider"></div>
    <div style="font-size: 10px; color:#555;">
        <div>Ngày lập: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Thu ngân: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #ccc;">
                <th align="left">Món ăn</th>
                <th align="right">SL</th>
                <th align="right">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="bold KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between;">
        <span>Tổng tiền hàng:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span>Thuế VAT:</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:14px; color:#000; margin-top:4px;" class="bold">
        <span>TỔNG CỘNG:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
    </div>
    <div class="dashed-divider"></div>
    <div style="font-size: 11px;">
        Thanh toán: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 15px; font-size:10px; color:#555;">
        <p>Cảm ơn & Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 13: Viettel Style E-Invoice (Grayscale standardized)
EINVOICE_VIETTEL = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <title>Hóa đơn giá trị gia tăng</title>
    <style>
        body {
            width: 794px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 30px; background-color: #ffffff; color: #212529;
            font-size: 13px; line-height: 1.5; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .header-table { width: 100%; border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 20px; }
        .logo-container { font-size: 24px; font-weight: 800; color: #000; }
        .store-info { font-size: 12px; color: #495057; }
        .invoice-title-container { text-align: center; margin-bottom: 25px; }
        .invoice-title { font-size: 22px; font-weight: 800; color: #000; margin-bottom: 5px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; border: 1px solid #dee2e6; padding: 15px; border-radius: 6px; }
        .info-section h3 { margin-top: 0; margin-bottom: 10px; font-size: 14px; color: #000; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; }
        .info-item { margin-bottom: 6px; display: flex; }
        .info-label { font-weight: bold; width: 120px; flex-shrink: 0; }
        .info-value { color: #495057; }
        table.items-table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
        table.items-table th { background-color: #333333; color: #ffffff; font-weight: 600; padding: 10px; text-align: left; border: 1px solid #333333; }
        table.items-table td { padding: 10px; border: 1px solid #dee2e6; }
        table.items-table tr:nth-child(even) { background-color: #f8f9fa; }
        .summary-container { display: flex; justify-content: flex-end; margin-bottom: 30px; }
        .summary-table { width: 320px; border-collapse: collapse; }
        .summary-table td { padding: 6px 10px; border: 1px solid #dee2e6; }
        .summary-table td.label { font-weight: bold; background-color: #f8f9fa; }
        .footer-sig { display: grid; grid-template-columns: 1fr 1fr; text-align: center; margin-top: 40px; }
        .signature-box { height: 100px; display: flex; align-items: center; justify-content: center; color: #adb5bd; border: 1px dashed #dee2e6; margin-top: 10px; border-radius: 4px; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <table class="header-table">
        <tr>
            <td class="logo-container" style="vertical-align: middle;">
                <span class="KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
            </td>
            <td class="store-info text-right" style="vertical-align: middle;">
                <div>Địa chỉ: <span class="KIE-address kie-field" data-label="address">{{address}}</span></div>
                <div>Điện thoại: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></div>
                <div>Mã số thuế: <span class="KIE-tax_code kie-field" data-label="tax_code">{{tax_code}}</span></div>
            </td>
        </tr>
    </table>
    
    <div class="invoice-title-container">
        <div class="invoice-title">HÓA ĐƠN GIÁ TRỊ GIA TĂNG</div>
        <div>(BẢN THỂ HIỆN HÓA ĐƠN ĐIỆN TỬ)</div>
        <div style="margin-top: 5px;">
            <span class="bold">Ngày lập:</span> <span class="KIE-date kie-field" data-label="date">{{date}}</span>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="bold">Số:</span> <span class="bold KIE-invoice_no kie-field" data-label="invoice_no" style="color: #000;">{{invoice_no}}</span>
        </div>
    </div>
    
    <div class="info-grid">
        <div class="info-section">
            <h3>Thông tin đơn vị bán hàng</h3>
            <div class="info-item">
                <div class="info-label">Đơn vị:</div>
                <div class="info-value bold">{{store_name}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Mã số thuế:</div>
                <div class="info-value">{{tax_code}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Người lập:</div>
                <div class="info-value KIE-cashier kie-field" data-label="cashier">{{cashier}}</div>
            </div>
        </div>
        <div class="info-section">
            <h3>Thông tin khách hàng mua hàng</h3>
            <div class="info-item">
                <div class="info-label">Khách hàng:</div>
                <div class="info-value">Khách mua hàng lẻ (Khách vãng lai)</div>
            </div>
            <div class="info-item">
                <div class="info-label">Thanh toán:</div>
                <div class="info-value KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Trạng thái:</div>
                <div class="info-value bold" style="color: #000;">ĐÃ THANH TOÁN</div>
            </div>
        </div>
    </div>
    
    <table class="items-table">
        <thead>
            <tr>
                <th style="width: 5%; text-align: center;">STT</th>
                <th style="width: 45%;">Tên hàng hóa, dịch vụ</th>
                <th style="width: 10%; text-align: center;">ĐVT</th>
                <th style="width: 10%; text-align: right;">Số lượng</th>
                <th style="width: 15%; text-align: right;">Đơn giá</th>
                <th style="width: 15%; text-align: right;">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td style="text-align: center;">{{item.no}}</td>
                <td>
                    <span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span>
                </td>
                <td style="text-align: center;">{{item.unit}}</td>
                <td style="text-align: right;">
                    <span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>
                </td>
                <td style="text-align: right;">
                    <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span>
                </td>
                <td style="text-align: right; font-weight: bold;">
                    <span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="summary-container">
        <table class="summary-table">
            <tr>
                <td class="label">Cộng tiền hàng:</td>
                <td class="text-right" style="font-weight: 500;">{{ "{:,.0f}".format(subtotal) }}đ</td>
            </tr>
            <tr>
                <td class="label">Thuế suất VAT:</td>
                <td class="text-right">{{vat_rate}}%</td>
            </tr>
            <tr>
                <td class="label">Tiền thuế VAT:</td>
                <td class="text-right"><span class="KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span></td>
            </tr>
            <tr>
                <td class="label" style="background-color: #f8f9fa;">Tổng cộng tiền:</td>
                <td class="text-right bold" style="background-color: #f8f9fa; font-size: 15px;">
                    <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
                </td>
            </tr>
        </table>
    </div>
    
    <div class="footer-sig">
        <div>
            <div class="bold">NGƯỜI MUA HÀNG</div>
            <div style="font-size: 11px; color: #6c757d;">(Ký, ghi rõ họ tên)</div>
            <div class="signature-box">Ký điện tử</div>
        </div>
        <div>
            <div class="bold">NGƯỜI BÁN HÀNG</div>
            <div style="font-size: 11px; color: #6c757d;">(Ký, ghi rõ họ tên, đóng dấu)</div>
            <div class="signature-box" style="border-color: #333; color: #333; font-weight: bold;">
                {{store_name}}<br>
                Ký bởi Hệ thống Thu ngân
            </div>
        </div>
    </div>
</body>
</html>
"""

# Template 14: VNPT Style E-Invoice (Grayscale standardized)
EINVOICE_VNPT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <title>Hóa đơn giá trị gia tăng</title>
    <style>
        body {
            width: 794px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 30px; background-color: #ffffff; color: #212529;
            font-size: 13px; line-height: 1.5; box-sizing: border-box;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .header-table { width: 100%; border-bottom: 2px solid #555; padding-bottom: 15px; margin-bottom: 20px; }
        .logo-container { font-size: 24px; font-weight: 800; color: #333; }
        .store-info { font-size: 12px; color: #495057; }
        .invoice-title-container { text-align: center; margin-bottom: 25px; }
        .invoice-title { font-size: 22px; font-weight: 800; color: #333; margin-bottom: 5px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; border: 1px solid #dee2e6; padding: 15px; border-radius: 6px; }
        .info-section h3 { margin-top: 0; margin-bottom: 10px; font-size: 14px; color: #333; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; }
        .info-item { margin-bottom: 6px; display: flex; }
        .info-label { font-weight: bold; width: 120px; flex-shrink: 0; }
        .info-value { color: #495057; }
        table.items-table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
        table.items-table th { background-color: #555555; color: #ffffff; font-weight: 600; padding: 10px; text-align: left; border: 1px solid #555555; }
        table.items-table td { padding: 10px; border: 1px solid #dee2e6; }
        table.items-table tr:nth-child(even) { background-color: #f8f9fa; }
        .summary-container { display: flex; justify-content: flex-end; margin-bottom: 30px; }
        .summary-table { width: 320px; border-collapse: collapse; }
        .summary-table td { padding: 6px 10px; border: 1px solid #dee2e6; }
        .summary-table td.label { font-weight: bold; background-color: #f8f9fa; }
        .footer-sig { display: grid; grid-template-columns: 1fr 1fr; text-align: center; margin-top: 40px; }
        .signature-box { height: 100px; display: flex; align-items: center; justify-content: center; color: #adb5bd; border: 1px dashed #dee2e6; margin-top: 10px; border-radius: 4px; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <table class="header-table">
        <tr>
            <td class="logo-container" style="vertical-align: middle;">
                <span class="KIE-store_name kie-field" data-label="store_name">{{store_name}}</span>
            </td>
            <td class="store-info text-right" style="vertical-align: middle;">
                <div>Địa chỉ: <span class="KIE-address kie-field" data-label="address">{{address}}</span></div>
                <div>Điện thoại: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></div>
                <div>Mã số thuế: <span class="KIE-tax_code kie-field" data-label="tax_code">{{tax_code}}</span></div>
            </td>
        </tr>
    </table>
    
    <div class="invoice-title-container">
        <div class="invoice-title">HÓA ĐƠN GIÁ TRỊ GIA TĂNG</div>
        <div>(BẢN THỂ HIỆN HÓA ĐƠN ĐIỆN TỬ)</div>
        <div style="margin-top: 5px;">
            <span class="bold">Ngày lập:</span> <span class="KIE-date kie-field" data-label="date">{{date}}</span>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <span class="bold">Số:</span> <span class="bold KIE-invoice_no kie-field" data-label="invoice_no" style="color: #333;">{{invoice_no}}</span>
        </div>
    </div>
    
    <div class="info-grid">
        <div class="info-section">
            <h3>Thông tin đơn vị bán hàng</h3>
            <div class="info-item">
                <div class="info-label">Đơn vị:</div>
                <div class="info-value bold">{{store_name}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Mã số thuế:</div>
                <div class="info-value">{{tax_code}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Người lập:</div>
                <div class="info-value KIE-cashier kie-field" data-label="cashier">{{cashier}}</div>
            </div>
        </div>
        <div class="info-section">
            <h3>Thông tin khách hàng mua hàng</h3>
            <div class="info-item">
                <div class="info-label">Khách hàng:</div>
                <div class="info-value">Khách mua hàng lẻ (Khách vãng lai)</div>
            </div>
            <div class="info-item">
                <div class="info-label">Thanh toán:</div>
                <div class="info-value KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Trạng thái:</div>
                <div class="info-value bold" style="color: #555;">ĐÃ THANH TOÁN</div>
            </div>
        </div>
    </div>
    
    <table class="items-table">
        <thead>
            <tr>
                <th style="width: 5%; text-align: center;">STT</th>
                <th style="width: 45%;">Tên hàng hóa, dịch vụ</th>
                <th style="width: 10%; text-align: center;">ĐVT</th>
                <th style="width: 10%; text-align: right;">Số lượng</th>
                <th style="width: 15%; text-align: right;">Đơn giá</th>
                <th style="width: 15%; text-align: right;">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td style="text-align: center;">{{item.no}}</td>
                <td>
                    <span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span>
                </td>
                <td style="text-align: center;">{{item.unit}}</td>
                <td style="text-align: right;">
                    <span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span>
                </td>
                <td style="text-align: right;">
                    <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span>
                </td>
                <td style="text-align: right; font-weight: bold;">
                    <span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="summary-container">
        <table class="summary-table">
            <tr>
                <td class="label">Cộng tiền hàng:</td>
                <td class="text-right" style="font-weight: 500;">{{ "{:,.0f}".format(subtotal) }}đ</td>
            </tr>
            <tr>
                <td class="label">Thuế suất VAT:</td>
                <td class="text-right">{{vat_rate}}%</td>
            </tr>
            <tr>
                <td class="label">Tiền thuế VAT:</td>
                <td class="text-right"><span class="KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span></td>
            </tr>
            <tr>
                <td class="label" style="background-color: #f2f2f2;">Tổng cộng tiền:</td>
                <td class="text-right bold" style="background-color: #f2f2f2; font-size: 15px;">
                    <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
                </td>
            </tr>
        </table>
    </div>
    
    <div class="footer-sig">
        <div>
            <div class="bold">NGƯỜI MUA HÀNG</div>
            <div style="font-size: 11px; color: #6c757d;">(Ký, ghi rõ họ tên)</div>
            <div class="signature-box">Ký điện tử</div>
        </div>
        <div>
            <div class="bold">NGƯỜI BÁN HÀNG</div>
            <div style="font-size: 11px; color: #6c757d;">(Ký, ghi rõ họ tên, đóng dấu)</div>
            <div class="signature-box" style="border-color: #555; color: #555; font-weight: bold;">
                {{store_name}}<br>
                Ký bởi Hệ thống Thu ngân
            </div>
        </div>
    </div>
</body>
</html>
"""

# Template 15: ShopeeFood Delivery Receipt Style (Grayscale standardized)
DELIVERY_SHOPEEFOOD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #333;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .shopee-bold { color: #000; font-weight: 800; font-size: 20px; text-transform: uppercase; }
        .divider { border-top: 2px solid #333; margin: 8px 0; }
        .dashed-divider { border-top: 1px dashed #333; margin: 6px 0; }
        .bold { font-weight: bold; }
        .delivery-box { background-color: #f2f2f2; border: 1px solid #333; border-radius: 4px; padding: 8px; margin: 8px 0; font-size: 11px; }
        table { width: 100%; font-size: 11px; margin: 6px 0; }
        td, th { padding: 4px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="shopee-bold KIE-store_name kie-field" data-label="store_name">{{store_name}}</span><br>
        <span style="font-size: 10px; color:#666;">Đại lý ủy quyền Delivery</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 13px; color: #000;">ĐƠN GIAO HÀNG / DELIVERY SLIP</div>
    <div class="text-center" style="font-size: 10px;">Mã đơn: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    
    <div class="delivery-box">
        <div class="bold" style="color: #000; margin-bottom: 4px;">ĐỊA CHỈ GIAO HÀNG:</div>
        <div>Địa chỉ: <span class="bold KIE-address kie-field" data-label="address">{{address}}</span></div>
        <div>Số điện thoại: <span class="bold KIE-phone kie-field" data-label="phone">{{phone}}</span></div>
    </div>
    
    <div style="font-size: 10px; color:#555;">
        <div>Thời gian đặt: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Tài xế nhận: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></div>
    </div>
    
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #eee;">
                <th align="left">Món ăn</th>
                <th align="right">SL</th>
                <th align="right">Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="bold KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between;">
        <span>Tiền hàng món:</span>
        <span>{{ "{:,.0f}".format(subtotal) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span>Phí VAT:</span>
        <span class="bold KIE-vat_amount kie-field" data-label="vat_amount">{{ "{:,.0f}".format(vat_amount) }}đ</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:14px; color:#000; margin-top:4px;" class="bold">
        <span>TỔNG KHÁCH TRẢ:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}đ</span>
    </div>
    <div class="dashed-divider"></div>
    <div style="font-size: 11px;">
        Trạng thái: <span class="bold KIE-payment_method kie-field" data-label="payment_method">{{payment_method}}</span>
    </div>
    <div class="text-center" style="margin-top: 15px; font-size:9px; color:#777;">
        <p>Cảm ơn bạn đã lựa chọn dịch vụ!</p>
    </div>
</body>
</html>
"""

# Template 16: MC-OCR Minimart AnAn (New)
MINIMART_ANAN = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 12px; line-height: 1.3;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000; margin: 8px 0; }
        table { width: 100%; font-size: 11px; border-collapse: collapse; }
        th, td { padding: 3px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="bold KIE-store_name kie-field" data-label="store_name" style="font-size: 16px;">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 11px; display: block; margin: 4px 0;" data-label="address">{{address}}</span>
        <span>SĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 13px;">HÓA ĐƠN BÁN LẺ</div>
    <div class="text-center" style="font-size: 10px;">Số phiếu: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="divider"></div>
    <table style="font-size: 10px;">
        <tr>
            <td>Ngày bán: <span class="KIE-date kie-field" data-label="date">{{date}}</span></td>
            <td align="right">TN: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></td>
        </tr>
    </table>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #000;">
                <th align="left">Tên hàng</th>
                <th align="right">SL</th>
                <th align="right">T.Tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between; font-size: 13px; font-weight: bold; margin-top: 4px;">
        <span>TỔNG CỘNG:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center" style="margin-top: 15px; font-size: 10px;">
        <p>Cảm ơn Quý khách. Hẹn gặp lại!</p>
    </div>
</body>
</html>
"""

# Template 17: MC-OCR Nhasach CamPha (New)
NHASACH_CAMPHA = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 12px; line-height: 1.4;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 2px solid #000; margin: 8px 0; }
        .sub-divider { border-top: 1px dashed #999; margin: 6px 0; }
        table { width: 100%; font-size: 11px; }
        th, td { padding: 4px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="bold KIE-store_name kie-field" data-label="store_name" style="font-size: 15px;">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 10px; display: block; margin-top: 4px;" data-label="address">{{address}}</span>
        <span>MST: <span class="KIE-tax_code kie-field" data-label="tax_code">{{tax_code}}</span></span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 14px;">HÓA ĐƠN THANH TOÁN</div>
    <div class="text-center" style="font-size: 10px;">Số phiếu: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="sub-divider"></div>
    <div style="font-size: 10px; display: flex; justify-content: space-between;">
        <span>Ngày: <span class="KIE-date kie-field" data-label="date">{{date}}</span></span>
        <span>TN: <span class="KIE-cashier kie-field" data-label="cashier">{{cashier}}</span></span>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px solid #000;">
                <th align="left">Tên sản phẩm</th>
                <th align="right">SL</th>
                <th align="right">T.Tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span> x <span class="KIE-item_price-{{item.no}} kie-field" data-label="item_price_{{item.no}}">{{ "{:,.0f}".format(item.price) }}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between; font-size: 14px; font-weight: bold;">
        <span>TỔNG CỘNG:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="sub-divider"></div>
    <div class="text-center" style="margin-top: 15px; font-size: 10px; color:#555;">
        <p>Cảm ơn Quý khách!</p>
    </div>
</body>
</html>
"""

# Template 18: MC-OCR CuaHang NamOanh (New)
CUAHANG_NAMOANH = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Courier Prime', monospace;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 13px; line-height: 1.3;
        }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000; margin: 8px 0; }
        table { width: 100%; font-size: 12px; }
        th, td { padding: 3px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="bold KIE-store_name kie-field" data-label="store_name" style="font-size: 17px;">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 11px; display: block; margin: 4px 0;" data-label="address">{{address}}</span>
        <span>SĐT: <span class="KIE-phone kie-field" data-label="phone">{{phone}}</span></span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold">PHIẾU BÁN HÀNG</div>
    <div class="divider"></div>
    <div style="font-size: 11px;">
        <div>Ngày lập: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Số HĐ: <span class="KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    </div>
    <div class="divider"></div>
    <table>
        <thead>
            <tr style="border-bottom: 1px dashed #000;">
                <th align="left">Tên hàng</th>
                <th align="right">SL</th>
                <th align="right">T.Tiền</th>
            </tr>
        </thead>
        <tbody>
            {% for item in items %}
            <tr>
                <td><span class="KIE-item_name-{{item.no}} kie-field" data-label="item_name_{{item.no}}">{{item.name}}</span></td>
                <td align="right"><span class="KIE-item_qty-{{item.no}} kie-field" data-label="item_qty_{{item.no}}">{{item.qty}}</span></td>
                <td align="right" class="bold"><span class="KIE-item_amount-{{item.no}} kie-field" data-label="item_amount_{{item.no}}">{{ "{:,.0f}".format(item.amount) }}</span></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between; font-size: 14px; font-weight: bold;">
        <span>TỔNG TIỀN:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center" style="margin-top: 15px; font-size: 10px;">
        <p>Hẹn gặp lại quý khách!</p>
    </div>
</body>
</html>
"""

# Template 19: Bank POS Slip (New)
POS_SLIP = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 350px; font-family: 'Roboto Mono', monospace;
            margin: 0; padding: 12px; background-color: #ffffff; color: #000;
            font-size: 11px; line-height: 1.3;
        }
        .text-center { text-align: center; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px dashed #000; margin: 6px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="bold KIE-store_name kie-field" data-label="store_name" style="font-size: 14px;">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 10px; display: block;" data-label="address">{{address}}</span>
        <span style="font-size: 12px; font-weight: bold; margin-top: 4px; display: block;">HÓA ĐƠN THANH TOÁN THẺ (POS)</span>
    </div>
    <div class="divider"></div>
    <div>TERMINAL ID: <span>TERM{{invoice_no[2:8]}}</span></div>
    <div>MERCHANT ID: <span>MERCH{{tax_code[:8]}}</span></div>
    <div>CARD TYPE: <span class="bold">VISA / DOMESTIC</span></div>
    <div>CARD NUMBER: <span>**** **** **** {{invoice_no[2:6]}}</span></div>
    <div class="divider"></div>
    <div>DATE/TIME: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
    <div>TRACE NO: <span>{{invoice_no[3:9]}}</span></div>
    <div>REF NO: <span>{{tax_code}}</span></div>
    <div class="divider"></div>
    <div style="display:flex; justify-content:space-between; font-size: 13px; font-weight: bold;">
        <span>AMOUNT:</span>
        <span class="KIE-total_amount kie-field" data-label="total_amount">VND {{ "{:,.0f}".format(total_amount) }}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center" style="margin-top: 15px; font-size: 9px;">
        <span>NO SIGNATURE REQUIRED</span><br>
        <span>THANK YOU & SEE YOU AGAIN</span>
    </div>
</body>
</html>
"""

# Template 20: Toll Ticket (New)
TOLL_TICKET = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 380px; font-family: 'Inter', sans-serif;
            margin: 0; padding: 15px; background-color: #ffffff; color: #000;
            font-size: 12px; line-height: 1.4; border: 1px solid #aaa;
        }
        .text-center { text-align: center; }
        .bold { font-weight: bold; }
        .divider { border-top: 1px solid #000; margin: 8px 0; }
        .dashed { border-top: 1px dashed #666; margin: 8px 0; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="text-center">
        <span class="bold KIE-store_name kie-field" data-label="store_name" style="font-size: 14px;">{{store_name}}</span><br>
        <span class="KIE-address kie-field" style="font-size: 10px; display: block;" data-label="address">{{address}}</span>
    </div>
    <div class="divider"></div>
    <div class="text-center bold" style="font-size: 15px; letter-spacing: 1px;">VÉ THU PHÍ DỊCH VỤ ĐƯỜNG BỘ</div>
    <div class="text-center" style="font-size: 10px; margin-top: 2px;">Mã vé: <span class="bold KIE-invoice_no kie-field" data-label="invoice_no">{{invoice_no}}</span></div>
    <div class="dashed"></div>
    <div style="font-size: 11px;">
        <div>Trạm thu phí: <span class="bold">{{store_name.replace("TRẠM THU PHÍ ", "")}}</span></div>
        <div>Thời gian qua trạm: <span class="KIE-date kie-field" data-label="date">{{date}}</span></div>
        <div>Phương tiện: <span>Xe con, xe tải nhẹ dưới 2 tấn</span></div>
    </div>
    <div class="dashed"></div>
    <div class="text-center" style="margin: 10px 0;">
        <span style="font-size: 13px; display: block;">MỆNH GIÁ:</span>
        <span class="bold KIE-total_amount kie-field" data-label="total_amount" style="font-size: 20px;">{{ "{:,.0f}".format(total_amount) }} đồng</span>
        <span style="font-size: 10px; display: block; color: #555;">(Đã bao gồm thuế giá trị gia tăng)</span>
    </div>
    <div class="divider"></div>
    <div class="text-center" style="font-size: 9px; color:#555;">
        <span>Chúc quý khách thượng lộ bình an!</span>
    </div>
</body>
</html>
"""

# Template 21: Ministry of Finance C45-BB dot-matrix printed receipt (New)
RECEIPT_C45_BB = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            width: 650px;
            height: 420px;
            font-family: Arial, sans-serif;
            background-color: #ffffff; color: #333333;
            margin: 0; padding: 20px;
            box-sizing: border-box;
            border: 2px solid #555555;
        }
        .bold { font-weight: bold; }
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .header-section { display: flex; justify-content: space-between; font-size: 12px; line-height: 1.4; }
        .title-section { text-align: center; margin-top: 15px; margin-bottom: 20px; }
        .main-title { font-size: 18px; font-weight: bold; letter-spacing: 1px; }
        .receipt-lines { font-size: 13px; line-height: 1.8; }
        .line-item { display: flex; align-items: flex-end; margin-bottom: 4px; }
        .line-label { flex-shrink: 0; }
        .line-dots { flex-grow: 1; border-bottom: 1px dotted #555555; display: flex; align-items: flex-end; padding-left: 10px; height: 20px; }
        .print-value { font-family: 'Roboto Mono', monospace; font-weight: bold; color: #000000; font-size: 14px; line-height: 1; }
        .signature-section { display: flex; justify-content: space-between; margin-top: 25px; font-size: 12px; text-align: center; }
        .sig-col { width: 30%; }
        .sig-box { height: 50px; }
        .kie-field { position: relative; display: inline-block; }
    </style>
</head>
<body>
    <div class="header-section">
        <div>
            <div class="line-item">
                <span class="line-label bold">Đơn vị:</span>
                <span class="line-dots"><span class="KIE-store_name kie-field print-value" data-label="store_name">{{store_name}}</span></span>
            </div>
            <div class="line-item" style="margin-top: 4px;">
                <span class="line-label bold">Địa chỉ:</span>
                <span class="line-dots"><span class="KIE-address kie-field print-value" data-label="address">{{address}}</span></span>
            </div>
        </div>
        <div class="text-right">
            <span class="bold">Mẫu số C45-BB</span><br>
            <span>(Ban hành theo QĐ số 19/2006/QĐ-BTC)</span><br>
            <span>Số: <span class="bold" style="font-family: 'Roboto Mono', monospace;">{{invoice_no[2:8]}}</span></span>
        </div>
    </div>
    
    <div class="title-section">
        <span class="main-title">BIÊN LAI THU TIỀN</span><br>
        <span class="KIE-date kie-field print-value" data-label="date" style="font-size: 12px; margin-top: 5px;">{{date}}</span>
    </div>
    
    <div class="receipt-lines">
        <div class="line-item">
            <span class="line-label">Họ và tên người nộp tiền:</span>
            <span class="line-dots"><span class="print-value">{{cashier}}</span></span>
        </div>
        <div class="line-item">
            <span class="line-label">Lý do nộp tiền:</span>
            <span class="line-dots"><span class="print-value">Thanh toán cước phí dịch vụ bán lẻ và hóa đơn tổng hợp</span></span>
        </div>
        <div class="line-item">
            <span class="line-label bold">Số tiền thu:</span>
            <span class="line-dots"><span class="KIE-total_amount kie-field print-value" data-label="total_amount">{{ "{:,.0f}".format(total_amount) }} VND</span></span>
        </div>
        <div class="line-item">
            <span class="line-label">Viết bằng chữ:</span>
            <span class="line-dots"><span class="print-value" style="font-size: 12px;">{{total_words}} chẵn</span></span>
        </div>
    </div>
    
    <div class="signature-section">
        <div class="sig-col">
            <span class="bold">Người nộp tiền</span><br>
            <span style="font-size: 10px; font-style: italic;">(Ký, họ tên)</span>
            <div class="sig-box"></div>
        </div>
        <div class="sig-col">
            <span class="bold">Người thu tiền</span><br>
            <span style="font-size: 10px; font-style: italic;">(Ký, họ tên)</span>
            <div class="sig-box"></div>
            <span class="bold" style="font-family: 'Roboto Mono', monospace; font-size:11px;">{{cashier}}</span>
        </div>
        <div class="sig-col">
            <span class="bold">Thủ quỹ</span><br>
            <span style="font-size: 10px; font-style: italic;">(Ký, họ tên)</span>
            <div class="sig-box"></div>
        </div>
    </div>
</body>
</html>
"""

# Map to bind template names to their HTML content strings
TEMPLATES_MAP = {
    "supermarket_winmart": SUPERMARKET_WINMART,
    "supermarket_lotte": SUPERMARKET_LOTTE,
    "supermarket_bachhoaxanh": SUPERMARKET_BACHHOAXANH,
    "convenience_circlek": CONVENIENCE_CIRCLEK,
    "convenience_gs25": CONVENIENCE_GS25,
    "convenience_7eleven": CONVENIENCE_7ELEVEN,
    "cafe_highlands": CAFE_HIGHLANDS,
    "cafe_phuclong": CAFE_PHUCLONG,
    "cafe_starbucks": CAFE_STARBUCKS,
    "restaurant_kfc": RESTAURANT_KFC,
    "restaurant_jollibee": RESTAURANT_JOLLIBEE,
    "einvoice_viettel": EINVOICE_VIETTEL,
    "einvoice_vnpt": EINVOICE_VNPT,
    "delivery_shopeefood": DELIVERY_SHOPEEFOOD,
    "minimart_anan": MINIMART_ANAN,
    "nhasach_campha": NHASACH_CAMPHA,
    "cuahang_namoanh": CUAHANG_NAMOANH,
    "pos_slip": POS_SLIP,
    "toll_ticket": TOLL_TICKET,
    "receipt_c45_bb": RECEIPT_C45_BB
}
