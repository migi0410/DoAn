-- ==============================================================================
-- AVIR-KIE: SCRIPT KHỞI TẠO TOÀN DIỆN CƠ SỞ DỮ LIỆU TRÊN SUPABASE MỚI
-- Copy toàn bộ đoạn script này và dán vào Supabase -> SQL Editor -> Nhấn RUN
-- ==============================================================================

-- 1. BẢNG TÀI KHOẢN NGƯỜI DÙNG (users)
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    organization VARCHAR(255),
    role VARCHAR(50) DEFAULT 'client', -- 'admin', 'accountant', 'cashier', 'client'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. BẢNG LƯU TRỮ HÓA ĐƠN & ĐỐI SOÁT KIỂM TOÁN (saved_receipts)
CREATE TABLE IF NOT EXISTS saved_receipts (
    id VARCHAR(64) PRIMARY KEY,
    created_at VARCHAR(64) NOT NULL,
    seller VARCHAR(255),
    address TEXT,
    receipt_time VARCHAR(64),
    total_cost VARCHAR(64),
    total_amount NUMERIC(15, 2) DEFAULT 0.0,
    item_count INTEGER DEFAULT 0,
    items_json TEXT,                    -- Lưu trữ mảng đối tượng JSON [ITEMS]
    is_valid INTEGER DEFAULT 1,         -- 1: Khớp số học 100%, 0: Lệch số học
    discrepancy NUMERIC(15, 2) DEFAULT 0.0, -- Độ lệch chênh lệch số tiền
    model_id VARCHAR(64),               -- Tên mô hình (qwen3_lora_v2)
    image_url TEXT,                     -- URL ảnh trên Cloud Storage
    notes TEXT,                         -- Ghi chú kế toán
    user_id VARCHAR(64) REFERENCES users(id) ON DELETE SET NULL
);

-- 3. CHỈ MỤC INDEX TĂNG TỐC TRUY VẤN
CREATE INDEX IF NOT EXISTS idx_saved_receipts_user_id ON saved_receipts(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_receipts_created_at ON saved_receipts(created_at);
CREATE INDEX IF NOT EXISTS idx_saved_receipts_seller ON saved_receipts(seller);

-- 4. SEED TÀI KHOẢN DEMO MẶC ĐỊNH (Mật khẩu: 123456)
INSERT INTO users (id, email, password_hash, full_name, organization, role)
VALUES 
    ('usr_winmart', 'ketoan@winmart.vn', '123456', 'Kế toán Trưởng WinMart', 'WinMart Retail Group', 'accountant'),
    ('usr_highlands', 'thungan@highlands.vn', '123456', 'Thu ngân Ca 1 Highlands Coffee', 'Highlands Coffee Vietnam', 'cashier'),
    ('usr_auditor', 'kiemtoan@fpt.edu.vn', '123456', 'Kiểm toán viên Độc lập FPT', 'FPT Auditing & Analytics', 'admin')
ON CONFLICT (email) DO NOTHING;

-- 5. SEED CÁC HÓA ĐƠN MẪU ĐÃ ĐỐI SOÁT
INSERT INTO saved_receipts (
    id, created_at, seller, address, receipt_time, 
    total_cost, total_amount, item_count, items_json, 
    is_valid, discrepancy, model_id, image_url, notes, user_id
)
VALUES 
    (
        'HD-20260915-001', 
        '15/09/2026 14:30:00', 
        'SIÊU THỊ WINMART', 
        'Tầng B1, Vincom Center, 72 Lê Thánh Tôn, Bến Nghé, Q.1, TP.HCM', 
        '18/04/2026 19:15:00', 
        '185.000', 
        185000.0, 
        5, 
        '[{"name": "Sữa Tươi Tiệt Trùng Vinamilk 1L", "qty": "2", "price": "36.000", "amount": "72.000"}, {"name": "Bánh Mì Sandwich Kinh Đô 250g", "qty": "1", "price": "22.000", "amount": "22.000"}, {"name": "Mì Hảo Hảo Tôm Chua Cay 75g", "qty": "5", "price": "4.600", "amount": "23.000"}, {"name": "Trứng Gà Ba Huân Hộp 10 Quả", "qty": "1", "price": "34.000", "amount": "34.000"}, {"name": "Nước Ngọt Coca-Cola Chai 1.5L", "qty": "1", "price": "34.000", "amount": "34.000"}]', 
        1, 
        0.0, 
        'qwen3_lora_v2', 
        '/templates_images/winmart_template.jpg', 
        'Hóa đơn bán lẻ WinMart - Kiểm toán khớp số học 100% (185,000 VND)', 
        'usr_winmart'
    ),
    (
        'HD-20260915-002', 
        '15/09/2026 15:10:00', 
        'HIGHLANDS COFFEE', 
        '29 Lê Duẩn, P. Bến Nghé, Quận 1, TP.HCM', 
        '10/05/2026 10:30:15', 
        '104.000', 
        104000.0, 
        2, 
        '[{"name": "Phin Sữa Đá (L)", "qty": "1", "price": "45.000", "amount": "45.000"}, {"name": "Trà Sen Vàng (L)", "qty": "1", "price": "59.000", "amount": "59.000"}]', 
        1, 
        0.0, 
        'qwen3_lora_v2', 
        '/templates_images/highland_template.jpg', 
        'Hóa đơn đồ uống Highlands Coffee - Cột đơn giá khuyết đã được Prompt v2 xử lý chính xác', 
        'usr_highlands'
    )
ON CONFLICT (id) DO NOTHING;
