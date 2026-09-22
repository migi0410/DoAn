import os
import json
import sqlite3
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
load_dotenv(os.path.join(BASE_DIR, ".env"))
SQLITE_PATH = os.path.join(BASE_DIR, "receipts_history.db")

DB_HOST = os.getenv("SUPABASE_DB_HOST", "aws-0-ap-southeast-1.pooler.supabase.com")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres.qwprbxxxbvueozffqhdd")
DB_PASS = os.getenv("SUPABASE_DB_PASSWORD", "")

USE_POSTGRES = True

def get_connection():
    global USE_POSTGRES
    if USE_POSTGRES:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                connect_timeout=5
            )
            return conn, "postgres"
        except Exception as e:
            print(f"[Database Warning] PostgreSQL connection failed ({e}). Falling back to local SQLite.")
            USE_POSTGRES = False

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def init_database():
    conn, engine = get_connection()
    cur = conn.cursor()
    try:
        if engine == "postgres":
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    organization TEXT,
                    role TEXT DEFAULT 'client',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_receipts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    seller TEXT,
                    address TEXT,
                    receipt_time TEXT,
                    total_cost TEXT,
                    total_amount NUMERIC,
                    item_count INTEGER,
                    items_json TEXT,
                    is_valid INTEGER,
                    discrepancy NUMERIC,
                    model_id TEXT,
                    image_url TEXT,
                    notes TEXT,
                    user_id TEXT
                );
            """)
            cur.execute("""
                INSERT INTO users (id, email, password_hash, full_name, organization, role)
                VALUES 
                    ('usr_winmart', 'ketoan@winmart.vn', '123456', 'Kế toán WinMart', 'WinMart', 'client'),
                    ('usr_highlands', 'thungan@highlands.vn', '123456', 'Thu ngân Highlands', 'Highlands', 'client')
                ON CONFLICT (email) DO NOTHING;
            """)
            # Seed demo receipts if table empty
            cur.execute("SELECT COUNT(*) FROM saved_receipts")
            count = cur.fetchone()[0]
            if count == 0:
                winmart_items = [
                    {"name": "Sữa Tươi Tiệt Trùng Vinamilk 1L", "qty": "2", "price": "36.000", "amount": "72.000"},
                    {"name": "Bánh Mì Sandwich Kinh Đô 250g", "qty": "1", "price": "22.000", "amount": "22.000"},
                    {"name": "Mì Hảo Hảo Tôm Chua Cay 75g", "qty": "5", "price": "4.600", "amount": "23.000"},
                    {"name": "Trứng Gà Ba Huân Hộp 10 Quả", "qty": "1", "price": "34.000", "amount": "34.000"},
                    {"name": "Nước Ngọt Coca-Cola Chai 1.5L", "qty": "1", "price": "34.000", "amount": "34.000"}
                ]
                highland_items = [
                    {"name": "Phin Sữa Đá (L)", "qty": "1", "price": "45.000", "amount": "45.000"},
                    {"name": "Trà Sen Vàng (L)", "qty": "1", "price": "59.000", "amount": "59.000"}
                ]
                cur.execute("""
                    INSERT INTO saved_receipts (id, created_at, seller, address, receipt_time, total_cost, total_amount, item_count, items_json, is_valid, discrepancy, model_id, image_url, notes, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    "HD-20260915-001", "15/09/2026 14:30:00", "SIÊU THỊ WINMART",
                    "Tầng B1, Vincom Center, 72 Lê Thánh Tôn, Bến Nghé, Q.1, TP.HCM",
                    "18/04/2026 19:15:00", "185.000", 185000.0, 5,
                    json.dumps(winmart_items, ensure_ascii=False), 1, 0.0, "qwen3_lora_v2",
                    "/templates_images/winmart_template.jpg", "Hóa đơn siêu thị bán lẻ - Đã đối soát khớp 100%", "usr_winmart"
                ))
                cur.execute("""
                    INSERT INTO saved_receipts (id, created_at, seller, address, receipt_time, total_cost, total_amount, item_count, items_json, is_valid, discrepancy, model_id, image_url, notes, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    "HD-20260915-002", "15/09/2026 15:10:00", "HIGHLANDS COFFEE",
                    "29 Lê Duẩn, P. Bến Nghé, Quận 1, TP.HCM",
                    "10/05/2026 10:30:15", "104.000", 104000.0, 2,
                    json.dumps(highland_items, ensure_ascii=False), 1, 0.0, "qwen3_lora_v2",
                    "/templates_images/highland_template.jpg", "Hóa đơn F&B chuỗi cà phê", "usr_highlands"
                ))
            conn.commit()
            print("[Database] Supabase PostgreSQL initialized successfully!")
        else:
            # SQLite setup
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    full_name TEXT,
                    organization TEXT,
                    role TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_receipts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    seller TEXT,
                    address TEXT,
                    receipt_time TEXT,
                    total_cost TEXT,
                    total_amount REAL,
                    item_count INTEGER,
                    items_json TEXT,
                    is_valid INTEGER,
                    discrepancy REAL,
                    model_id TEXT,
                    image_url TEXT,
                    notes TEXT,
                    user_id TEXT
                );
            """)
            cur.execute("INSERT OR IGNORE INTO users (id, email, password_hash, full_name, organization, role) VALUES ('usr_winmart', 'ketoan@winmart.vn', '123456', 'Kế toán WinMart', 'WinMart', 'client')")
            cur.execute("INSERT OR IGNORE INTO users (id, email, password_hash, full_name, organization, role) VALUES ('usr_highlands', 'thungan@highlands.vn', '123456', 'Thu ngân Highlands', 'Highlands', 'client')")
            conn.commit()
            print("[Database] SQLite initialized successfully!")
    finally:
        cur.close()
        conn.close()

def db_get_user(email: str, password: str = None):
    conn, engine = get_connection()
    try:
        if engine == "postgres":
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if password:
                cur.execute("SELECT * FROM users WHERE email = %s AND password_hash = %s", (email, password))
            else:
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
            return dict(row) if row else None
        else:
            cur = conn.cursor()
            if password:
                cur.execute("SELECT * FROM users WHERE email = ? AND password_hash = ?", (email, password))
            else:
                cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()

def db_create_user(user_id: str, email: str, password: str, full_name: str, organization: str, role: str = "client"):
    conn, engine = get_connection()
    cur = conn.cursor()
    try:
        if engine == "postgres":
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, organization, role) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, email, password, full_name, organization, role)
            )
        else:
            cur.execute(
                "INSERT INTO users (id, email, password_hash, full_name, organization, role) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, email, password, full_name, organization, role)
            )
        conn.commit()
        return True
    finally:
        cur.close()
        conn.close()

def db_save_receipt(rec: dict):
    conn, engine = get_connection()
    cur = conn.cursor()
    try:
        if engine == "postgres":
            cur.execute("""
                INSERT INTO saved_receipts (
                    id, created_at, seller, address, receipt_time, total_cost, total_amount, item_count, items_json, is_valid, discrepancy, model_id, image_url, notes, user_id
                ) VALUES (
                    %(id)s, %(created_at)s, %(seller)s, %(address)s, %(receipt_time)s, %(total_cost)s, %(total_amount)s, %(item_count)s, %(items_json)s, %(is_valid)s, %(discrepancy)s, %(model_id)s, %(image_url)s, %(notes)s, %(user_id)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    seller = EXCLUDED.seller,
                    address = EXCLUDED.address,
                    receipt_time = EXCLUDED.receipt_time,
                    total_cost = EXCLUDED.total_cost,
                    total_amount = EXCLUDED.total_amount,
                    item_count = EXCLUDED.item_count,
                    items_json = EXCLUDED.items_json,
                    is_valid = EXCLUDED.is_valid,
                    discrepancy = EXCLUDED.discrepancy,
                    model_id = EXCLUDED.model_id,
                    image_url = EXCLUDED.image_url,
                    notes = EXCLUDED.notes,
                    user_id = EXCLUDED.user_id;
            """, rec)
        else:
            cur.execute("""
                INSERT OR REPLACE INTO saved_receipts (
                    id, created_at, seller, address, receipt_time, total_cost, total_amount, item_count, items_json, is_valid, discrepancy, model_id, image_url, notes, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec["id"], rec["created_at"], rec["seller"], rec["address"], rec["receipt_time"],
                rec["total_cost"], rec["total_amount"], rec["item_count"], rec["items_json"],
                rec["is_valid"], rec["discrepancy"], rec["model_id"], rec["image_url"],
                rec["notes"], rec["user_id"]
            ))
        conn.commit()
        return True
    finally:
        cur.close()
        conn.close()

def db_get_receipts(user_id: str):
    if not user_id:
        return []
    conn, engine = get_connection()
    try:
        if engine == "postgres":
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM saved_receipts WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM saved_receipts WHERE user_id = ? ORDER BY rowid DESC", (user_id,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()

def db_delete_receipt(receipt_id: str, user_id: str = None):
    conn, engine = get_connection()
    cur = conn.cursor()
    try:
        if engine == "postgres":
            if user_id:
                cur.execute("DELETE FROM saved_receipts WHERE id = %s AND user_id = %s", (receipt_id, user_id))
            else:
                cur.execute("DELETE FROM saved_receipts WHERE id = %s", (receipt_id,))
        else:
            if user_id:
                cur.execute("DELETE FROM saved_receipts WHERE id = ? AND user_id = ?", (receipt_id, user_id))
            else:
                cur.execute("DELETE FROM saved_receipts WHERE id = ?", (receipt_id,))
        conn.commit()
        return True
    finally:
        cur.close()
        conn.close()
