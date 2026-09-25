import re
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def clean_currency(val_str: str) -> float:
    if not val_str:
        return 0.0
    cleaned = re.sub(r"[^\d]", "", str(val_str))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def is_summary_line(name: str) -> bool:
    n = name.strip().lower()
    if not n:
        return True
    patterns = [
        r'^(?:tổng|tong)\s*[:\.]?\s*\d*\s*(?:mặt hàng|mat hang|món|mon|sp|sản phẩm|san pham)?',
        r'^(?:tổng\s*cộng|tong\s*cong)\b',
        r'^(?:tổng\s*tiền|tong\s*tien)\b',
        r'^(?:số\s*tiền\s*thanh\s*toán|so\s*tien\s*thanh\s*toan)\b',
        r'^(?:thanh\s*toán|thanh\s*toan)\b',
        r'^(?:tiền\s*mặt|tien\s*mat)\b',
        r'^(?:tiền\s*thừa|tiền\s*trả\s*lại)\b',
        r'^(?:nhận\s*của\s*khách|khách\s*đưa)\b',
        r'^(?:t\s*tiền|thành\s*tiền)\b',
        r'^(?:điểm\s*tích\s*lũy|tổng\s*tích\s*lũy)\b',
    ]
    for p in patterns:
        if re.search(p, n):
            return True
    return False

def is_brand_or_item_start(name: str) -> bool:
    name_clean = name.strip()
    if not name_clean:
        return False
    words = name_clean.split()
    first_word = words[0]
    first_two = " ".join(words[:2])
    if first_word.isupper() and len(first_word) >= 2:
        return True
    if first_two.isupper():
        return True
    category_prefixes = [
        "sữa", "bánh", "nước", "trà", "cà phê", "cafe", "thịt", "cá", "gà", "rau", 
        "trứng", "dầu", "gạo", "mì", "khăn", "kem", "xà phòng", "bột", "combo"
    ]
    if any(name_clean.lower().startswith(p) for p in category_prefixes):
        return True
    return False

def has_item_unit_end(name: str) -> bool:
    pattern = r'(?:\d+[\.,]?\d*\s*(?:g|kg|ml|l|gr|lon|chai|hộp|hop|gói|goi|cái|cai|l1|l2))\b'
    return bool(re.search(pattern, name.strip(), re.I))

def reconcile_receipt_items(items, total_cost_str=""):
    """
    Robust production reconciliation engine for Vietnamese retail receipts:
    1. Filters summary rows (Tổng tiền, Tiền thừa...).
    2. Handles multi-line continuation items (item dài rớt dòng).
    3. Handles decoupled trailing desynchronized lines.
    4. Synchronizes item quantities, unit prices, and amounts.
    """
    if not items:
        return items

    # Filter out empty names or summary lines
    valid_candidates = []
    for it in items:
        name = str(it.get("name") or it.get("ITEM_NAME") or "").strip()
        if not name or is_summary_line(name):
            continue
        valid_candidates.append({
            "name": name,
            "qty": str(it.get("qty") or it.get("ITEM_QTY") or "1").strip() or "1",
            "price": str(it.get("price") or it.get("ITEM_PRICE") or "").strip(),
            "amount": str(it.get("amount") or it.get("ITEM_AMOUNT") or "").strip(),
        })

    if len(valid_candidates) <= 1:
        return valid_candidates

    # Check if any items have empty amounts
    has_empty_amount = any(not clean_currency(it.get("amount", "")) for it in valid_candidates)
    if not has_empty_amount:
        return valid_candidates

    items_with_amount = [it for it in valid_candidates if clean_currency(it.get("amount", ""))]
    num_valid = len(items_with_amount)
    if num_valid == 0:
        return valid_candidates

    # Check if it's trailing empty desynchronization pattern:
    first_empty_idx = next(i for i, it in enumerate(valid_candidates) if not clean_currency(it.get("amount", "")))
    is_trailing_empty = all(not clean_currency(valid_candidates[k].get("amount", "")) for k in range(first_empty_idx, len(valid_candidates)))

    if is_trailing_empty and first_empty_idx == num_valid and num_valid > 1:
        valid_prices = [
            {
                "qty": it.get("qty", "1") or "1",
                "price": it.get("price", "") or it.get("amount", ""),
                "amount": it.get("amount", "")
            }
            for it in items_with_amount
        ]
        product_blocks = []
        curr_block = []
        for i, it in enumerate(valid_candidates):
            raw_name = it["name"]
            if not curr_block:
                curr_block.append(raw_name)
            else:
                prev_text = curr_block[-1]
                is_new = (has_item_unit_end(prev_text) or is_brand_or_item_start(raw_name))
                remaining_blocks_needed = num_valid - len(product_blocks)
                remaining_items = len(valid_candidates) - i
                if (is_new and len(product_blocks) < num_valid - 1) or (remaining_items == remaining_blocks_needed):
                    product_blocks.append(" ".join(curr_block))
                    curr_block = [raw_name]
                else:
                    curr_block.append(raw_name)
        if curr_block:
            product_blocks.append(" ".join(curr_block))

        if len(product_blocks) == num_valid:
            reconciled = []
            for blk, pinfo in zip(product_blocks, valid_prices):
                clean_name = re.sub(r'\s+', ' ', blk).strip()
                reconciled.append({
                    "name": clean_name,
                    "qty": pinfo["qty"],
                    "price": pinfo["price"],
                    "amount": pinfo["amount"]
                })
            return reconciled

    # Interleaved / standard multi-line merge:
    # 1. Forward merge: any line without amount merges into preceding item with amount
    # 2. Reverse merge: if initial line has no amount, merge into subsequent item with amount
    reconciled = []
    for it in valid_candidates:
        amt = clean_currency(it.get("amount", ""))
        name = it["name"]
        if amt > 0:
            reconciled.append(dict(it))
        else:
            if reconciled:
                reconciled[-1]["name"] = re.sub(r'\s+', ' ', f"{reconciled[-1]['name']} {name}").strip()
            else:
                reconciled.append(dict(it))

    # If first item had no amount and second has amount:
    if len(reconciled) >= 2 and not clean_currency(reconciled[0].get("amount", "")) and clean_currency(reconciled[1].get("amount", "")):
        reconciled[1]["name"] = re.sub(r'\s+', ' ', f"{reconciled[0]['name']} {reconciled[1]['name']}").strip()
        reconciled.pop(0)

    # Any remaining item without amount that could not be merged:
    # If total_cost is declared, check if last item can be solved:
    declared_total = clean_currency(total_cost_str)
    if declared_total > 0:
        current_sum = sum(clean_currency(it.get("amount", "")) for it in reconciled)
        diff = declared_total - current_sum
        for it in reconciled:
            if not clean_currency(it.get("amount", "")) and diff > 0:
                it["amount"] = f"{int(diff):,}".replace(",", ".")
                if not it.get("price"):
                    it["price"] = it["amount"]
                break

    return reconciled

# Run tests
t1 = [
    {"name": "Sữa Tươi Tiệt Trùng Vinamilk", "qty": "", "price": "", "amount": ""},
    {"name": "100% Có Đường 1L", "qty": "2", "price": "36.000", "amount": "72.000"},
    {"name": "Bánh Mì Sandwich Kinh Đô", "qty": "1", "price": "22.000", "amount": "22.000"},
    {"name": "Lúa Mạch 250g", "qty": "", "price": "", "amount": ""},
]
print("Test 1 Result:", json.dumps(reconcile_receipt_items(t1), indent=2, ensure_ascii=False))

t2 = [
    {"name": "Trà Sen Vàng (L)", "qty": "1", "price": "", "amount": "65.000"},
    {"name": "(Thêm sen, 50% đường)", "qty": "", "price": "", "amount": ""},
    {"name": "Bánh Mì Thịt Nướng", "qty": "1", "price": "", "amount": "39.000"}
]
print("Test 2 Result:", json.dumps(reconcile_receipt_items(t2), indent=2, ensure_ascii=False))
