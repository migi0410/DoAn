import re
from difflib import SequenceMatcher

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_number(text):
    if text is None:
        return ""
    digits = re.sub(r'[^\d]', '', str(text))
    return digits

def string_similarity(a, b):
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def eval_single_entity(pred_val, gt_val, is_numeric=False, sim_threshold=0.70):
    has_gt = bool(gt_val and str(gt_val).strip())
    has_pred = bool(pred_val and str(pred_val).strip())
    
    if not has_gt and not has_pred:
        return 1, 0, 0, 1.0, 1.0, 1.0, 1, 1.0
    if has_gt and not has_pred:
        return 0, 0, 1, 0.0, 0.0, 0.0, 0, 0.0
    if not has_gt and has_pred:
        return 0, 1, 0, 0.0, 0.0, 0.0, 0, 0.0
        
    if is_numeric:
        num_gt = normalize_number(gt_val)
        num_pred = normalize_number(pred_val)
        exact = int(num_gt == num_pred and len(num_gt) > 0)
        sim = 1.0 if exact else 0.0
        is_match = bool(exact)
    else:
        norm_gt = normalize_text(gt_val)
        norm_pred = normalize_text(pred_val)
        exact = int(norm_gt == norm_pred)
        sim = string_similarity(gt_val, pred_val)
        is_match = (sim >= sim_threshold)
        
    if is_match:
        return 1, 0, 0, 1.0, 1.0, 1.0, exact, sim
    else:
        return 0, 1, 1, 0.0, 0.0, 0.0, exact, sim

def evaluate_all_8_fields(pred_json, gt_json):
    """
    Đánh giá chi tiết TOÀN BỘ các chỉ số KIE:
    - Similarity: Seller_Sim, Total_Sim
    - Items: Item_Recall, Item_Precision, Item_F1
    - Chi tiết 8 trường: SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, ITEM_NAME, ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT
    - Tổng hợp: Macro-F1, Micro-F1, Exact Match (EM)
    """
    fields = [
        "SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST",
        "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE", "ITEM_AMOUNT"
    ]
    
    if not pred_json or not isinstance(pred_json, dict):
        res = {
            "valid_json": False,
            "Seller_Sim": 0.0,
            "Total_Sim": 0.0,
            "Item_Recall": 0.0,
            "Item_Precision": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "exact_match": 0.0
        }
        for f in fields:
            res[f"{f}_P"] = 0.0
            res[f"{f}_R"] = 0.0
            res[f"{f}_F1"] = 0.0
        return res

    results = {"valid_json": True}
    total_tp, total_fp, total_fn = 0, 0, 0
    all_f1s = []
    exact_matches = []

    # 1. 4 Header Fields
    header_configs = [
        ("SELLER", False),
        ("ADDRESS", False),
        ("TIMESTAMP", False),
        ("TOTAL_COST", True)
    ]
    for fname, is_num in header_configs:
        tp, fp, fn, p, r, f1, em, sim = eval_single_entity(
            pred_json.get(fname, ""),
            gt_json.get(fname, ""),
            is_numeric=is_num
        )
        results[f"{fname}_P"] = round(p * 100, 1)
        results[f"{fname}_R"] = round(r * 100, 1)
        results[f"{fname}_F1"] = round(f1 * 100, 1)
        if fname == "SELLER":
            results["Seller_Sim"] = round(sim * 100, 1)
        elif fname == "TOTAL_COST":
            results["Total_Sim"] = round(sim * 100, 1)
            
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_f1s.append(f1)
        exact_matches.append(em)

    # 2. 4 Item Array Fields
    gt_names = gt_json.get("ITEM_NAME", [])
    if not isinstance(gt_names, list): gt_names = [gt_names] if gt_names else []
    gt_qtys = gt_json.get("ITEM_QTY", [])
    if not isinstance(gt_qtys, list): gt_qtys = [gt_qtys] if gt_qtys else []
    gt_prices = gt_json.get("ITEM_PRICE", [])
    if not isinstance(gt_prices, list): gt_prices = [gt_prices] if gt_prices else []
    gt_amounts = gt_json.get("ITEM_AMOUNT", [])
    if not isinstance(gt_amounts, list): gt_amounts = [gt_amounts] if gt_amounts else []

    pred_names = pred_json.get("ITEM_NAME", [])
    if not isinstance(pred_names, list): pred_names = [pred_names] if pred_names else []
    pred_qtys = pred_json.get("ITEM_QTY", [])
    if not isinstance(pred_qtys, list): pred_qtys = [pred_qtys] if pred_qtys else []
    pred_prices = pred_json.get("ITEM_PRICE", [])
    if not isinstance(pred_prices, list): pred_prices = [pred_prices] if pred_prices else []
    pred_amounts = pred_json.get("ITEM_AMOUNT", [])
    if not isinstance(pred_amounts, list): pred_amounts = [pred_amounts] if pred_amounts else []

    # Bipartite matching based on ITEM_NAME
    matched_pairs = []
    matched_gt_indices = set()
    for p_idx, p_name in enumerate(pred_names):
        best_sim = 0.0
        best_gt_idx = -1
        for g_idx, g_name in enumerate(gt_names):
            if g_idx in matched_gt_indices:
                continue
            sim = string_similarity(p_name, g_name)
            if sim > best_sim:
                best_sim = sim
                best_gt_idx = g_idx
        if best_sim >= 0.65 and best_gt_idx != -1:
            matched_pairs.append((p_idx, best_gt_idx))
            matched_gt_indices.add(best_gt_idx)

    # ITEM_NAME & Global Item Recall/Precision
    tp_name = len(matched_pairs)
    fp_name = len(pred_names) - tp_name
    fn_name = len(gt_names) - tp_name
    p_name = tp_name / (tp_name + fp_name) if (tp_name + fp_name) > 0 else (1.0 if not gt_names else 0.0)
    r_name = tp_name / (tp_name + fn_name) if (tp_name + fn_name) > 0 else (1.0 if not pred_names else 0.0)
    f1_name = (2 * p_name * r_name) / (p_name + r_name) if (p_name + r_name) > 0 else 0.0
    
    results["ITEM_NAME_P"] = round(p_name * 100, 1)
    results["ITEM_NAME_R"] = round(r_name * 100, 1)
    results["ITEM_NAME_F1"] = round(f1_name * 100, 1)
    results["Item_Recall"] = round(r_name * 100, 1)
    results["Item_Precision"] = round(p_name * 100, 1)
    
    total_tp += tp_name; total_fp += fp_name; total_fn += fn_name
    all_f1s.append(f1_name)

    # ITEM_QTY, ITEM_PRICE, ITEM_AMOUNT
    for col_key, pred_col, gt_col in [
        ("ITEM_QTY", pred_qtys, gt_qtys),
        ("ITEM_PRICE", pred_prices, gt_prices),
        ("ITEM_AMOUNT", pred_amounts, gt_amounts)
    ]:
        tp_col = 0
        for p_idx, g_idx in matched_pairs:
            p_val = normalize_number(pred_col[p_idx]) if p_idx < len(pred_col) else ""
            g_val = normalize_number(gt_col[g_idx]) if g_idx < len(gt_col) else ""
            if p_val and g_val and p_val == g_val:
                tp_col += 1
                
        fp_col = len(pred_col) - tp_col
        fn_col = len(gt_col) - tp_col
        p_col = tp_col / (tp_col + fp_col) if (tp_col + fp_col) > 0 else (1.0 if not gt_col else 0.0)
        r_col = tp_col / (tp_col + fn_col) if (tp_col + fn_col) > 0 else (1.0 if not pred_col else 0.0)
        f1_col = (2 * p_col * r_col) / (p_col + r_col) if (p_col + r_col) > 0 else 0.0
        results[f"{col_key}_P"] = round(p_col * 100, 1)
        results[f"{col_key}_R"] = round(r_col * 100, 1)
        results[f"{col_key}_F1"] = round(f1_col * 100, 1)
        total_tp += tp_col; total_fp += fp_col; total_fn += fn_col
        all_f1s.append(f1_col)

    # 3. Macro & Micro F1
    results["macro_f1"] = round((sum(all_f1s) / len(all_f1s)) * 100, 1) if all_f1s else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    results["micro_f1"] = round(((2 * micro_p * micro_r) / (micro_p + micro_r)) * 100, 1) if (micro_p + micro_r) > 0 else 0.0
    results["exact_match"] = round((sum(exact_matches) / len(exact_matches)) * 100, 1) if exact_matches else 0.0

    return results
