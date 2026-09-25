import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))
from backend.docaligner import DocAligner
from backend.document_processor import four_point_transform, enhance_illumination

da = DocAligner()

def create_desk_composition(clean_img_path, desk_type, angle, out_w, out_h, scale_factor=0.74):
    clean_img = cv2.imread(clean_img_path)
    if clean_img is None:
        raise ValueError(f"Cannot read {clean_img_path}")
    rh, rw = clean_img.shape[:2]
    
    # 1. Generate desk texture
    if desk_type == "dark_oak":
        desk = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y = np.linspace(0, 1, out_h)[:, None]
        x = np.linspace(0, 1, out_w)[None, :]
        grain = np.random.normal(0, 3, (out_h, 1)).astype(np.float32)
        grain = cv2.resize(grain, (out_w, out_h))
        r = (48 + 12*y - 6*x + grain).clip(20, 80).astype(np.uint8)
        g = (35 + 8*y - 4*x + grain*0.7).clip(15, 65).astype(np.uint8)
        b = (27 + 6*y - 3*x + grain*0.5).clip(10, 55).astype(np.uint8)
        desk = cv2.merge([b, g, r])
    elif desk_type == "cafe_wood":
        desk = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y = np.linspace(0, 1, out_h)[:, None]
        x = np.linspace(0, 1, out_w)[None, :]
        grain = np.random.normal(0, 4, (out_h, 1)).astype(np.float32)
        grain = cv2.resize(grain, (out_w, out_h))
        r = (68 + 15*y - 8*x + grain).clip(30, 110).astype(np.uint8)
        g = (50 + 10*y - 5*x + grain*0.8).clip(20, 85).astype(np.uint8)
        b = (36 + 8*y - 4*x + grain*0.6).clip(15, 65).astype(np.uint8)
        desk = cv2.merge([b, g, r])
    elif desk_type == "slate_counter":
        desk = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        noise = np.random.normal(0, 4, (out_h, out_w)).astype(np.float32)
        y = np.linspace(0, 1, out_h)[:, None]
        desk[:, :, 0] = (52 + 10*y + noise).clip(35, 80).astype(np.uint8)
        desk[:, :, 1] = (56 + 10*y + noise).clip(40, 85).astype(np.uint8)
        desk[:, :, 2] = (60 + 10*y + noise).clip(45, 90).astype(np.uint8)
    else:  # office_desk
        desk = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        noise = np.random.normal(0, 3, (out_h, out_w)).astype(np.float32)
        y = np.linspace(0, 1, out_h)[:, None]
        desk[:, :, 0] = (75 + 15*y + noise).clip(50, 115).astype(np.uint8)
        desk[:, :, 1] = (80 + 15*y + noise).clip(55, 120).astype(np.uint8)
        desk[:, :, 2] = (85 + 15*y + noise).clip(60, 125).astype(np.uint8)

    # Vignette lighting
    rows, cols = out_h, out_w
    kernel_x = cv2.getGaussianKernel(cols, cols*0.85)
    kernel_y = cv2.getGaussianKernel(rows, rows*0.85)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = (mask / mask.max()).clip(0.68, 1.0)[:, :, None]
    desk = (desk * vignette).astype(np.uint8)

    # 2. Scale receipt so it sits inside desk with ~14% margin
    target_w = int(out_w * scale_factor)
    scale = target_w / rw
    scaled_w = target_w
    scaled_h = int(rh * scale)
    scaled = cv2.resize(clean_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # 3. Create full-canvas layers for transformation
    paper_layer = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    alpha_layer = np.zeros((out_h, out_w), dtype=np.uint8)

    cx, cy = out_w // 2, out_h // 2
    x0 = cx - scaled_w // 2
    y0 = cy - scaled_h // 2
    x1 = x0 + scaled_w
    y1 = y0 + scaled_h

    paper_layer[y0:y1, x0:x1] = scaled
    alpha_layer[y0:y1, x0:x1] = 255

    # Rotate around canvas center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot_paper = cv2.warpAffine(paper_layer, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=[0, 0, 0])
    rot_alpha = cv2.warpAffine(alpha_layer, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)

    # Soft ambient drop shadow under paper
    shadow_mask = cv2.GaussianBlur(rot_alpha, (35, 35), 14)
    # Offset shadow slightly down-right
    M_shadow = np.float32([[1, 0, 6], [0, 1, 8]])
    shadow_shifted = cv2.warpAffine(shadow_mask, M_shadow, (out_w, out_h), borderValue=0)
    s_val = (shadow_shifted.astype(np.float32) / 255.0 * 0.58)[:, :, None]

    # Darken desk by shadow
    desk = (desk.astype(np.float32) * (1.0 - s_val)).astype(np.uint8)

    # Composite paper onto desk
    mask_3d = (rot_alpha.astype(np.float32) / 255.0)[:, :, None]
    desk = (rot_paper * mask_3d + desk * (1.0 - mask_3d)).astype(np.uint8)

    # 4. Detect with DocAligner ONNX
    pts = da(desk)
    warped = four_point_transform(desk, pts)
    enhanced = enhance_illumination(warped)

    corners = [
        {"label": "P1 (Top-Left)", "x": round(float(pts[0][0]/out_w*100), 1), "y": round(float(pts[0][1]/out_h*100), 1)},
        {"label": "P2 (Top-Right)", "x": round(float(pts[1][0]/out_w*100), 1), "y": round(float(pts[1][1]/out_h*100), 1)},
        {"label": "P3 (Bottom-Right)", "x": round(float(pts[2][0]/out_w*100), 1), "y": round(float(pts[2][0]/out_h*100), 1)},
        {"label": "P4 (Bottom-Left)", "x": round(float(pts[3][0]/out_w*100), 1), "y": round(float(pts[3][0]/out_h*100), 1)},
    ]
    return desk, enhanced, corners

samples = [
    {
        "id": "sample_winmart",
        "clean": "frontend/public/templates_images/preprocessed/winmart_template.jpg",
        "desk": "dark_oak",
        "angle": -3.8,
        "w": 560,
        "h": 820,
        "scale": 0.72,
        "out_name": "winmart_template.jpg"
    },
    {
        "id": "sample_highland",
        "clean": "frontend/public/templates_images/preprocessed/highland_template.jpg",
        "desk": "cafe_wood",
        "angle": 3.2,
        "w": 520,
        "h": 860,
        "scale": 0.70,
        "out_name": "highland_template.jpg"
    },
    {
        "id": "sample_circlek",
        "clean": "frontend/public/templates_images/preprocessed/circle_k_template.webp",
        "desk": "slate_counter",
        "angle": -4.0,
        "w": 520,
        "h": 860,
        "scale": 0.70,
        "out_name": "circle_k_template.webp"
    },
    {
        "id": "sample_viettel",
        "clean": "frontend/public/templates_images/preprocessed/viettel_template.jpg",
        "desk": "office_desk",
        "angle": -2.8,
        "w": 840,
        "h": 1180,
        "scale": 0.72,
        "out_name": "viettel_template.jpg"
    }
]

# Run generation
results = {}
for s in samples:
    raw_img, prep_img, corners = create_desk_composition(s["clean"], s["desk"], s["angle"], s["w"], s["h"], s["scale"])
    results[s["id"]] = corners
    print(f"=== {s['id']} ===")
    print("Corners:", corners)
    print(f"Raw: {raw_img.shape[1]}x{raw_img.shape[0]}, Prep: {prep_img.shape[1]}x{prep_img.shape[0]}")
    
    # Sync to all relevant directories
    dest_dirs = [
        "frontend/public/templates_images",
        "src/frontend/public/templates_images",
        "backend/templates_images",
        "src/backend/templates_images",
    ]
    for d in dest_dirs:
        os.makedirs(os.path.join(d, "raw"), exist_ok=True)
        os.makedirs(os.path.join(d, "preprocessed"), exist_ok=True)
        cv2.imwrite(os.path.join(d, "raw", s["out_name"]), raw_img)
        cv2.imwrite(os.path.join(d, "preprocessed", s["out_name"]), prep_img)
        cv2.imwrite(os.path.join(d, s["out_name"]), prep_img)

# For Phuc Long, detect from existing raw image
pl_raw = cv2.imread("frontend/public/templates_images/raw/phuc_long_template.jpg")
if pl_raw is not None:
    pts_pl = da(pl_raw)
    h_pl, w_pl = pl_raw.shape[:2]
    results["sample_phuclong"] = [
        {"label": "P1 (Top-Left)", "x": round(float(pts_pl[0][0]/w_pl*100), 1), "y": round(float(pts_pl[0][1]/h_pl*100), 1)},
        {"label": "P2 (Top-Right)", "x": round(float(pts_pl[1][0]/w_pl*100), 1), "y": round(float(pts_pl[1][1]/h_pl*100), 1)},
        {"label": "P3 (Bottom-Right)", "x": round(float(pts_pl[2][0]/w_pl*100), 1), "y": round(float(pts_pl[2][0]/h_pl*100), 1)},
        {"label": "P4 (Bottom-Left)", "x": round(float(pts_pl[3][0]/w_pl*100), 1), "y": round(float(pts_pl[3][1]/h_pl*100), 1)},
    ]
    print("=== sample_phuclong ===")
    print("Corners:", results["sample_phuclong"])

print("\n\nFINAL_RESULTS_JSON:")
print(json.dumps(results, indent=2))
