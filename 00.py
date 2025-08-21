import pydicom, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_propagation
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift
import numpy as np
import cv2
import os

# --------- Alignment (symmetry & top-unify) settings ----------
enable_align      = True   # turn on/off auto rotation + vertical alignment
align_top_row     = 60     # after alignment, the top-most skull pixel will be placed at this row
align_fill_hu     = 0.0    # fill value for HU during rotation/shift (water)
align_fill_mask   = 0      # fill value for mask during rotation/shift


# --------- 3) 윈도우(기본: Bone-ish) ----------
def window(img, center, width):
    lo, hi = center - width/2.0, center + width/2.0
    imgw = np.clip(img, lo, hi)
    return (imgw - lo) / (hi - lo)  # 0..1

def enforce_upward_convex(skull_mask, hu, top_y, d1=40, d2=100):
    """Ensure the skull arc is convex upward (∩). If width increases with depth, flip vertically.
    d1,d2 are row offsets below top_y to compare widths."""
    H, W = skull_mask.shape
    r1 = min(H-1, top_y + d1)
    r2 = min(H-1, top_y + d2)
    w_top = int(skull_mask[top_y, :].sum()) if 0 <= top_y < H else 0
    w1 = int(skull_mask[r1, :].sum()) if 0 <= r1 < H else 0
    w2 = int(skull_mask[r2, :].sum()) if 0 <= r2 < H else 0
    need_flip = False
    # For convex UP: width should DECREASE as we go deeper from the top.
    # If it increases (w1 > w_top or w2 > w1), then it's upside-down -> flip.
    if (w1 > w_top) or (w2 > w1):
        need_flip = True
    # Fallback: if most mass is in the LOWER half, flip
    if not need_flip:
        ys, xs = np.nonzero(skull_mask)
        if ys.size > 0 and ys.mean() > (H/2):
            need_flip = True
    if need_flip:
        skull_mask = np.flipud(skull_mask)
        hu = np.flipud(hu)
    return skull_mask, hu, need_flip

os.makedirs("9", exist_ok=True)
out_dirs = {
    "mask": "9/mask_bin",
    "skull": "9/skull_only",
    "overlay": "9/overlay",
    "dataset": "9/dataset"
}
for d in out_dirs.values():
    os.makedirs(d, exist_ok=True)

out_dirs_dataset = {
    'x': f"{out_dirs['dataset']}/x",
    'mask': f"{out_dirs['dataset']}/mask",
    'preview': f"{out_dirs['dataset']}/preview"
}
for d in out_dirs_dataset.values():
    os.makedirs(d, exist_ok=True)

# Loop through frames 200 to 400 inclusive
for frame_idx in range(230, 350):
    # 데이터 경로
    dcm_path = f"data/OMID Skull Pilot Samples 9 June 2020_435_Mouse_09_133727/OMID Skull Pilot Samples 9 June 2020_435_Mouse_09_133727_{frame_idx:04d}.dcm"

    # --------- 1) 로드 ----------
    ds = pydicom.dcmread(dcm_path)

    wc = ds.get("WindowCenter", None)
    ww = ds.get("WindowWidth", None)

    print(f"Frame {frame_idx}: WindowCenter: {wc}, WindowWidth: {ww}")
    window_center = wc
    window_width = ww

    # 메타: 해상도
    dx, dy = (ds.PixelSpacing if "PixelSpacing" in ds else [np.nan, np.nan])  # mm/px
    thk = float(ds.get("SliceThickness", np.nan))

    print(f"Frame {frame_idx}: PixelSpacing: dx={dx} mm, dy={dy} mm, SliceThickness={thk} mm")

    # --------- 2) 픽셀 배열 ----------
    px = ds.pixel_array.astype(np.int16)           # raw detector value
    slope = float(ds.get("RescaleSlope", 1.0))     # CT면 보통 1
    interc = float(ds.get("RescaleIntercept", 0.0))# CT면 보통 -1024 등
    hu = px * slope + interc                      # Hounsfield Unit

    hu[hu < 25] = 0
    hu[350:, :] = 0  # Zero out all voxels below y=350

    seed = hu >= 700          # 확실한 피질골
    cand = hu >= 300          # 해면질 포함
    skull_mask = binary_propagation(seed, mask=cand)  # 씨드에 연결된 cand만 확장

    hu[~skull_mask] = 0       # 뼈 아닌 것 전부 물로 치환

    mask = hu >= 300
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = labels == largest_label
    else:
        largest_mask = mask
    hu[~largest_mask] = 0

    # ---- (1) 스컬 마스크 정의 (HU>0: 우리 전처리 후 스컬만 양수) ----
    skull_mask = (hu > 0)

    # ---- Auto alignment: rotate to maximize symmetry (PCA major axis -> horizontal) and unify top height ----
    if enable_align and skull_mask.any():
        # 1) Compute PCA on TOP band of the mask to avoid side dominance
        ys, xs = np.nonzero(skull_mask)
        top_y0 = ys.min()
        band_rows = 120  # ~2.4 mm at 0.02 mm/px
        band_sel = ys <= (top_y0 + band_rows)
        xs_band = xs[band_sel]
        ys_band = ys[band_sel]
        if xs_band.size < 9:
            xs_band, ys_band = xs, ys  # fallback to full mask
        cx = xs_band.mean()
        cy = ys_band.mean()
        X = np.column_stack([xs_band - cx, ys_band - cy])  # [Nb,2]
        C = (X.T @ X) / max(1, X.shape[0]-1)
        eigvals, eigvecs = np.linalg.eigh(C)
        v_major = eigvecs[:, 1]
        angle_deg = np.degrees(np.arctan2(v_major[1], v_major[0]))
        rot_deg = -angle_deg
        # 2) Apply rotation to HU and mask (keep shape, fill with water/0)
        hu = ndi_rotate(hu, rot_deg, reshape=False, order=1, mode='constant', cval=align_fill_hu)
        skull_mask = ndi_rotate(skull_mask.astype(np.uint8), rot_deg, reshape=False, order=0, mode='constant', cval=align_fill_mask).astype(bool)
        # 3) After rotation, shift vertically so that the top-most skull pixel hits align_top_row
        if skull_mask.any():
            top_y = int(np.min(np.nonzero(skull_mask)[0]))
            skull_mask, hu, flipped = enforce_upward_convex(skull_mask, hu, top_y)
            # recompute top after optional flip
            if skull_mask.any():
                top_y = int(np.min(np.nonzero(skull_mask)[0]))
                # vertical shift to align top row (clamped to keep content inside)
                ys_all, xs_all = np.nonzero(skull_mask)
                min_y, max_y = int(ys_all.min()), int(ys_all.max())
                shift_rows_desired = align_top_row - top_y
                allowed_up = min_y  # how many rows we can move up before hitting 0
                allowed_down = (skull_mask.shape[0] - 1) - max_y
                shift_rows = int(np.clip(shift_rows_desired, -allowed_up, allowed_down))
                if shift_rows != 0:
                    hu = ndi_shift(hu, shift=(shift_rows, 0), order=1, mode='constant', cval=align_fill_hu)
                    skull_mask = ndi_shift(skull_mask.astype(np.uint8), shift=(shift_rows, 0), order=0, mode='constant', cval=align_fill_mask).astype(bool)
                    # recompute top_y after shift
                    top_y = int(np.min(np.nonzero(skull_mask)[0]))
                # horizontal shift: center the apex x at image center
                H, W = skull_mask.shape
                # gather apex neighborhood rows to robustly estimate peak x
                band = 6  # rows below top
                ys_ap, xs_ap = np.nonzero(skull_mask[max(0, top_y):min(H, top_y + band), :])
                if xs_ap.size == 0:
                    # fallback: use all rows but take minimal y per column
                    ys_all, xs_all = np.nonzero(skull_mask)
                    min_y = np.minimum.reduceat(ys_all, np.unique(xs_all, return_index=True)[1]) if xs_all.size else np.array([])
                    if min_y.size:
                        xs_ap = np.unique(xs_all)
                if xs_ap.size:
                    apex_x = int(np.median(xs_ap))
                    center_x = W // 2
                    shift_cols_desired = center_x - apex_x
                    # compute current bbox
                    ys_all, xs_all = np.nonzero(skull_mask)
                    min_x, max_x = int(xs_all.min()), int(xs_all.max())
                    allowed_left = min_x                   # how many cols we can move left before hitting 0
                    allowed_right = (W - 1) - max_x        # how many cols we can move right before hitting W-1
                    shift_cols = int(np.clip(shift_cols_desired, -allowed_left, allowed_right))
                    if shift_cols != 0:
                        hu = ndi_shift(hu, shift=(0, shift_cols), order=1, mode='constant', cval=align_fill_hu)
                        skull_mask = ndi_shift(skull_mask.astype(np.uint8), shift=(0, shift_cols), order=0, mode='constant', cval=align_fill_mask).astype(bool)

    # ---- Save aligned binary mask (0/255) ----
    mask_png = f"{out_dirs['mask']}/skull_mask_{frame_idx:04d}.png"
    try:
        from imageio.v2 import imwrite as imwrite_io
        imwrite_io(mask_png, (skull_mask.astype(np.uint8) * 255))
    except Exception:
        plt.imsave(mask_png, skull_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)

    # ---- (1a) 스컬만 보이게 시각화 PNG (윈도우 사용) ----
    vis_skulled = window(hu, center=window_center, width=window_width)
    skull_png = f"{out_dirs['skull']}/skull_only_{frame_idx:04d}.png"
    plt.figure(figsize=(6,6))
    plt.imshow(vis_skulled, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.savefig(skull_png, bbox_inches="tight")
    plt.close()

    # ---- (1a-ov) 마스크 윤곽 오버레이 PNG (빨강 윤곽) ----
    # 에지: 이진 마스크에서 침식 차집합으로 경계 추출, 가시성을 위해 한 번 팽창
    edge = skull_mask & ~binary_erosion(skull_mask, structure=np.ones((3,3), dtype=bool))
    edge = binary_dilation(edge, structure=np.ones((3,3), dtype=bool), iterations=1)

    # 그레이(0..1)→RGB로 스택 후 경계 픽셀을 빨강으로 칠하기
    overlay = np.dstack([vis_skulled, vis_skulled, vis_skulled])
    overlay[edge, 0] = 1.0  # R
    overlay[edge, 1] = 0.0  # G
    overlay[edge, 2] = 0.0  # B

    overlay_png = f"{out_dirs['overlay']}/overlay_{frame_idx:04d}.png"
    plt.imsave(overlay_png, overlay, vmin=0.0, vmax=1.0)

    # ---- (2) HU 정규화 설정: hu_max = 3000 ----
    hu_max = 3500.0
    hu_clip = np.clip(hu, 0.0, hu_max)
    x01 = hu_clip / hu_max              # [0,1]
    x = x01 * 2.0 - 1.0                 # [-1,1]
    # ---- (3) 배경을 -1로 강제 ----
    x[~skull_mask] = -1.0

    # 저장: 학습 입력(.npy) + 마스크(.npy) + 미리보기 PNG
    np.save(f"{out_dirs_dataset['x']}/dataset_x_{frame_idx:04d}.npy", x.astype(np.float32))
    np.save(f"{out_dirs_dataset['mask']}/dataset_mask_{frame_idx:04d}.npy", skull_mask.astype(np.uint8))

    # 9-bit 프리뷰 ([-1,1] -> [0,255])
    preview8 = np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)
    try:
        from imageio.v2 import imwrite as imwrite_io
        imwrite_io(f"{out_dirs_dataset['preview']}/dataset_x_{frame_idx:04d}.png", preview8)
    except Exception:
        plt.imsave(f"{out_dirs_dataset['preview']}/dataset_x_{frame_idx:04d}.png", preview8, cmap="gray", vmin=0, vmax=255)

    print(slope, interc)
    print(f"Frame {frame_idx}: Raw pixel range: [{px.min()} .. {px.max()}]")
    print(f"Frame {frame_idx}: HU range:        [{hu.min()} .. {hu.max()}]")
