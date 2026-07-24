"""
Stage 1: Camera preprocessing pipeline
iPhone → L(x, y, t) in LMS cone space

Requirements:
    pip install opencv-python numpy matplotlib

Usage:
    python pipeline_stage1.py

Controls:
    q  — quit
    s  — validate current frame (plots LMS channels)
    1  — try camera index 1 (iPhone)
    0  — try camera index 0 (built-in Mac camera)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA_INDEX  = 1        # iPhone via Continuity Camera (try 0 if this fails)
GRID_SIZE     = (32, 32) # Electrode grid — change to match your target array
BUFFER_FRAMES = 10       # Temporal history: 10 frames ≈ 333ms at 30fps

# ── Colour transforms ──────────────────────────────────────────────────────────

def linearise_gamma(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Camera pixels are gamma-encoded (sRGB). Undo that so pixel values
    are proportional to actual light intensity. Output: float32 in [0, 1].
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    v = frame_rgb.astype(np.float32) / 255.0
    linear = np.where(
        v < 0.04045,
        v / 12.92,
        ((v + 0.055) / 1.055) ** 2.4
    )
    return linear.astype(np.float32)


# sRGB (linear, D65) → LMS cone responses
# Source: Hunt & Pointer (2011) "Measuring Colour", 4th ed.
M_RGB_TO_LMS = np.array([
    [0.3811, 0.5783, 0.0402],   # L cone  — long wavelength  (~red)
    [0.1967, 0.7244, 0.0782],   # M cone  — medium wavelength (~green)
    [0.0241, 0.1288, 0.8444],   # S cone  — short wavelength  (~blue)
], dtype=np.float32)


def rgb_to_lms(frame_linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB → LMS cone responses via matrix multiplication.
    Output shape: same (H, W, 3) but channels are now L, M, S cones.
    """
    h, w, _ = frame_linear.shape
    pixels   = frame_linear.reshape(-1, 3)       # (H*W, 3)
    lms      = pixels @ M_RGB_TO_LMS.T           # (H*W, 3)
    return lms.reshape(h, w, 3)                  # (H, W, 3)


def downsample(frame_lms: np.ndarray) -> np.ndarray:
    """
    Resize to electrode grid. INTER_AREA averages pixels — correct for
    downsampling (avoids aliasing). Each output pixel = one RGC patch.
    """
    return cv2.resize(
        frame_lms,
        GRID_SIZE,
        interpolation=cv2.INTER_AREA
    )


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_frame(grid: np.ndarray, buffer: np.ndarray) -> None:
    """
    Plot all three LMS channels side by side and print diagnostics.
    What to look for:
      - L and M channels look similar (both respond to luminance)
      - S channel lights up mainly for blue/purple objects
      - All values should be in [0, 1]
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    titles = ['L cone (long/~red)', 'M cone (mid/~green)', 'S cone (short/~blue)', 'L+M (luminance)']
    data   = [grid[:,:,0], grid[:,:,1], grid[:,:,2], (grid[:,:,0] + grid[:,:,1]) / 2]
    cmaps  = ['Reds', 'Greens', 'Blues', 'gray']

    for ax, d, t, c in zip(axes, data, titles, cmaps):
        im = ax.imshow(d, cmap=c, vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Stage 1 output — {GRID_SIZE[0]}×{GRID_SIZE[1]} grid', fontsize=11)
    plt.tight_layout()

    print("\n── Stage 1 diagnostics ──────────────────────")
    print(f"  Buffer shape  : {buffer.shape}  (rows, cols, LMS, frames)")
    print(f"  Value range   : [{buffer.min():.4f}, {buffer.max():.4f}]  (should be 0–1)")
    print(f"  L channel mean: {grid[:,:,0].mean():.4f}")
    print(f"  M channel mean: {grid[:,:,1].mean():.4f}")
    print(f"  S channel mean: {grid[:,:,2].mean():.4f}")
    print("─────────────────────────────────────────────\n")

    plt.show()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(camera_index: int = CAMERA_INDEX) -> np.ndarray:
    """
    Opens the camera, runs the preprocessing pipeline, and returns the
    final temporal buffer L(x, y, t) ready for Stage 2 (RGC encoding).

    Returns
    -------
    L : np.ndarray, shape (GRID_SIZE[1], GRID_SIZE[0], 3, BUFFER_FRAMES)
        Axes: (spatial_y, spatial_x, LMS_channel, time)
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}.")
        print("Try pressing '1' or '0' while running, or edit CAMERA_INDEX above.")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\nCamera {camera_index} connected — {w}×{h} @ {fps:.0f}fps")
    print("Controls:  q=quit   s=validate   0/1=switch camera\n")

    # L(x, y, t): rolling temporal buffer in LMS space
    rows, cols = GRID_SIZE[1], GRID_SIZE[0]
    L = np.zeros((rows, cols, 3, BUFFER_FRAMES), dtype=np.float32)

    latest_grid = np.zeros((rows, cols, 3), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Dropped frame.")
            continue

        # ── The three transforms ──────────────────────────────────────────────
        linear = linearise_gamma(frame)        # uint8 BGR → float32 RGB linear
        lms    = rgb_to_lms(linear)            # RGB → LMS cone responses
        grid   = np.clip(downsample(lms), 0, 1) # downsample + clamp to [0,1]
        # ─────────────────────────────────────────────────────────────────────

        # Update rolling buffer (drop oldest frame, add newest)
        L = np.roll(L, -1, axis=3)
        L[:, :, :, -1] = grid
        latest_grid = grid

        # Live preview: show L cone channel at 10× size
        preview_channel = L[:, :, 0, -1]   # L cone = rough luminance
        preview = cv2.resize(
            (preview_channel * 255).astype(np.uint8),
            (320, 320),
            interpolation=cv2.INTER_NEAREST   # pixelated = shows grid clearly
        )
        cv2.putText(preview, f'{GRID_SIZE[0]}x{GRID_SIZE[1]} grid',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.imshow('Stage 1 — L cone channel (press s to validate, q to quit)', preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            validate_frame(latest_grid, L)
        elif key == ord('0'):
            cap.release()
            return run(0)
        elif key == ord('1'):
            cap.release()
            return run(1)

    cap.release()
    cv2.destroyAllWindows()
    print("\nPipeline stopped.")
    print(f"L buffer ready for Stage 2. Shape: {L.shape}")
    print("Pass this array to your RGC encoding (Step 2).\n")
    return L   # hand off to Stage 2


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    run(idx)