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
CAMERA_INDEX  = 1        # iPhone via Continuity Camera (try 0 if this fails) - number identifying which camera device your computer should connect to, when more than one is available - index 1 happens to correspond to the iPhone (index 0 being the Mac's built-in camera)
GRID_SIZE     = (32, 32) # Electrode grid — change to match your target array
BUFFER_FRAMES = 10       # Temporal history: 10 frames ≈ 333ms at 30fps - how many video frames it can hold in its memory to manipulate - for Vtemporal (need to know how pixels change over time)
                         # 10 chose to match time taken for a full neuron to fire. 100 to 300ms; for its magnitude to increase and then decrease
                         ## so it's 10 frames/30 frames per second = 0.333s = 300ms - enough time to capture a space in time where the information about the frame could be sent/responded to by a amacrine cell
                         ## so if a camera has a different fps - adjust the temporal frame history to meet the 300ms target
V_MAX         = 30.0     # peak photoreceptor voltage swing, mV — single source of truth
# ── Colour transforms ──────────────────────────────────────────────────────────

def linearise_gamma(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Camera pixels are gamma-encoded (sRGB). Undo that so pixel values
    are proportional to actual light intensity. Output: float32 in [0, 1].
    After this step you have  I (linear light intensity) received by photodiodes for RGB
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) ## reorders the channels to RGB to fit our standard, historically convention label for windows bitmap software was BGR, 
    v = frame_rgb.astype(np.float32) / 255.0 ## divides each pixel by 255 to normalise the scale - 0 to 1 - (the raw output would be 0 to 255 - 8-bit channels for each colour pixel could inhabit - harder to tell intensities 0 to 255 vs 0 to 1); astype converts integers into floats
    linear = np.where(
        v < 0.04045, 
        v / 12.92,
        ((v + 0.055) / 1.055) ** 2.4
    ) ## for condition v<0.04045, apply A otherwise apply B; B is inverse power law; A
    ## The 0.055 and 1.055 are offset/scaling constants baked into the sRGB standard's specific curve definition
    ## 2.4 is the gamma exponent; so we are undoing the ~2.4-power compression
    ## at v = 0.04045, the two curves meet - exact crossover point where the two piecewise segments of the sRGB curve meet with matching slope (continuous derivative)
    return linear.astype(np.float32) ## ensures we get 7dp output


# sRGB (linear, D65) → LMS cone responses
# Source: Hunt & Pointer (2011) "Measuring Colour", 4th ed.
M_RGB_TO_LMS = np.array([ 
    [0.3811, 0.5783, 0.0402],   # L cone  — long wavelength  (~red) = each cones sensitivity to R; G; B
    [0.1967, 0.7244, 0.0782],   # M cone  — medium wavelength (~green)
    [0.0241, 0.1288, 0.8444],   # S cone  — short wavelength  (~blue)
], dtype=np.float32)


def rgb_to_lms(frame_linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB → LMS cone responses via matrix multiplication.
    Output shape: same (H, W, 3) but channels are now L, M, S cones.
    After this step you have  I (linear light intensity) received by CONES for LMS
    """
    h, w, _ = frame_linear.shape ## Unpacks the image's dimensions. _ throws away the third value (channel count, always 3) since we don't need it here
    pixels   = frame_linear.reshape(-1, 3)       # (H*W, 3) image arrives as shape (H, W, 3) — a 2D grid of 3-value pixels - FLAT - height, width - 3 channels = RGB
    lms      = pixels @ M_RGB_TO_LMS.T           # (H*W, 3) #Transposing flips rows→columns, aligning the matrix correctly for this multiplication order
    return lms.reshape(h, w, 3)                  # (H, W, 3)

def naka_rushton(lms: np.ndarray, v_max: float = V_MAX, sigma: float = 0.5, n: float = 2.0) -> np.ndarray:
    """
    Apply Naka-Rushton Equation to factor nonlinear voltage saturation point of L,M,S cones.
    Output: VL(I), VM(I), VS(I). V(I) = Vmax * I^n / (I^n + sigma^n)
    So a light intensity increases voltage produced gets smaller
    """
    lms_n   = np.power(lms, n)          # I^n — same op applied to all 3 channels at once
    sigma_n = sigma ** n                 # sigma^n — a scalar, same for every pixel/channel
    v = v_max * lms_n / (lms_n + sigma_n + 1e-8)   # +epsilon avoids 0/0 when lms == 0
    return v.astype(np.float32)
    # n = hill coeffient = steepness of curve; high n = sharper fast part of curve; low n = smoother, curvier - close to saturation point
    # Vmax = ceiling voltage = assumed to be 30V or etc; will validate in v3 by checking Naka & Rushton's original 1966 paper, or later cone electrophysiology work
    # n = 2 - because (from research) when n=1 you get a Michael-Menten curve - but with n 2-4; you get a sigmoidal shape which is closer to what's observed
    # epsilon = 0.5; assumption that between N-R's curve of 0 to 1, the midpoint = 0.5; but this varies + is a curve not a straight line
    # a real cone's semi-saturation constant is measured from actual light-response data - validate in v3


def downsample(frame_lms: np.ndarray) -> np.ndarray:
    """
    Resize to electrode grid. INTER_AREA averages pixels — correct for
    downsampling (avoids aliasing). Each output pixel = one RGC patch.
    """
    return cv2.resize( 
        frame_lms,
        GRID_SIZE,
        interpolation=cv2.INTER_AREA
    )## SHRINKS the image to fit the electrode grid size of 32x32
    ## INTER_AREA computes a proper area-weighted average of all source pixels falling inside each destination pixel's footprin



# ── Validation ─────────────────────────────────────────────────────────────────

def validate_frame(grid: np.ndarray, buffer: np.ndarra, v_max: float = V_MAX) -> None:
    """
    Plot all three LMS channels side by side and print diagnostics.
    What to look for:
      - L and M channels look similar (both respond to luminance)
      - S channel lights up mainly for blue/purple objects
      - All values should be in [0, 1]
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5)) # 4 plots
    titles = ['L cone (long/~red)', 'M cone (mid/~green)', 'S cone (short/~blue)', 'L+M (luminance)'] # names 
    data   = [grid[:,:,0], grid[:,:,1], grid[:,:,2], (grid[:,:,0] + grid[:,:,1]) / 2] # what to populate each graph
    cmaps  = ['Reds', 'Greens', 'Blues', 'gray']

    for ax, d, t, c in zip(axes, data, titles, cmaps):
        im = ax.imshow(d, cmap=c, vmin=0, vmax=V_MAX)
        ax.set_title(t, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='mV')

    fig.suptitle(f'Stage 1 output — {GRID_SIZE[0]}×{GRID_SIZE[1]} grid (mV)', fontsize=11)
    plt.tight_layout()

    print("\n── Stage 1 diagnostics ──────────────────────")
    print(f"  Buffer shape  : {buffer.shape}  (rows, cols, LMS, frames)")
    print(f"  Value range   : [{buffer.min():.4f}, {buffer.max():.4f}] mV (should be 0–{v_max:.0f})")
    print(f"  V_L (L cone)  : mean={grid[:,:,0].mean():.3f} mV  min={grid[:,:,0].min():.3f}  max={grid[:,:,0].max():.3f}")
    print(f"  V_M (M cone)  : mean={grid[:,:,1].mean():.3f} mV  min={grid[:,:,1].min():.3f}  max={grid[:,:,1].max():.3f}")
    print(f"  V_S (S cone)  : mean={grid[:,:,2].mean():.3f} mV  min={grid[:,:,2].min():.3f}  max={grid[:,:,2].max():.3f}")
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

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the iphone's resolution - like 1920×1080 @ 30fps BEFORE downsample shrinks it to 32 x 32
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the iphone's resolution  - 1920×1080 @ 30fps BEFORE downsample shrinks it to 32 x 32
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
        v_lms  = naka_rushton(lms, v_max=V_MAX)             # Converting LMS Intensities to Voltages
        grid   = np.clip(downsample(v_lms), 0, V_MAX) # downsample - fit onto 32x32 grid
        # ─────────────────────────────────────────────────────────────────────

        # Update rolling buffer (drop oldest frame, add newest)
        L = np.roll(L, -1, axis=3)
        L[:, :, :, -1] = grid
        latest_grid = grid

        # Live preview: show L cone channel at 10× size
        preview_channel = L[:, :, 0, -1]   # L cone = rough luminance; you can superimpose the other cones on top to get what the cones see 
        # (becuase you are providing an RGB display with the channels it needs to display an image - this is unrelated to what our eyes are doing and more about how a screen/camera works) - but this will be black and white because you have chosen one cone. 
        preview = cv2.resize(
            ((preview_channel/V_MAX)* 255).astype(np.uint8),
            (320, 320),
            interpolation=cv2.INTER_NEAREST   # pixelated = shows grid clearly for us as a programmer to see - enlarge grid by 10x
        )
        cv2.putText(preview, f'{GRID_SIZE[0]}x{GRID_SIZE[1]} grid',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
        cv2.imshow('Stage 1 — L cone channel (press s to validate, q to quit)', preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            validate_frame(latest_grid, L, v_max=V_MAX)
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