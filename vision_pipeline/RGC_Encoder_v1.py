"""
Stage 2: RGC encoding pipeline
L(x, y, t) → synthetic RGC spike trains

Install: pip install scipy
Run standalone: python RGC_Encoder_v1.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ── Parameters (Chichilnisky 2001, Pillow 2008) ────────────────────────────────
FRAME_RATE_HZ = 30
DT_MS  = 1000 / FRAME_RATE_HZ
DT_S   = DT_MS / 1000.0

# DoG spatial (pixels, 32x32 grid ≈ 20° visual field)
M_SIGMA_C = 1.0;  M_SIGMA_S = 6.0;  M_WEIGHT = 0.65
P_SIGMA_C = 0.6;  P_SIGMA_S = 4.0;  P_WEIGHT = 0.85

# Biphasic temporal kernel (ms)
M_TAU1 = 15.0;  M_TAU2 = 45.0;  M_NEG = 0.6
P_TAU1 = 40.0;  P_TAU2 = 120.0; P_NEG = 0.3

# Max firing rates
M_MAX_HZ = 100.0
P_MAX_HZ = 50.0

# ── Spatial filter ─────────────────────────────────────────────────────────────

def dog_filter(frame, sigma_c, sigma_s, weight):
    centre   = gaussian_filter(frame, sigma=sigma_c)
    surround = gaussian_filter(frame, sigma=sigma_s)
    return centre - weight * surround

# ── Temporal kernel ────────────────────────────────────────────────────────────

def build_kernel(tau1, tau2, neg_gain, n_frames, dt_ms):
    lags = np.arange(0, n_frames) * dt_ms
    k = np.exp(-lags**2 / (2*tau1**2)) - neg_gain * np.exp(-lags**2 / (2*tau2**2))
    if np.max(np.abs(k)) > 0:
        k /= np.max(np.abs(k))
    return k.astype(np.float32)

# ── Nonlinearity ───────────────────────────────────────────────────────────────

def soft_rectify(g, max_hz):
    rate = np.log1p(np.exp(g * 5.0))
    rate = rate / (np.max(rate) + 1e-8) * max_hz
    return np.clip(rate, 0, max_hz)

# ── Poisson spikes ─────────────────────────────────────────────────────────────

def poisson_spikes(rate_hz, dt_s):
    prob = np.clip(rate_hz * dt_s, 0, 1)
    return (np.random.rand(*rate_hz.shape) < prob).astype(np.float32)

# ── Encoder class ──────────────────────────────────────────────────────────────

class RGCEncoder:
    def __init__(self, n_frames=10, dt_ms=33.3):
        self.dt_s = dt_ms / 1000.0
        self.kernel_M = build_kernel(M_TAU1, M_TAU2, M_NEG, n_frames, dt_ms)
        self.kernel_P = build_kernel(P_TAU1, P_TAU2, P_NEG, n_frames, dt_ms)

    def encode(self, L):
        # Input signals
        lum   = (L[:,:,0,:] + L[:,:,1,:]) / 2.0   # luminance for M
        chrom =  L[:,:,0,:] - L[:,:,1,:]           # L-M chromatic for P

        # Temporal filtering
        g_M = np.tensordot(lum,   self.kernel_M, axes=([2],[0]))
        g_P = np.tensordot(chrom, self.kernel_P, axes=([2],[0]))

        # Spatial filtering
        g_M = dog_filter(g_M, M_SIGMA_C, M_SIGMA_S, M_WEIGHT)
        g_P = dog_filter(g_P, P_SIGMA_C, P_SIGMA_S, P_WEIGHT)

        # Firing rates
        rates_M = soft_rectify(g_M, M_MAX_HZ)
        rates_P = soft_rectify(g_P, P_MAX_HZ)

        # Spikes
        spikes_M = poisson_spikes(rates_M, self.dt_s)
        spikes_P = poisson_spikes(rates_P, self.dt_s)

        return spikes_M, spikes_P, rates_M, rates_P

# ── Validation plot ────────────────────────────────────────────────────────────

def validate_encoder(L, encoder):
    spikes_M, spikes_P, rates_M, rates_P = encoder.encode(L)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    lum_frame   = (L[:,:,0,-1] + L[:,:,1,-1]) / 2
    chrom_frame =  L[:,:,0,-1] - L[:,:,1,-1]

    axes[0,0].imshow(lum_frame,   cmap='gray',   vmin=0, vmax=1)
    axes[0,0].set_title('Input: luminance (L+M)/2')
    axes[0,1].imshow(chrom_frame, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes[0,1].set_title('Input: chromatic L-M')
    axes[0,2].imshow(rates_M, cmap='hot', vmin=0, vmax=M_MAX_HZ)
    axes[0,2].set_title(f'M firing rate (max {rates_M.max():.1f} Hz)')
    axes[0,3].imshow(rates_P, cmap='hot', vmin=0, vmax=P_MAX_HZ)
    axes[0,3].set_title(f'P firing rate (max {rates_P.max():.1f} Hz)')
    axes[1,0].imshow(spikes_M, cmap='binary_r')
    axes[1,0].set_title(f'M spikes ({int(spikes_M.sum())} events)')
    axes[1,1].imshow(spikes_P, cmap='binary_r')
    axes[1,1].set_title(f'P spikes ({int(spikes_P.sum())} events)')

    axes[1,2].plot(encoder.kernel_M, 'b-o', markersize=4)
    axes[1,2].axhline(0, color='gray', lw=0.5)
    axes[1,2].set_title('M temporal kernel (parasol)')
    axes[1,2].set_xlabel('Frame (oldest → newest)')

    axes[1,3].plot(encoder.kernel_P, 'r-o', markersize=4)
    axes[1,3].axhline(0, color='gray', lw=0.5)
    axes[1,3].set_title('P temporal kernel (midget)')
    axes[1,3].set_xlabel('Frame (oldest → newest)')

    for ax in axes[0]:
        ax.axis('off')

    fig.suptitle('Stage 2 — RGC encoding output', fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n── Stage 2 diagnostics ──────────────────────")
    print(f"  M spikes/frame : {int(spikes_M.sum())}  ({spikes_M.mean()*100:.1f}% units fired)")
    print(f"  P spikes/frame : {int(spikes_P.sum())}  ({spikes_P.mean()*100:.1f}% units fired)")
    print(f"  M rate range   : {rates_M.min():.1f} – {rates_M.max():.1f} Hz")
    print(f"  P rate range   : {rates_P.min():.1f} – {rates_P.max():.1f} Hz")
    print(f"  M kernel       : {encoder.kernel_M.round(2)}")
    print(f"  P kernel       : {encoder.kernel_P.round(2)}")
    print("─────────────────────────────────────────────\n")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Stage 2 standalone test — synthetic grey scene with bright bar onset")

    H, W, N = 32, 32, 10
    L_test = np.ones((H, W, 3, N), dtype=np.float32) * 0.3
    L_test[:, 12:20, :, 5:] = 0.9   # bright bar appears halfway through buffer

    encoder = RGCEncoder(n_frames=N, dt_ms=1000/30)
    validate_encoder(L_test, encoder)

    print("To integrate with Stage 1 (LMS_Config_v1.py):")
    print("  from RGC_Encoder_v1 import RGCEncoder")
    print("  encoder = RGCEncoder()")
    print("  spikes_M, spikes_P, rates_M, rates_P = encoder.encode(L)")