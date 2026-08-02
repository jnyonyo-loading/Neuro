# Camera to V1 Encoder
The goal of this code is to translate the pixels in camera feed into firing patterns that neurons in the visual cortex will recognise.
By stimulating V1, vision will be produced. V1 will orchestrate the downstream feedforward/feedback mechnasims, naturally engrained, to so.

## Requirements 
```bash
pip install opencv-python numpy matplotlib scipy
```
## Usage
```bash
python3 LMS_Config_v1.py
```

## Parameters
| Name | Meaning | Default |
|---|---|---|
| GRID_SIZE | electrode grid (w×h) | 32×32 |
| V_MAX | peak photoreceptor voltage swing (mV) | 30.0 |
| FRAME_RATE_HZ | camera capture rate | 30 |
| BUFFER_FRAMES | temporal history |10|

## Stage 1 — LMS_Config: camera → cone voltage
To see light enters the eye, and is converted into voltages along photoreceptors (e.g., Cones), and passed along to retinal ganglion cells as voltages.
Once at a Retinal Ganglion Cell, this analogous continous voltage will be converted into discrete packet of action potentials to be passed along the Axon into LGN to V1.
The goal of LMS_Config is to produce the initial voltages likely produced by the three cone subtypes (L,M,S), to be used in RGC encoder.

This will load up the camera and simulate what a patient should theoretically see with their electrode count.

A few parameters to adjust for:
- Your electrode count - how many are there h x w = grid size of the camera
- Frames per second rate
- Max voltage that the photoreceptors can take = which I have noted the upper range for = 30mV

Linearise our Gamma function - old cameras have standards that affect how brightness is scaled.
- This means that light being detected by the camera is not a direct estimation of true light intensity in a scene
- So we have to remove this standard to accurately capture the raw light intensity entering the camera (/eye) 
- Then we'll apply the transformations that mimick how our photoreceptors process the light intensity
- Enabling us to get a good approximation to the voltage generated, and firing signal that an eye would deliver to V1 

Define the bit-channels your camera has or you want for you image - 8-bit/12-bit etc. 
- divide every pixel by it in v = frame_rgb.astype(np.float32) / 255.0 
- this will normalise the grid to know where the brightest spot is by comparing other bits to it
- 8-bit channel means that there are 2^8 possible options/binary code that will represent that pixel

```bash
def linearise_gamma(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Camera pixels are gamma-encoded (sRGB). Undo that so pixel values
    are proportional to actual light intensity. Output: float32 in [0, 1].
    After this step you have  I (linear light intensity) received by photodiodes for RGB
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) ## reorders the channels to RGB to fit our standard, historically convention label for windows bitmap software was BGR, 
    v = frame_rgb.astype(np.float32) / 255.0 ## divides each pixel by 255 to normalise the scale - 0 to 1 - 8-bit channels for each colour pixel could inhabit
    linear = np.where(
        v < 0.04045, 
        v / 12.92,
        ((v + 0.055) / 1.055) ** 2.4
    ) ## for condition v<0.04045, apply A otherwise apply B; B is inverse power law; A
    return linear.astype(np.float32)
```

Once you have linearised the gamma function, we must now convert the light intensities into LMS voltages
- The 'photoreceptors' in the camera are photodiodes - small cells of lightsensistive coloured dyes (RGB)
- These dyes elicit a voltage when interacted with, similar to rods and cones.
- However they do not share the saturation point that physical receptors have. So as light intensity increases, the voltage increases too.
- Secondly the wavelengths captured by RGB, are not the same as those captured by L,M,S cones. They are close.


So we must convert the voltages induced by light in the RGBs to those that LMS cones will experience - using established empirically derived figure.
This describes what % of Red, Blue, Green wavelengths are present in eac L,M,S cones. 
These numbers act like weights/coefficients that will be multiplied with whatever voltages are outputted.

```bash
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
```

Once the proto-LMS voltages are produced, they will have a saturation point (non-linear) function applied onto it, to mimick to true biological behaviour.
A hill function (part of the Naka-Rushton equation). The L,M,S volatges now will be more akin to what is achieved in the eye.

```bash
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

```

Once we have our voltages, we now have to downsize (or downsample) them to match the volume of electrodes to whom we are sending a message.
- because they are limited in the number of inputs they can receive about a visual scene.

```bash
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
    ## INTER_AREA computes a proper area-weighted average of all source pixels falling inside each destination pixel's footprint
```

---


# Stage 2: Encoder (LMS voltages to firing patterns sent to V1)

In the Encoder these voltages are transformed into firing patterns along the axons
Our assumption is that most (~80%) of these signals are preserved when passing through the optical tract, the LGN, and V1
Our second assumption is that most of the information collected and usef for opponents process - detecting colour, and centre-surround - is handled by L and M cones (80-90%).
So we have focused our programs and maths on the L and M cones, the bipolar cells that receive the inputs to output a colour and luminescense opponent voltages.
As well as the process that transforms them voltages that induce firing rates along the axon.

## Requirements 
```bash
pip install opencv-python numpy matplotlib scipy
```
## Parameters

### Timing
| Name | Meaning | Default |
|---|---|---|
| `FRAME_RATE_HZ` | camera capture rate | 30 |
| `DT_MS` | time per frame (ms) — `1000/FRAME_RATE_HZ` | 33.3 |

### DoG spatial filter (pixels, 32×32 grid ≈ 20° visual field — *unverified, needs deriving from actual camera FOV*)
| Name | Meaning | Default | Source |
|---|---|---|---|
| `M_SIGMA_C` | M-pathway center width | 1.0 | illustrative, not measured |
| `M_SIGMA_S` | M-pathway surround width | 6.0 | illustrative, not measured |
| `M_WEIGHT` | M-pathway surround/center gain ratio | 0.65 | illustrative, not measured |
| `P_SIGMA_C` | P-pathway center width | 0.6 | illustrative, not measured |
| `P_SIGMA_S` | P-pathway surround width | 4.0 | illustrative, not measured |
| `P_WEIGHT` | P-pathway surround/center gain ratio | 0.85 | illustrative, not measured |

### Biphasic temporal kernel (ms)
| Name | Meaning | Default |
|---|---|---|
| `M_TAU1` | M-pathway fast (excitatory) decay constant | 15.0 |
| `M_TAU2` | M-pathway slow (inhibitory) decay constant | 45.0 |
| `M_NEG` | M-pathway inhibitory lobe depth relative to excitatory peak | 0.6 |
| `P_TAU1` | P-pathway fast (excitatory) decay constant | 40.0 |
| `P_TAU2` | P-pathway slow (inhibitory) decay constant | 120.0 |
| `P_NEG` | P-pathway inhibitory lobe depth relative to excitatory peak | 0.3 |

### Max firing rates
| Name | Meaning | Default |
|---|---|---|
| `M_MAX_HZ` | ceiling firing rate, magnocellular/parasol pathway | 100.0 Hz |
| `P_MAX_HZ` | ceiling firing rate, parvocellular/midget pathway | 50.0 Hz |

### First establish parameters
- there are fixed constants used in each transformation that are empirically derived. but we have chosen a ball park figure for this example. future versions will have better justifications.
- After each opponent (Vrg, and Vlum) is derived, they undergo a filtering process to have a time and space registered onto them. (ie. this is wherea and when this colour / object appears)
- each filter contributes to a final voltage - called a generator potential - this is fed to the RGCs to determine the strength of the firing rate (how important is this signal)
- the temporal filter occurs first - it's a convolution between frames - ie summing up the set of changes from now to the previous 10 frames
-- the neurons firing when they receive an input have two phases per frame - a fast excitatory phase and a slow inhibitory phase - how long this takes is an empircally derived figure (t1, t2)
-- varies depending on which RGC is receiving the input - bipolar cells (Parvo path) vs parasol cells (Magno path)

``` bash
# ── Temporal kernel ────────────────────────────────────────────────────────────

def build_kernel(tau1, tau2, neg_gain, n_frames, dt_ms):
    lags = np.arange(n_frames - 1, -1, -1) * dt_ms
    k = np.exp(-lags**2 / (2*tau1**2)) - neg_gain * np.exp(-lags**2 / (2*tau2**2))
    if np.max(np.abs(k)) > 0:
        k /= np.max(np.abs(k))
    return k.astype(np.float32)

```

- spatial filter occurs after the time generator potential has been formed - it's a convolution too - but it compares the brightness/blurrines of the receptive field in that space. 
-- it creates a centre sharp map and blurred surround map and gives the pixels values for each map. it weights more heavily for those in the centre of the scene, and less for those in the blurry background (Far from the centre) - similar to a normal distribution curve (like Gaussian curve) 
-- there are empirically dervied figures.

``` bash
# ── Spatial filter ─────────────────────────────────────────────────────────────

def dog_filter(frame, sigma_c, sigma_s, weight):
    centre   = gaussian_filter(frame, sigma=sigma_c)
    surround = gaussian_filter(frame, sigma=sigma_s)
    return centre - weight * surround

```

``` bash
# ── Parameters (Chichilnisky 2001, Pillow 2008) ────────────────────────────────
FRAME_RATE_HZ = 30
DT_MS  = 1000 / FRAME_RATE_HZ
DT_S   = DT_MS / 1000.0

# DoG spatial (pixels, 32x32 grid ≈ 20° visual field) - "illustrative, not measured"
M_SIGMA_C = 1.0;  M_SIGMA_S = 6.0;  M_WEIGHT = 0.65
P_SIGMA_C = 0.6;  P_SIGMA_S = 4.0;  P_WEIGHT = 0.85

# Biphasic temporal kernel (ms)
M_TAU1 = 15.0;  M_TAU2 = 45.0;  M_NEG = 0.6
P_TAU1 = 40.0;  P_TAU2 = 120.0; P_NEG = 0.3

# Max firing rates
M_MAX_HZ = 100.0
P_MAX_HZ = 50.0
```

Then we apply the sums to deduce the colour and spatial opponents

Once oppoents and their generator potential have been determined
a soft rectification is applied to make the voltages positive (as there are no negative action potential)

and the voltages are then normalised (0 to 1), to deduce which areas are most centre vs not; most bright/coloured vs not
- divide the voltages in the receptive field by the biggest voltage which tells what proportion (%) of the scene it represents
- after which this percentage is applied to the max firing rate for the axons along the pathways (Magno v Parvo) to deduce the number of impulses sent along each axon.

```bash
# ── Nonlinearity ───────────────────────────────────────────────────────────────

def soft_rectify(g, max_hz):
    rate = np.log1p(np.exp(g * 5.0))
    rate = rate / (np.max(rate) + 1e-8) * max_hz #normalisation
    return np.clip(rate, 0, max_hz)

``` 
