"""
Editable parameters for the next training run.

ATMOSPHERE_SAMPLES is the total number of atmospheres generated per epoch.
They are split into train and validation according to TRAIN_FRAC. Nothing is
saved as a dataset.
"""

# ============================================================
# SOURCE
# ============================================================
SOURCE_WAVELENGTH = 635e-9
SOURCE_PIXEL_PITCH = 3.74e-6


# ============================================================
# TELESCOPE
# ============================================================
TELESCOPE_DIAMETER = 0.6
TELESCOPE_RESOLUTION = 128
TELESCOPE_SPIDERS = 0
TELESCOPE_SPIDERS_PX = 0
TELESCOPE_CENTRAL_OBSTRUCTION_PX = 0
TELESCOPE_SAMP = 2


# ============================================================
# PYRAMID WFS
# ============================================================
WFS_HEADS = 4
WFS_ALPHA = 2.8
WFS_RETURN_TYPE = "pupils"  # "pupils" | "full_frame"
WFS_FILTER_RATIO = 1.0
WFS_CROP_POS_NOISE = 2
WFS_CROP_SIZE_NOISE = 2
WFS_OFFSET = 10


# ============================================================
# NEURAL NETWORK
# ============================================================
MODEL = "TinyResNetWFS" # "ConvNeXtTiny", "GcVit", "TinyResNetWFS"
WTS = None
NN_RESOLUTION = 36


# ============================================================
# ATMOSPHERE — PER STAGE
# ============================================================
# Total online samples generated per epoch: train + validation.
ATMOSPHERE_SAMPLES = [100]

# D/r0 is sampled uniformly inside this interval, then r0 = D / (D/r0).
ATMOSPHERE_DR0_RANGE = [[5.0, 40.0]]
ATMOSPHERE_L0 = [25.0]
ATMOSPHERE_l0 = [1e-10]

# The number of entries determines the number of atmospheric layers.
ATMOSPHERE_FRACTIONAL_R0 = [[0.5, 0.3, 0.2]]

# Each range is divided into N contiguous intervals, one per layer.
# One random value is sampled inside each interval for every online batch.
ATMOSPHERE_WIND_SPEED_RANGE = [[0.0, 15.0]]
ATMOSPHERE_WIND_DIRECTION_RANGE = [[0.0, 360.0]]
ATMOSPHERE_ALTITUDE_RANGE = [[0.0, 10000]]

ATMOSPHERE_FRAME_RATE = [10000.0]
ATMOSPHERE_SEED = [1234]
ATMOSPHERE_VALIDATION_SEED_OFFSET = [10000000]

ATMOSPHERE_N_SUBHARMONIC_LEVELS = [3]
ATMOSPHERE_SUBHARMONIC_MODE = ["full"]          # "oopao" | "full"
ATMOSPHERE_DIRECTION_CONVENTION = ["oopao"]     # "cartesian" | "oopao"
ATMOSPHERE_NORMALIZE_FRACTIONAL_R0 = [True]
ATMOSPHERE_REMOVE_PISTON = [False]
ATMOSPHERE_STORE_COMPONENTS = [False]

# D/r0 refers to r0 at this wavelength. By default it is the source wavelength.
ATMOSPHERE_R0_REFERENCE_WAVELENGTH = [SOURCE_WAVELENGTH]

# When True, pass the generated atmospheric amplitude |field| to the WFS.
# When False, the WFS receives only the geometric telescope pupil.
ATMOSPHERE_SCINTILLATION = [False]

ATMOSPHERE_FROZEN_FLOW_MODE = ["analytic"]      # "analytic" | "periodic_screen"
ATMOSPHERE_PROPAGATION_MODE = ["geometric"]     # "geometric" | "asm_delta"
ATMOSPHERE_DELTA_MODE = ["final"]               # "final" | "per_step"
ATMOSPHERE_ASM_EXTRA_PIXELS = ["auto"]          # int | "auto" | None
ATMOSPHERE_ASM_MIN_PHYSICAL_MARGIN = [0.75]
ATMOSPHERE_ASM_PADDING_FACTOR = [2.0]
ATMOSPHERE_DELTA_WRAP_WARNING_THRESHOLD = [5.969026041820607]  # 1.9*pi
ATMOSPHERE_TEMPORAL_REANCHOR_INTERVAL = [0]

ATMOSPHERE_MODES = 97
DM_BASIS = True
DM_BASIS_TYPE = "ACTUATOR"  # "ACTUATOR" | "ZERNIKE"
DM_NAME = "BAX370_MRS"


# ============================================================
# TRAINING — PER STAGE
# ============================================================
PRECISION = "single"
NORM_TYPE = ["zscore"] 

# Phi-only spatial loss.
LOSS_TYPE = ["std"] # "std_grad_local", "std"
LOSS_EPS = [1e-8]

TRAIN_FRAC = [0.8]
BATCH_SIZE = [20]
EPOCHS = [200]
LR = [7.5e-4]
WEIGHT_DECAY = [0.0]
LR_GAMMA = [0.9]

CL_ITER = [20]
CL_GAIN_RANGE = [[0.3, 1.0]]
NOISE = [True]
