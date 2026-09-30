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
# Spiders
TELESCOPE_SPIDERS = 0 # 4 <-- RAMA
TELESCOPE_SPIDERS_PX = 0 # 5.0 <-- RAMA 
# Cada ángulo representa UN brazo.
#
# Convención:
#   0 deg   -> derecha
#   90 deg  -> abajo
#   180 deg -> izquierda
#   270 deg -> arriba
#
# None -> comportamiento automático anterior.
#
# Ejemplo para cuatro brazos:
TELESCOPE_SPIDER_ANGLES_DEG = None # [-60.0, 42.0, 115.0, 210.0] <-- RAMA
# Obstrucción central
TELESCOPE_CENTRAL_OBSTRUCTION_PX = 0.0# 45.0 <-- RAMA
# Desplazamiento respecto del centro geométrico [px]
#
# +x -> derecha
# +y -> abajo
TELESCOPE_CENTRAL_OBSTRUCTION_OFFSET_PX = (
    0.0,
    0.0,
)
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
MODEL = "ConvNeXtTiny" # "ConvNeXtTiny", "GcVit", "TinyResNetWFS"
WTS = None
NN_RESOLUTION = 36


# ============================================================
# ATMOSPHERE — PER STAGE
# ============================================================
# Total online samples generated per epoch: train + validation.
ATMOSPHERE_SAMPLES = [100000]

# D/r0 is sampled uniformly inside this interval, then r0 = D / (D/r0).
ATMOSPHERE_DR0_RANGE = [[5.0, 40.0]]
# Distribution used to generate atmospheric strength.
#
# "uniform_dr0":
#     D/r0 ~ Uniform(dr0_min, dr0_max)
#     Recommended for the current training definition.
#
# "uniform_r0":
#     r0 ~ Uniform(r0_min, r0_max)
ATMOSPHERE_R0_SAMPLING = [
    "uniform_dr0"
]
ATMOSPHERE_L0 = [25.0]
ATMOSPHERE_l0 = [1e-10]

# fractional_r0 entries are associated from low to high altitude.
ATMOSPHERE_FRACTIONAL_R0 = [[0.5, 0.3, 0.2]]

# Atmospheric layer sampling strategy.
#
# "altitude_stratified":
#     Recommended for training.
#
#     altitude:
#         One random layer is generated inside each vertical
#         altitude interval.
#
#     wind speed:
#         Each layer samples independently over the complete
#         configured wind-speed range.
#
#     wind direction:
#         Each layer samples independently over the complete
#         angular range.
#
# "legacy_stratified":
#     Previous behaviour. Altitude, wind speed and direction
#     are all stratified according to layer index.
ATMOSPHERE_LAYER_SAMPLING = [
    "altitude_stratified"
]

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
# CAMERA NOISE
# ============================================================

CAMERA_LOW_PEAK_E = (30.0, 150.0, True)
CAMERA_LOW_BG_E = (0.5, 5.0, True)
CAMERA_LOW_READ_SIGMA_E = (1.5, 3.5, False)
CAMERA_LOW_BIAS_DN = (0.0, 3.0, False)

CAMERA_NORMAL_PEAK_E = (150.0, 1500.0, True)
CAMERA_NORMAL_BG_E = (0.2, 3.0, True)
CAMERA_NORMAL_READ_SIGMA_E = (0.8, 2.0, False)
CAMERA_NORMAL_BIAS_DN = (0.0, 3.0, False)

CAMERA_GOOD_PEAK_E = (1500.0, 7000.0, True)
CAMERA_GOOD_BG_E = (0.05, 1.0, True)
CAMERA_GOOD_READ_SIGMA_E = (0.5, 1.2, False)
CAMERA_GOOD_BIAS_DN = (0.0, 3.0, False)

CAMERA_P_LOW = 0.30
CAMERA_P_NORMAL = 0.55
CAMERA_P_GOOD = 0.15

CAMERA_PARAMETER_MODE = "per_sample"
CAMERA_SHOT_NOISE = "poisson"

CAMERA_OUTPUT_MODE = "Mono8"
CAMERA_MONO16_ALIGN = "lsb"
CAMERA_USE_STE_ADC = False

CAMERA_AUTO_GAIN = True
CAMERA_ADC_HEADROOM = 0.90
CAMERA_MIN_GAIN_E_PER_DN = 1e-6

CAMERA_ADD_PRNU = True
CAMERA_PRNU_SIGMA = 0.005

CAMERA_ADD_DSNU = True
CAMERA_DSNU_SIGMA_E = 0.2

CAMERA_SEED_OFFSET = 30_000_000


# ============================================================
# TRAINING — PER STAGE
# ============================================================
PRECISION = "single"
NORM_TYPE = ["zscore_global"] # "zscore", "zscore_global"

# Phi-only spatial loss.
LOSS_TYPE = ["std"] # "std_grad_local", "std"
LOSS_EPS = [1e-8]

TRAIN_FRAC = [0.8]
BATCH_SIZE = [20]
EPOCHS = [200]
LR = [7.5e-4]
WEIGHT_DECAY = [0.0]
LR_GAMMA = [0.999]

CL_ITER = [20]
CL_GAIN_RANGE = [[0.3, 1.0]]
NOISE = [True]
