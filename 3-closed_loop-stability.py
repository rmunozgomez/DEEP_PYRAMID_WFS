import os
import gc
import argparse
import torch
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal, Union, Any

from GENERAL_FUNCTIONS.camera_noise import *
from GENERAL_FUNCTIONS.functions_torch import *
from NN.model_manager import ModelManager

from ATMOSPHERE.atmosphere import Atmosphere
from MODAL_BASIS.Zernike import (
    get_zernike,
    get_zernike_on_pupil,
    zernike_compose_torch,
    zernike_decompose_torch,
)

# =========================================================
# USER CONFIGURATION (EDIT THIS TOP BLOCK ONLY)
# =========================================================

# -------------------------
# Shared test configuration
# -------------------------
DEFAULT_TEST_NAME = "Stability"
DEFAULT_DEVICE = "cuda:4"

seed = 400

# closed loop parameters
# NOTE:
#   n_samples below is retained only for inherited/legacy helper functions.
#   The LONG-STABILITY test does NOT use it. Its total number of frames is
#   derived from stability_duration_minutes and frame_rate.
n_samples = 25000
cl_sample = 100

# Long-duration stability test
stability_duration_minutes = 10.0
stability_max_metric_points = 1000

kp = 0.0
ki = 1.0
noise_flag = True
    
# Pupila física usada en TODO el flujo óptico: atmósfera, descomposición,
# control, propagación WFS, WFE, Strehl, PSF y plots. Las matrices de base
# se conservan exactamente como fueron entrenadas/calibradas.
use_telescope_pupil_in_propagation = True
propagation_central_obstruction_diam_px = 0.0
propagation_spiders = 0
propagation_spiders_px = 0.0
propagation_soft_edge_px = 0.0

# La familia de spiders recibe una rotación aleatoria, pero queda fija durante
# toda la ejecución para representar una pupila física estática.
propagation_pupil_seed = seed

# Noise test mode:
#   noise_level = 0..10
#       0  -> mejor caso realista: alta señal, bajo background, bajo read noise
#       10 -> peor caso realista: baja señal, alto background, alto read noise
#   "all" ejecuta todos los niveles 0..10 en carpetas separadas.
#
# Importante:
#   El nivel de ruido define una distribución BASE fija.
#   En cada forward solo hay pequeñas variaciones alrededor de esa base,
#   evitando cambios bruscos tipo weak/normal/strong.
noise_level = "all"
noise_output_mode = "Mono8"   # "Mono8", "Mono12" o "Mono16"
noise_adc_headroom = 0.90      # usa hasta el 90% del ADC para evitar saturación
noise_param_jitter = 0.10      # variación pequeña alrededor del nivel fijo

phase_colormap = "viridis"
propagation_colormap = "hot"
psf_colormap = "hot"

# atmosphere parameters
#
# r0 has the same two modes as the new ATMOSPHERE generator:
#   r0 = 0.03             -> fixed r0 for the complete temporal sequence
#   r0 = [0.02, 0.08]     -> one r0 ~ U(min,max) is drawn when gen() starts
#                             and remains fixed during every update() frame.
#
# This closed-loop script uses batch_size=1 because it evaluates one common
# evolving atmosphere shared by every model, exactly like the old script.
r0 = 0.02
fractional_r0   = [0.5, 0.3, 0.2]
wind_direction  = [90, 240, 120]
wind_speed      = [8.0, 10.0, 10.0]
altitude        = [0.0, 1000.0, 2000.0]

L0              = 15
l0              = 1e-10
frame_rate      = 1000

# Physical support used by ATMOSPHERE. Keep 0.6 to reproduce the old test.
# Set to None if you prefer to use the telescope diameter stored in the model.
atmosphere_telescope_diameter = 0.6

# Phase wavelength. None -> wavelength stored in the first model config.
atmosphere_wavelength = None
atmosphere_r0_reference_wavelength = 500e-9

# New ATMOSPHERE options. The defaults below reproduce the old geometric test.
atmosphere_scintillation = False
atmosphere_propagation_mode = "geometric"     # "geometric" or "asm_delta"
atmosphere_delta_mode = "final"               # "final" or "per_step"
atmosphere_frozen_flow_mode = "analytic"      # "analytic" or "periodic_screen"
atmosphere_direction_convention = "oopao"
atmosphere_n_subharmonic_levels = 3
atmosphere_subharmonic_mode = "full"          # "full" or "oopao"
atmosphere_normalize_fractional_r0 = True
atmosphere_remove_piston = False
atmosphere_temporal_reanchor_interval = 256

# ASM parameters are ignored by the geometric mode but are ready for scintillation.
atmosphere_asm_extra_pixels = "auto"
atmosphere_asm_min_physical_margin = 0.75
atmosphere_asm_padding_factor = 2.0
atmosphere_delta_wrap_warning_threshold = 1.9 * np.pi

# Performance: warm fixed FFT/cuFFT paths once before the real sequence.
atmosphere_warmup = True
atmosphere_warmup_updates = 2

# The sequence is generated once and reused by every model/noise level.
# "device" reproduces the old script's fast behavior (the whole atmosphere is
# already on the GPU). Use "cpu" only if you prefer to save VRAM.
atmosphere_sequence_storage = "device"       # "device" or "cpu"

# Optional reproducibility dump. Disabled by default because 10k x 128 x 128
# float32 frames are large and saving them does not improve the closed-loop run.
save_atmosphere_sequence_pt = False

# signo del lazo
loop_sign = -1.0   # si diverge, probar +1.0

# opcional: limitar el integrador para evitar windup
use_integrator_clamp = False
integrator_limit = 5.0

# -------------------------
# Models to compare
# -------------------------
MODELS_TO_TEST = [
    {
        "train_path": "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS_EVOLVE_V2/TRAIN/single/phiRes_128/nnRes_32/DM_IDEAL/ZERNIKE/nModes_209/MODEL_ConvNeXtTiny/PYR4_32_std_grad_local",
        "stage": 0,
        "Name": "ConvNeXTiny_32_std_grad_local_209",
    },
    
]

# =========================================================
# PARSER
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop PI comparison using the Torch ATMOSPHERE generator")
    parser.add_argument("--test_name", type=str, default=DEFAULT_TEST_NAME, help="Name of the test output folder")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to run the test on, e.g. cuda:0 or cpu")
    parser.add_argument("--noise_level", type=str, default=noise_level, help="Noise level: integer 0..10, or 'all'")
    parser.add_argument("--noise_output_mode", type=str, default=noise_output_mode, choices=["Mono8", "Mono12", "Mono16"], help="ADC quantization mode")
    parser.add_argument("--noise_param_jitter", type=float, default=noise_param_jitter, help="Small relative variation around the fixed noise level")
    parser.add_argument(
        "--duration_minutes",
        type=float,
        default=stability_duration_minutes,
        help="Physical duration of the stability test in minutes.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=stability_max_metric_points,
        help="Maximum number of WFE/Strehl samples stored and plotted.",
    )
    parser.add_argument(
        "--propagation_obstruction_px",
        type=float,
        default=propagation_central_obstruction_diam_px,
        help="Diámetro en píxeles de la obstrucción central usada solo por WFS.propagate.",
    )
    parser.add_argument(
        "--propagation_spiders",
        type=int,
        default=propagation_spiders,
        help="Número de spiders equiespaciados usados solo por WFS.propagate.",
    )
    parser.add_argument(
        "--propagation_spiders_px",
        type=float,
        default=propagation_spiders_px,
        help="Ancho de cada spider en píxeles usado solo por WFS.propagate.",
    )
    parser.add_argument(
        "--disable_propagation_obstruction",
        action="store_true",
        help="Desactiva obstrucción y spiders en la propagación, sin cambiar la pupila ideal.",
    )
    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================

def circular_pupil(
    n: int,
    *,
    device=None,
    dtype=torch.float32,
    soft_edge_px: float = 0.0,
):
    """Genera una pupila circular de shape (1,1,n,n)."""
    if n <= 0:
        raise ValueError("n debe ser > 0")

    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    radius_px = min(cx, cy, (n - 1 - cx), (n - 1 - cy))

    y = torch.arange(n, device=device, dtype=torch.float32)
    x = torch.arange(n, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    R = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    if soft_edge_px <= 0.0:
        pupil = (R <= radius_px).to(dtype)
        return pupil.unsqueeze(0).unsqueeze(0)

    w = float(soft_edge_px)
    pupil = torch.ones((n, n), device=device, dtype=torch.float32)
    pupil = torch.where(R >= (radius_px + w), torch.zeros_like(pupil), pupil)

    trans = (R > radius_px) & (R < (radius_px + w))
    t = (R[trans] - radius_px) / w
    pupil[trans] = 0.5 * (1.0 + torch.cos(torch.pi * t))
    return pupil.to(dtype).unsqueeze(0).unsqueeze(0)


def circular_pupil_telescope(
    n: int,
    *,
    device=None,
    dtype=torch.float32,
    soft_edge_px: float = 0.0,
    spiders: int = 0,
    spiders_px: float = 0.0,
    central_obstruction_diam_px: float = 0.0,
    theta0: Optional[torch.Tensor] = None,
):
    """
    Pupila circular con obstrucción central y spiders radiales equiespaciados.

    ``theta0`` permite fijar la rotación inicial. Si es None, se genera una
    rotación aleatoria. La salida tiene shape (1,1,n,n).
    """
    if n <= 0:
        raise ValueError("n debe ser > 0")
    if spiders < 0:
        raise ValueError("spiders debe ser >= 0")
    if spiders_px < 0:
        raise ValueError("spiders_px debe ser >= 0")
    if central_obstruction_diam_px < 0:
        raise ValueError("central_obstruction_diam_px debe ser >= 0")
    if central_obstruction_diam_px > n:
        raise ValueError("central_obstruction_diam_px no puede ser mayor que n")

    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    radius_px = min(cx, cy, (n - 1 - cx), (n - 1 - cy))

    y = torch.arange(n, device=device, dtype=torch.float32)
    x = torch.arange(n, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    Xc = X - cx
    Yc = Y - cy
    R = torch.sqrt(Xc**2 + Yc**2)

    pupil = circular_pupil(
        n=n,
        device=device,
        dtype=torch.float32,
        soft_edge_px=soft_edge_px,
    ).squeeze(0).squeeze(0)

    if central_obstruction_diam_px > 0.0:
        r_obs = float(central_obstruction_diam_px) / 2.0
        pupil = pupil * (R >= r_obs).to(torch.float32)

    if spiders > 0 and spiders_px > 0.0:
        if theta0 is None:
            theta0 = torch.rand((), device=device) * (2.0 * torch.pi / spiders)
        else:
            theta0 = torch.as_tensor(theta0, device=device, dtype=torch.float32)

        spider_mask = torch.ones((n, n), device=device, dtype=torch.float32)
        half_width = float(spiders_px) / 2.0

        for k in range(spiders):
            theta = theta0 + k * 2.0 * torch.pi / spiders
            ux = torch.cos(theta)
            uy = torch.sin(theta)

            dist_to_line = torch.abs(-uy * Xc + ux * Yc)
            radial_coord = ux * Xc + uy * Yc
            one_arm = radial_coord >= 0.0

            spider_region = (
                (dist_to_line <= half_width)
                & one_arm
                & (R <= radius_px)
            )
            spider_mask = torch.where(
                spider_region,
                torch.zeros_like(spider_mask),
                spider_mask,
            )

        pupil = pupil * spider_mask

    return pupil.to(dtype).unsqueeze(0).unsqueeze(0)


def build_wfs_propagation_pupil(
    ideal_pupil: torch.Tensor,
    *,
    enabled: bool,
    central_obstruction_diam_px: float,
    spiders: int,
    spiders_px: float,
    soft_edge_px: float = 0.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Construye la pupila usada únicamente por el WFS.

    La máscara telescópica se multiplica por ``ideal_pupil`` para conservar
    exactamente la apertura exterior y cualquier máscara propia de una base DM.
    ``ideal_pupil`` nunca se modifica in-place.
    """
    if not enabled:
        return ideal_pupil.clone()

    if ideal_pupil.ndim != 4 or ideal_pupil.shape[0] != 1 or ideal_pupil.shape[1] != 1:
        raise ValueError(
            "ideal_pupil debe tener shape (1,1,H,W). "
            f"Llegó {tuple(ideal_pupil.shape)}"
        )
    if ideal_pupil.shape[-2] != ideal_pupil.shape[-1]:
        raise ValueError("La pupila debe ser cuadrada para construir spiders y obstrucción.")

    n = int(ideal_pupil.shape[-1])
    theta0 = None
    if spiders > 0 and spiders_px > 0.0:
        if seed is None:
            theta0 = torch.rand((), device=ideal_pupil.device) * (2.0 * torch.pi / spiders)
        else:
            generator_device = ideal_pupil.device.type if ideal_pupil.device.type == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed))
            theta0 = torch.rand((), device=ideal_pupil.device, generator=generator) * (
                2.0 * torch.pi / spiders
            )

    telescope_mask = circular_pupil_telescope(
        n=n,
        device=ideal_pupil.device,
        dtype=ideal_pupil.dtype,
        soft_edge_px=soft_edge_px,
        spiders=spiders,
        spiders_px=spiders_px,
        central_obstruction_diam_px=central_obstruction_diam_px,
        theta0=theta0,
    )
    return ideal_pupil.clone() * telescope_mask


# =========================================================
# CAMERA NOISE AUGMENTER
# =========================================================
# Diseño:
#   - Sin blur.
#   - Sin pixel_response_box.
#   - Sin full_well clipping.
#   - Con shot noise, background, readout noise, PRNU/DSNU opcional.
#   - Con cuantización ADC siempre activa.
#   - Auto-gain para que la imagen quepa dentro de 255/4095 sin saturar.
#
# Cambio principal:
#   En vez de weak/normal/strong aleatorio, ahora se usa noise_level=0..10.
#   Cada nivel define una distribución base fija y en cada forward solo se
#   aplican pequeñas variaciones alrededor de esa base.
#
# Nota:
#   El augmenter funciona con cualquier tensor (B,C,H,W).
#   Para I_crop normalmente C=4; para I_full puede ser C=1.
# =========================================================

ParamMode = Literal["per_batch", "per_sample", "per_channel"]
ShotNoiseMode = Literal["poisson", "gaussian"]
OutputMode = Literal["Mono8", "Mono12", "Mono16"]


@dataclass
class Range:
    low: float
    high: float
    log: bool = False

    def sample(
        self,
        shape: Tuple[int, ...],
        like: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        u = torch.rand(shape, device=like.device, dtype=like.dtype, generator=generator)

        if self.log:
            lo = torch.log(torch.tensor(self.low, device=like.device, dtype=like.dtype))
            hi = torch.log(torch.tensor(self.high, device=like.device, dtype=like.dtype))
            return torch.exp(lo + (hi - lo) * u)

        return self.low + (self.high - self.low) * u


@dataclass
class CameraNoiseAugmentConfig:
    # Distribuciones ya calculadas para un nivel fijo 0..10
    peak_e: Range
    bg_e: Range
    read_sigma_e: Range
    bias_dn: Range

    parameter_mode: ParamMode = "per_channel"
    shot_noise: ShotNoiseMode = "poisson"

    output_mode: OutputMode = "Mono8"
    mono16_align: AlignMode = "lsb"
    use_ste_adc: bool = False

    auto_gain: bool = True
    adc_headroom: float = 0.90
    min_gain_e_per_dn: float = 1e-6

    add_prnu: bool = True
    prnu_sigma: float = 0.005

    add_dsnu: bool = True
    dsnu_sigma_e: float = 0.2

    noise_level: int = 5
    return_metadata: bool = False


def _adc_max_dn(mode: OutputMode) -> float:
    if mode == "Mono8":
        return 255.0
    if mode == "Mono12":
        return 4095.0
    if mode == "Mono16":
        # Asumimos 12 bits efectivos dentro del contenedor Mono16.
        return 4095.0
    raise ValueError("output_mode debe ser 'Mono8', 'Mono12' o 'Mono16'")


def _param_shape(x: torch.Tensor, mode: ParamMode) -> Tuple[int, int]:
    x = ensure_4d(x)
    B, C, _, _ = x.shape

    if mode == "per_batch":
        return (1, 1)
    if mode == "per_sample":
        return (B, 1)
    if mode == "per_channel":
        return (B, C)

    raise ValueError("parameter_mode debe ser 'per_batch', 'per_sample' o 'per_channel'")


def _broadcast_param(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if p.ndim != 2:
        raise ValueError(f"Se esperaba parámetro 2D. Llegó {tuple(p.shape)}")
    return p[:, :, None, None].to(device=x.device, dtype=x.dtype)


def _interp_log(a: float, b: float, t: float) -> float:
    """
    Interpolación logarítmica entre a y b.
    Útil para parámetros con escala multiplicativa como peak_e y bg_e.
    """
    return float(np.exp(np.log(a) * (1.0 - t) + np.log(b) * t))


def _interp_lin(a: float, b: float, t: float) -> float:
    return float(a * (1.0 - t) + b * t)


def _range_around(base: float, jitter: float, *, log: bool, min_value: float = 1e-12) -> Range:
    """
    Crea un rango pequeño alrededor de un valor base.

    jitter=0.10 significa aproximadamente ±10%.
    Para log=True, el muestreo es multiplicativo.
    """
    jitter = float(max(jitter, 0.0))
    lo = max(base * (1.0 - jitter), min_value)
    hi = max(base * (1.0 + jitter), lo + min_value)
    return Range(lo, hi, log=log)


def validate_noise_level(noise_level: Union[int, str]) -> int:
    try:
        level = int(noise_level)
    except Exception as exc:
        raise ValueError("noise_level debe ser un entero entre 0 y 10, o 'all' en main().") from exc

    if level < 0 or level > 10:
        raise ValueError("noise_level debe estar entre 0 y 10.")

    return level


def build_noise_config_from_level(
    noise_level: int,
    output_mode: OutputMode = "Mono8",
    *,
    adc_headroom: float = 0.90,
    param_jitter: float = 0.10,
    return_metadata: bool = False,
) -> CameraNoiseAugmentConfig:
    """
    Construye una configuración de ruido fija para un nivel 0..10.

    level=0:
        alta señal, bajo background, bajo read noise.

    level=10:
        baja señal, alto background, alto read noise.

    La distribución se mantiene estable: en cada forward se muestrean pequeñas
    variaciones alrededor del valor base del nivel.
    """
    L = validate_noise_level(noise_level)
    t = L / 10.0

    # Señal: decrece con el nivel de ruido.
    # 0 -> ~9000 e- peak, 10 -> ~35 e- peak.
    peak_base = _interp_log(9000.0, 35.0, t)

    # Background: aumenta con el nivel.
    # 0 -> ~0.03 e-, 10 -> ~55 e-.
    bg_base = _interp_log(0.03, 55.0, t)

    # Readout noise: aumenta con el nivel, rango realista para pruebas robustas.
    # 0 -> 0.35 e-, 10 -> 8 e-.
    read_base = _interp_lin(0.35, 8.0, t)

    # Bias digital pequeño. No es crítico porque después normalizas.
    bias_base = _interp_lin(0.5, 4.0, t)

    # FPN también crece suavemente con el nivel.
    prnu_sigma = _interp_lin(0.002, 0.012, t)
    dsnu_sigma_e = _interp_lin(0.05, 0.60, t)

    return CameraNoiseAugmentConfig(
        peak_e=_range_around(peak_base, param_jitter, log=True, min_value=1e-6),
        bg_e=_range_around(bg_base, param_jitter, log=True, min_value=1e-6),
        read_sigma_e=_range_around(read_base, param_jitter, log=False, min_value=1e-6),
        bias_dn=_range_around(bias_base, param_jitter, log=False, min_value=0.0),
        parameter_mode="per_channel",
        shot_noise="poisson",
        output_mode=output_mode,
        mono16_align="lsb",
        use_ste_adc=False,
        auto_gain=True,
        adc_headroom=adc_headroom,
        min_gain_e_per_dn=1e-6,
        add_prnu=True,
        prnu_sigma=prnu_sigma,
        add_dsnu=True,
        dsnu_sigma_e=dsnu_sigma_e,
        noise_level=L,
        return_metadata=return_metadata,
    )


def auto_electrons_to_dn_no_saturation(
    e: torch.Tensor,
    bias_dn: torch.Tensor,
    output_mode: OutputMode,
    *,
    headroom: float = 0.90,
    min_gain_e_per_dn: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = ensure_4d(e)

    max_dn = _adc_max_dn(output_mode)
    usable_max_dn = max_dn * headroom

    e_max = x.amax(dim=(-2, -1), keepdim=True).clamp_min(min_gain_e_per_dn)
    available_dn = (usable_max_dn - bias_dn).clamp_min(1.0)

    gain_e_per_dn = (e_max / available_dn).clamp_min(min_gain_e_per_dn)
    dn_analog = x / gain_e_per_dn + bias_dn

    return dn_analog, gain_e_per_dn


@dataclass
class CameraNoiseAugmenter:
    cfg: CameraNoiseAugmentConfig
    generator: Optional[torch.Generator] = None

    def __call__(self, I_unit: torch.Tensor) -> torch.Tensor:
        x = ensure_4d(I_unit)
        x = clamp_nonneg(x, 0.0)

        shape = _param_shape(x, self.cfg.parameter_mode)

        # Parámetros muestreados alrededor del nivel fijo.
        # No hay saltos entre regímenes distintos en cada forward.
        peak_e_raw = self.cfg.peak_e.sample(shape, x, self.generator)
        bg_e_raw = self.cfg.bg_e.sample(shape, x, self.generator)
        read_sigma_e_raw = self.cfg.read_sigma_e.sample(shape, x, self.generator)
        bias_dn_raw = self.cfg.bias_dn.sample(shape, x, self.generator)

        peak_e = _broadcast_param(peak_e_raw, x)
        bg_e = _broadcast_param(bg_e_raw, x)
        read_sigma_e = _broadcast_param(read_sigma_e_raw, x)
        bias_dn = _broadcast_param(bias_dn_raw, x)

        # Intensidad arbitraria -> electrones esperados.
        lam_e = to_expected_electrons_from_unit(x, peak_e=peak_e)

        # PRNU: variación multiplicativa pequeña de sensibilidad.
        if self.cfg.add_prnu and self.cfg.prnu_sigma > 0:
            prnu = 1.0 + self.cfg.prnu_sigma * randn_like_compat(lam_e, self.generator)
            lam_e = lam_e * prnu.clamp_min(0.0)

        # Background/dark antes de Poisson.
        lam_e = add_background_e(lam_e, bg_e=bg_e)

        # Shot noise.
        if self.cfg.shot_noise == "poisson":
            e = shot_noise_poisson(lam_e)
        elif self.cfg.shot_noise == "gaussian":
            e = shot_noise_gaussian_approx(lam_e, generator=self.generator)
        else:
            raise ValueError("shot_noise debe ser 'poisson' o 'gaussian'")

        # DSNU: offset espacial pequeño en electrones.
        if self.cfg.add_dsnu and self.cfg.dsnu_sigma_e > 0:
            dsnu = self.cfg.dsnu_sigma_e * randn_like_compat(e, self.generator)
            e = e + dsnu

        # Readout noise.
        e = read_noise_e(e, sigma_e=read_sigma_e, generator=self.generator)
        e = e.clamp_min(0.0)

        if not self.cfg.auto_gain:
            raise NotImplementedError("Esta versión usa auto_gain=True para evitar saturación ADC artificial.")

        # e- -> DN ajustando gain para que quepa dentro del ADC.
        dn, gain_e_per_dn = auto_electrons_to_dn_no_saturation(
            e,
            bias_dn=bias_dn,
            output_mode=self.cfg.output_mode,
            headroom=self.cfg.adc_headroom,
            min_gain_e_per_dn=self.cfg.min_gain_e_per_dn,
        )

        # Cuantización ADC siempre activa.
        dn_q = quantize_by_mode(
            dn,
            mode=self.cfg.output_mode,
            use_ste=self.cfg.use_ste_adc,
            align=self.cfg.mono16_align,
        )

        if self.cfg.return_metadata:
            metadata = {
                "noise_level": self.cfg.noise_level,
                "peak_e": peak_e_raw.detach(),
                "bg_e": bg_e_raw.detach(),
                "read_sigma_e": read_sigma_e_raw.detach(),
                "bias_dn": bias_dn_raw.detach(),
                "gain_e_per_dn": gain_e_per_dn.detach(),
                "adc_headroom": self.cfg.adc_headroom,
                "output_mode": self.cfg.output_mode,
                "prnu_sigma": self.cfg.prnu_sigma,
                "dsnu_sigma_e": self.cfg.dsnu_sigma_e,
            }
            return dn_q, metadata

        return dn_q


def build_noise_pipe(
    noise_level=5,
    output_mode="Mono8",
    return_metadata=False,
    param_jitter=0.10,
):
    """
    Construye el noise augmenter para closed loop.

    noise_level:
        Entero 0..10.
        0  -> mejor caso realista.
        10 -> peor caso realista.

    La distribución del ruido queda fijada por noise_level. En cada forward
    solo cambia ligeramente por param_jitter.
    """
    level = validate_noise_level(noise_level)

    cfg = build_noise_config_from_level(
        noise_level=level,
        output_mode=output_mode,
        adc_headroom=noise_adc_headroom,
        param_jitter=param_jitter,
        return_metadata=return_metadata,
    )

    return CameraNoiseAugmenter(cfg)

def load_model_bundle(model_info, device):
    train_path = model_info["train_path"]
    stage = model_info["stage"]

    nn_path = train_path + ("/stage_0/wts/model_full.pt" if stage is None else f"/stage_{stage}/wts/model_full.pt")
    config_path = train_path + "/config/all_cfg.pt"
    wfs_path = train_path + "/WFS/WFS.pt"

    cfg = torch.load(config_path, map_location="cpu")
    telescope_cfg = cfg["telescope_cfg"]
    wfs_cfg = cfg["wfs_cfg"]
    model_cfg = cfg["model_cfg"]
    stage_cfg = cfg["stages_cfg"][0 if stage is None else stage]
    atmosphere_cfg = stage_cfg["atmosphere"]
    train_cfg = stage_cfg["train"]

    precision = get_precision(train_cfg["precision"])

    WFS = torch.load(wfs_path, weights_only=False, map_location=device)
    WFS.crop_pos_noise = 0
    WFS.crop_size_noise = 0

    NN = ModelManager().load_full(nn_path, device=device).eval()

    return {
        "cfg": cfg,
        "telescope_cfg": telescope_cfg,
        "wfs_cfg": wfs_cfg,
        "model_cfg": model_cfg,
        "stage_cfg": stage_cfg,
        "atmosphere_cfg": atmosphere_cfg,
        "train_cfg": train_cfg,
        "precision": precision,
        "WFS": WFS,
        "NN": NN,
    }


# =========================================================
# TORCH ATMOSPHERE SEQUENCE
# =========================================================

@dataclass
class AtmosphereSequence:
    """One common temporal realization generated by ATMOSPHERE.

    phase:
        Tensor [T,1,H,W] in radians, stored either on the compute device or CPU.

    amplitude:
        Optional tensor [T,1,H,W]. It is stored only when
        atmosphere_scintillation=True. In geometric mode the optical amplitude
        is unity, so no extra tensor is allocated.
    """

    phase: torch.Tensor
    amplitude: Optional[torch.Tensor]
    sampled_r0: torch.Tensor
    info: Dict[str, Any]


def _source_wavelength_from_bundle(bundle: Dict[str, Any]) -> float:
    source_cfg = bundle["cfg"].get("source_cfg", bundle["cfg"].get("source", None))
    if source_cfg is None:
        # all_cfg.pt written by the current training main stores source_cfg at
        # the top level of cfg. Keep a clear failure rather than guessing.
        raise KeyError("No se encontró source_cfg en all_cfg.pt.")
    if isinstance(source_cfg, dict):
        return float(source_cfg["wavelength"])
    return float(source_cfg.wavelength)


@torch.no_grad()
def generate_atmosphere_sequence(
    *,
    ref_bundle: Dict[str, Any],
    resolution: int,
    model_telescope_diameter: float,
    precision,
    device: str,
    output_dir: str,
) -> AtmosphereSequence:
    """Generate the complete common atmosphere with the new Torch backend.

    This replaces get_evolutive_atmosphere/OOPAO. The Atmosphere object stays
    on the requested device and advances through update(). By default the full
    sequence also stays on that device, matching the old script and avoiding a
    host-to-device transfer inside every closed-loop frame.
    """

    D_atm = (
        float(model_telescope_diameter)
        if atmosphere_telescope_diameter is None
        else float(atmosphere_telescope_diameter)
    )

    wavelength = (
        _source_wavelength_from_bundle(ref_bundle)
        if atmosphere_wavelength is None
        else float(atmosphere_wavelength)
    )

    if atmosphere_scintillation and atmosphere_propagation_mode != "asm_delta":
        print(
            "[ATMOSPHERE] WARNING: scintillation=True with propagation_mode!='asm_delta'. "
            "The geometric field has unit amplitude, so no scintillation will be present."
        )

    atm = Atmosphere(
        batch_size=1,
        resolution=int(resolution),
        telescope_diameter=D_atm,
        frame_rate=float(frame_rate),
        r0=r0,
        L0=float(L0),
        l0=float(l0),
        wind_speed=wind_speed,
        fractional_r0=fractional_r0,
        altitude=altitude,
        wind_direction=wind_direction,
        n_subharmonic_levels=int(atmosphere_n_subharmonic_levels),
        subharmonic_mode=str(atmosphere_subharmonic_mode),
        direction_convention=str(atmosphere_direction_convention),
        normalize_fractional_r0=bool(atmosphere_normalize_fractional_r0),
        remove_piston=bool(atmosphere_remove_piston),
        store_components=False,
        device=device,
        dtype=precision.real,
        seed=int(seed),
        wavelength=wavelength,
        r0_reference_wavelength=float(atmosphere_r0_reference_wavelength),
        propagation_mode=str(atmosphere_propagation_mode),
        asm_extra_pixels=atmosphere_asm_extra_pixels,
        asm_min_physical_margin=float(atmosphere_asm_min_physical_margin),
        asm_padding_factor=float(atmosphere_asm_padding_factor),
        delta_mode=str(atmosphere_delta_mode),
        delta_wrap_warning_threshold=float(atmosphere_delta_wrap_warning_threshold),
        temporal_reanchor_interval=int(atmosphere_temporal_reanchor_interval),
        frozen_flow_mode=str(atmosphere_frozen_flow_mode),
    )

#    if atmosphere_warmup:
#        atm.warmup(n_updates=int(atmosphere_warmup_updates))

    if atmosphere_sequence_storage not in ("device", "cpu"):
        raise ValueError("atmosphere_sequence_storage debe ser 'device' o 'cpu'.")

    sequence_device = device if atmosphere_sequence_storage == "device" else "cpu"

    phase_all = torch.empty(
        (n_samples, 1, resolution, resolution),
        dtype=precision.real,
        device=sequence_device,
    )

    store_amplitude = bool(atmosphere_scintillation)
    amplitude_all = None
    if store_amplitude:
        amplitude_all = torch.empty_like(phase_all)

    # gen() is frame zero. With r0=[min,max] and batch_size=1 this draws one
    # random r0 that remains fixed for the entire evolving sequence.
    phase = atm.gen(seed=int(seed))

    pbar = tqdm(range(n_samples), desc="Generating ATMOSPHERE sequence", leave=True)
    for n in pbar:
        if n > 0:
            phase = atm.update()

        # Atmosphere public shape is [1,B,H,W]. B=1 in this test.
        phase_frame = phase[0, 0].detach()
        if phase_all.device != phase_frame.device:
            phase_frame = phase_frame.to(phase_all.device)
        phase_all[n, 0].copy_(phase_frame)

        if amplitude_all is not None:
            amplitude_frame = atm.field.abs()[0, 0].detach()
            if amplitude_all.device != amplitude_frame.device:
                amplitude_frame = amplitude_frame.to(amplitude_all.device)
            amplitude_all[n, 0].copy_(amplitude_frame)

        if n == 0 or (n + 1) % max(1, n_samples // 20) == 0:
            pbar.set_postfix(
                r0=f"{float(atm.r0_batch[0].item()):.5f}",
                frame=atm.frame_index,
            )

    sampled_r0 = atm.r0_batch.detach().cpu().clone()
    info = atm.info()
    info["sampled_r0_batch_m"] = sampled_r0
    info["sequence_frames"] = int(n_samples)
    info["scintillation_used_by_closed_loop"] = bool(atmosphere_scintillation)
    info["sequence_storage_device"] = str(phase_all.device)

    sequence = AtmosphereSequence(
        phase=phase_all,
        amplitude=amplitude_all,
        sampled_r0=sampled_r0,
        info=info,
    )

    if save_atmosphere_sequence_pt:
        torch.save(
            {
                "phase": phase_all.detach().cpu(),
                "amplitude": None if amplitude_all is None else amplitude_all.detach().cpu(),
                "sampled_r0": sampled_r0,
                "atmosphere_info": info,
            },
            os.path.join(output_dir, "atmosphere_sequence_torch.pt"),
        )

    return sequence


def get_sequence_frame(
    sequence: AtmosphereSequence,
    index: int,
    *,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return phase/amplitude frame as [1,1,H,W] on the compute device."""
    phase = sequence.phase[index:index + 1].to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )

    amplitude = None
    if sequence.amplitude is not None:
        amplitude = sequence.amplitude[index:index + 1].to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    return phase, amplitude


# =========================================================
# MODEL-SPECIFIC BASIS HANDLING
# =========================================================
# Each model can now use its own reconstruction/control basis.
# This is necessary when comparing networks trained with different
# bases or a different number of output modes.
#
# Supported cases:
#   1) atmosphere_cfg["dm_basis"] == True
#      Loads the exact DM basis associated with that model/training run.
#      You may optionally override the path in MODELS_TO_TEST using:
#           "basis_path": "/path/to/basis.pt"
#      or:
#           "dm_basis_path": "/path/to/basis.pt"
#
#   2) atmosphere_cfg["dm_basis"] == False
#      Builds a standard Zernike basis for that model using its own
#      atmosphere_cfg["n_modes"].
#
# The ideal residual is computed with the same basis used by the model.
# Therefore, a model trained with 70 Zernike modes is compared against
# the 70-mode ideal, and a model trained with an ACTUATOR DM basis is
# compared against the ideal projection on that ACTUATOR basis.
# =========================================================

@dataclass
class ModelBasisBundle:
    telescope_pupil: torch.Tensor
    zDecomposeMat: torch.Tensor
    zComposeMat: torch.Tensor
    n_modes: int
    basis_kind: str
    dm_basis: bool
    dm_basis_type: Optional[str] = None
    dm_name: Optional[str] = None
    basis_path: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


def _as_4d_pupil(pupil: torch.Tensor, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Ensures pupil has shape (1,1,H,W), on the requested device and dtype.
    """
    if not torch.is_tensor(pupil):
        pupil = torch.as_tensor(pupil)

    pupil = pupil.to(device=device, dtype=dtype)

    if pupil.ndim == 2:
        pupil = pupil[None, None, ...]
    elif pupil.ndim == 3:
        pupil = pupil[None, ...]
    elif pupil.ndim == 4:
        pass
    else:
        raise ValueError(f"telescope_pupil debe tener 2D, 3D o 4D. Llegó {tuple(pupil.shape)}")

    if pupil.shape[0] != 1 or pupil.shape[1] != 1:
        raise ValueError(f"telescope_pupil debe quedar como (1,1,H,W). Llegó {tuple(pupil.shape)}")

    return pupil


def _get_model_n_modes(model_info: Dict[str, Any], bundle: Dict[str, Any]) -> int:
    """
    Priority:
      1) explicit override in MODELS_TO_TEST[...]["n_modes"]
      2) training atmosphere_cfg["n_modes"]
    """
    if "n_modes" in model_info and model_info["n_modes"] is not None:
        return int(model_info["n_modes"])

    atmosphere_cfg = bundle["atmosphere_cfg"]
    if "n_modes" not in atmosphere_cfg:
        raise KeyError(
            f"El modelo {model_info.get('Name', '<unknown>')} no tiene atmosphere_cfg['n_modes']. "
            "Agrega 'n_modes' en MODELS_TO_TEST para este modelo."
        )
    return int(atmosphere_cfg["n_modes"])


def _resolve_dm_basis_path(model_info: Dict[str, Any], bundle: Dict[str, Any]) -> str:
    """
    Resolves the path of the DM basis used by this model.

    Recommended for maximum safety:
        MODELS_TO_TEST = [{..., "basis_path": "/exact/path/to/basis.pt"}]

    If no explicit path is given, it recreates the convention used by your
    dataset folders.
    """
    for key in ("basis_path", "dm_basis_path"):
        if key in model_info and model_info[key] is not None:
            return str(model_info[key])

    atmosphere_cfg = bundle["atmosphere_cfg"]
    telescope_cfg = bundle["telescope_cfg"]

    dm_basis_type = atmosphere_cfg.get("dm_basis_type", None)
    dm_name = atmosphere_cfg.get("dm_name", None)

    if dm_basis_type is None or dm_name is None:
        raise KeyError(
            f"El modelo {model_info.get('Name', '<unknown>')} tiene dm_basis=True, "
            "pero falta atmosphere_cfg['dm_basis_type'] o atmosphere_cfg['dm_name']. "
            "Puedes resolverlo agregando 'basis_path' en MODELS_TO_TEST."
        )

    resolution = int(telescope_cfg["resolution"])
    return (
        "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS_EVOLVE/"
        f"DATASET/DEFORMABLE_MIRROR_BASIS/{dm_name}/"
        f"{dm_basis_type}_BASIS_RES_{resolution}.pt"
    )


def _slice_compose_matrix(zComposeMat: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    zComposeMat is expected to be mode-major, usually (n_modes, H*W) or
    compatible with zernike_compose_torch(z, zComposeMat). We slice the
    first dimension because that is the modal dimension in your current code.
    """
    if zComposeMat.ndim < 2:
        raise ValueError(f"zComposeMat debe tener al menos 2 dimensiones. Llegó {tuple(zComposeMat.shape)}")

    if zComposeMat.shape[0] < n_modes:
        raise ValueError(
            f"zComposeMat no tiene suficientes modos: shape={tuple(zComposeMat.shape)}, n_modes={n_modes}"
        )

    return zComposeMat[:n_modes, ...]


def _slice_decompose_matrix(zDecomposeMat: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    zDecomposeMat is expected to be coefficient-last, usually (H*W, n_modes)
    or compatible with zernike_decompose_torch(phi, zDecomposeMat). We slice
    the last dimension because that is the modal dimension in your current code.
    """
    if zDecomposeMat.ndim < 2:
        raise ValueError(f"zDecomposeMat debe tener al menos 2 dimensiones. Llegó {tuple(zDecomposeMat.shape)}")

    if zDecomposeMat.shape[-1] < n_modes:
        raise ValueError(
            f"zDecomposeMat no tiene suficientes modos: shape={tuple(zDecomposeMat.shape)}, n_modes={n_modes}"
        )

    return zDecomposeMat[..., :n_modes]


def build_model_basis_bundle(
    model_info: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> ModelBasisBundle:

    Name = model_info.get("Name", "<unknown>")

    atmosphere_cfg = bundle["atmosphere_cfg"]
    telescope_cfg = bundle["telescope_cfg"]
    WFS = bundle["WFS"]

    resolution = int(
        telescope_cfg["resolution"]
    )

    diameter = float(
        telescope_cfg["diameter"]
    )

    n_modes = _get_model_n_modes(
        model_info,
        bundle,
    )

    # =========================================================
    # Pupila física EXACTA usada durante el entrenamiento
    # =========================================================

    if not hasattr(WFS, "telescope_pupil"):
        raise AttributeError(
            "El WFS cargado no contiene telescope_pupil."
        )

    trained_telescope_pupil = _as_4d_pupil(
        WFS.telescope_pupil,
        device=device,
        dtype=dtype,
    )

    if (
        trained_telescope_pupil.shape[-2] != resolution
        or trained_telescope_pupil.shape[-1] != resolution
    ):
        raise ValueError(
            "WFS.telescope_pupil no coincide con "
            "telescope_cfg['resolution']."
        )

    dm_basis = bool(
        atmosphere_cfg.get(
            "dm_basis",
            False,
        )
    )
    if dm_basis:
        basis_path = _resolve_dm_basis_path(model_info, bundle)

        if not os.path.exists(basis_path):
            raise FileNotFoundError(
                f"No se encontró la base DM para el modelo {Name}:\n{basis_path}\n"
                "Solución recomendada: agrega 'basis_path' explícito en MODELS_TO_TEST para este modelo."
            )

        dm_data = torch.load(basis_path, map_location="cpu")

        required_keys = ["zComposeMat", "zDecomposeMat", "telescope_pupil"]
        missing = [k for k in required_keys if k not in dm_data]
        if len(missing) > 0:
            raise KeyError(f"La base {basis_path} no contiene las llaves requeridas: {missing}")

        zComposeMat = dm_data[
            "zComposeMat"
        ]

        zDecomposeMat = dm_data[
            "zDecomposeMat"
        ]

        # Pupila ORIGINAL almacenada con la base DM.
        # Solo se usa para comprobar compatibilidad.
        dm_pupil = _as_4d_pupil(
            dm_data["telescope_pupil"],
            device=device,
            dtype=dtype,
        )

        if (
            dm_pupil.shape[-2] != resolution
            or dm_pupil.shape[-1] != resolution
        ):
            raise ValueError(
                f"La pupil de la base DM de {Name} "
                "no coincide con telescope_cfg['resolution']. "
                f"pupil={tuple(dm_pupil.shape)}, "
                f"resolution={resolution}"
            )

        # Las matrices del DM se mantienen EXACTAMENTE
        # como fueron guardadas.
        zComposeMat = zComposeMat.to(
            device=device,
            dtype=dtype,
        )

        zDecomposeMat = zDecomposeMat.to(
            device=device,
            dtype=dtype,
        )
        dm_basis_type = atmosphere_cfg.get("dm_basis_type", "UNKNOWN")
        dm_name = atmosphere_cfg.get("dm_name", "UNKNOWN")
        basis_kind = f"DM_{dm_basis_type}"

        return ModelBasisBundle(
            telescope_pupil=trained_telescope_pupil,
            zDecomposeMat=zDecomposeMat,
            zComposeMat=zComposeMat,
            n_modes=n_modes,
            basis_kind=basis_kind,
            dm_basis=True,
            dm_basis_type=dm_basis_type,
            dm_name=dm_name,
            basis_path=basis_path,
            info={
                "Name": Name,
                "resolution": resolution,
                "diameter": diameter,
                "n_modes": n_modes,
                "zComposeMat_shape": tuple(zComposeMat.shape),
                "zDecomposeMat_shape": tuple(zDecomposeMat.shape),
                "dm_pupil_shape": tuple(
                    dm_pupil.shape
                ),
                "telescope_pupil_shape": tuple(
                    trained_telescope_pupil.shape
                ),
            },
        )

    # ============================================================
    # STANDARD ZERNIKE BASIS
    #
    # Los Zernikes se generan primero sobre una pupila circular
    # ideal y después se restringen a la pupila física utilizada
    # durante el entrenamiento.
    # ============================================================

    ideal_pupil = circular_pupil(
        n=resolution,
        device=device,
        dtype=dtype,
        soft_edge_px=0.0,
    )

    (
        zDecomposeMat,
        zComposeMat,
    ) = get_zernike_on_pupil(
        ideal_pupil=(
            ideal_pupil
            .squeeze()
            .detach()
            .cpu()
        ),

        physical_pupil=(
            trained_telescope_pupil
            .squeeze()
            .detach()
            .cpu()
        ),

        diameter=diameter,
        nModes=n_modes,
        type="torch",
    )

    zDecomposeMat = zDecomposeMat.to(
        device=device,
        dtype=dtype,
    )

    zComposeMat = zComposeMat.to(
        device=device,
        dtype=dtype,
    )

    return ModelBasisBundle(
        telescope_pupil=(
            trained_telescope_pupil
        ),

        zDecomposeMat=zDecomposeMat,
        zComposeMat=zComposeMat,

        n_modes=n_modes,

        basis_kind="ZERNIKE",

        dm_basis=False,
        dm_basis_type=None,
        dm_name=None,
        basis_path=None,

        info={
            "Name": Name,
            "resolution": resolution,
            "diameter": diameter,
            "n_modes": n_modes,

            "zComposeMat_shape": tuple(
                zComposeMat.shape
            ),

            "zDecomposeMat_shape": tuple(
                zDecomposeMat.shape
            ),

            "telescope_pupil_shape": tuple(
                trained_telescope_pupil.shape
            ),
        },
    )

def validate_shared_model_compatibility(
    model_info: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    ref_resolution: int,
    ref_diameter: float,
    ref_precision_name: str,
) -> None:
    """
    Only validates what must be common to generate one shared atmosphere.
    n_modes and basis are intentionally NOT checked here because they can
    differ per model.
    """
    Name = model_info.get("Name", "<unknown>")
    telescope_cfg = bundle["telescope_cfg"]
    train_cfg = bundle["train_cfg"]

    if int(telescope_cfg["resolution"]) != int(ref_resolution):
        raise ValueError(
            f"El modelo {Name} no tiene la misma resolution. "
            f"Esperada {ref_resolution}, recibida {telescope_cfg['resolution']}."
        )

    if float(telescope_cfg["diameter"]) != float(ref_diameter):
        raise ValueError(
            f"El modelo {Name} no tiene el mismo telescope diameter. "
            f"Esperado {ref_diameter}, recibido {telescope_cfg['diameter']}."
        )

    if train_cfg["precision"] != ref_precision_name:
        raise ValueError(
            f"El modelo {Name} no tiene la misma precision. "
            f"Esperada {ref_precision_name}, recibida {train_cfg['precision']}."
        )


def get_model_reference_psf_peak(model_basis: ModelBasisBundle, resolution: int, device: str, dtype: torch.dtype) -> float:
    """
    Reference diffraction-limited PSF peak for this model's pupil.
    If pupils differ between bases, Strehl must use each model's own reference peak.
    """
    phi_ref = torch.zeros((1, 1, resolution, resolution), device=device, dtype=dtype)
    I_psf_ref = get_psf(
        telescope_pupil=model_basis.telescope_pupil,
        phi=phi_ref,
        fovPx=resolution * 4,
    )
    return float(I_psf_ref.max().item())


def assert_network_output_matches_basis(z_test: torch.Tensor, model_basis: ModelBasisBundle, model_name: str) -> None:
    """
    Prevents silent errors when the NN output size and the selected basis size
    are inconsistent.
    """
    if z_test.ndim != 2:
        raise ValueError(f"La salida de la red de {model_name} debe ser 2D (B,n_modes). Llegó {tuple(z_test.shape)}")

    nn_modes = int(z_test.shape[-1])
    if nn_modes != int(model_basis.n_modes):
        raise ValueError(
            f"Mismatch entre la salida de la red y la base del modelo {model_name}: "
            f"NN output={nn_modes}, basis n_modes={model_basis.n_modes}. "
            "Revisa atmosphere_cfg['n_modes'] o agrega 'n_modes' explícito en MODELS_TO_TEST."
        )



@torch.no_grad()
def remove_piston(phi, pupil):
    mask = (pupil > 0).to(phi.dtype)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    piston = (phi * mask).sum(dim=(-2, -1), keepdim=True) / denom
    return (phi - piston) * mask


@torch.no_grad()
def get_wfe_rms(phi, pupil):
    phi0 = remove_piston(phi, pupil)
    mask = (pupil > 0).to(phi0.dtype)
    denom = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    rms = torch.sqrt((phi0.pow(2) * mask).sum(dim=(-2, -1)) / denom)
    return rms.squeeze(1).cpu()


@torch.no_grad()
def get_strehl_from_psf(I_psf, I_psf_ref_peak):
    peak = I_psf.amax(dim=(-2, -1)).squeeze(1)
    strehl = peak / max(float(I_psf_ref_peak), 1e-12)
    return strehl.cpu()


def _prepare_vis_tensor(x, use_log=False, range_min_max=False):
    x = x.detach()

    if use_log:
        x = torch.log10(x.clamp_min(1e-12))

    x = x.squeeze(1)

    if range_min_max is False:
        vmin = float(x.min().item())
        vmax = float(x.max().item())
    else:
        vmin = None
        vmax = None

    return x.detach().cpu().numpy().astype(np.float32), vmin, vmax


def _get_gif_frame_indices(n_frames: int, max_frames: int = 500) -> np.ndarray:
    """
    Returns uniformly distributed frame indices for GIF export.

    If n_frames <= max_frames, every frame is kept.
    If n_frames > max_frames, exactly max_frames are selected across the
    complete sequence, including the first and last frame.
    """
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)

    return np.linspace(0, n_frames - 1, max_frames, dtype=np.int64)


def save_transition_gif_fast(
    tensor_all,
    loop_closed_flag,
    out_path,
    cmap="hot",
    fps=15,
    use_log=False,
    range_min_max=False,
    visual_gamma=1.0,
    log_dynamic_range_db=None,
):
    gif_indices = _get_gif_frame_indices(int(tensor_all.shape[0]), max_frames=500)

    # Keep the original global normalization when requested, but only move
    # the selected GIF frames to CPU/NumPy.
    if range_min_max is False:
        x_range = tensor_all.detach()
        if use_log:
            x_range = torch.log10(x_range.clamp_min(1e-12))
        vmin = float(x_range.min().item())
        vmax = float(x_range.max().item())
    else:
        vmin = None
        vmax = None

    # Normal visualization path remains unchanged.
    # For PSF GIFs, log_dynamic_range_db can instead express each frame relative
    # to its own peak in dB. This uses the full colormap without driving most
    # visible structure into the yellow/white end of ``hot``.
    x_vis, _, _ = _prepare_vis_tensor(
        tensor_all[gif_indices],
        use_log=(use_log and log_dynamic_range_db is None),
        range_min_max=True,
    )

    cmap_obj = plt.get_cmap(cmap)
    frames = []

    for k in tqdm(range(x_vis.shape[0]), desc=f"Saving GIF: {os.path.basename(out_path)}", leave=False):
        img = x_vis[k]

        if log_dynamic_range_db is not None:
            dynamic_range_db = float(log_dynamic_range_db)
            if dynamic_range_db <= 0:
                raise ValueError("log_dynamic_range_db debe ser > 0")

            # Intensity PSF in dB relative to the peak of THIS frame:
            # peak -> 0 dB -> top of colormap
            # <= -dynamic_range_db -> bottom of colormap
            peak = max(float(np.max(img)), 1e-30)
            rel = np.clip(img / peak, 1e-30, None)
            img_db = 10.0 * np.log10(rel)
            img_db = np.clip(img_db, -dynamic_range_db, 0.0)
            norm = (img_db + dynamic_range_db) / dynamic_range_db
            norm = norm.astype(np.float32)
        else:
            if vmin is None or vmax is None:
                mn = np.min(img)
                mx = np.max(img)
            else:
                mn = vmin
                mx = vmax

            if np.isclose(mx, mn):
                norm = np.zeros_like(img, dtype=np.float32)
            else:
                norm = np.clip((img - mn) / (mx - mn), 0.0, 1.0)

        # Purely visual tone adjustment applied AFTER the original normalization.
        # visual_gamma > 1 darkens mid/high normalized intensities, reducing the
        # saturated yellow/white area while preserving the full PSF structure.
        # The physical PSF tensor and every metric remain untouched.
        if visual_gamma <= 0:
            raise ValueError("visual_gamma debe ser > 0")
        if not np.isclose(visual_gamma, 1.0):
            norm = np.power(norm, float(visual_gamma)).astype(np.float32)

        rgb = cmap_obj(norm)[..., :3]
        rgb = (255.0 * rgb).astype(np.uint8)
        frames.append(rgb)

    imageio.mimsave(out_path, frames, loop=0)




def _to_numpy_2d_for_image(x, use_log=False):
    """
    Converts a tensor/image-like object to a 2D numpy array for saving.
    Accepts shapes like (H,W), (1,H,W), (1,1,H,W).
    """
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if use_log:
            x = torch.log10(x.clamp_min(1e-12))
        x = x.squeeze()
        x = x.numpy()
    else:
        x = np.asarray(x)
        if use_log:
            x = np.log10(np.clip(x, 1e-12, None))
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Se esperaba imagen 2D después de squeeze. Llegó {x.shape}")

    return x.astype(np.float32)


def save_single_image(
    img,
    out_path,
    cmap="hot",
    title=None,
    use_log=False,
    range_min_max=True,
):
    """
    Saves a single 2D image as PNG.
    """
    x = _to_numpy_2d_for_image(img, use_log=use_log)

    plt.figure(figsize=(6, 5))
    if range_min_max:
        im = plt.imshow(x, cmap=cmap)
    else:
        im = plt.imshow(x, cmap=cmap, vmin=float(np.min(x)), vmax=float(np.max(x)))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def get_four_channel_order_from_wfs(WFS):
    """
    Returns the display order for a 4-channel pyramid crop.

    The crop_pyr implementation, outside the special resolution==32 branch,
    fills channels following self.coords. To display the 4 pupils correctly,
    we sort the crop centers spatially as:

        top-left, top-right,
        bottom-left, bottom-right

    If coords are not available or are not 4 points, it falls back to [0,1,2,3].
    """
    try:
        coords = getattr(WFS, "coords", None)
        if coords is None:
            return [0, 1, 2, 3]

        if torch.is_tensor(coords):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)

        coords_np = np.asarray(coords_np, dtype=np.float64)
        if coords_np.ndim != 2 or coords_np.shape[0] != 4 or coords_np.shape[1] < 2:
            return [0, 1, 2, 3]

        # coords[:,0] = x, coords[:,1] = y according to crop_pyr.
        idx = np.arange(4)
        top_two = idx[np.argsort(coords_np[:, 1])[:2]]
        bottom_two = idx[np.argsort(coords_np[:, 1])[2:]]

        top_two = top_two[np.argsort(coords_np[top_two, 0])]
        bottom_two = bottom_two[np.argsort(coords_np[bottom_two, 0])]

        return [int(top_two[0]), int(top_two[1]), int(bottom_two[0]), int(bottom_two[1])]

    except Exception:
        return [0, 1, 2, 3]


def make_4channel_mosaic(frame_4ch, channel_order=None):
    """
    frame_4ch: (4,H,W). Returns a 2x2 mosaic.

    channel_order:
        [top_left, top_right, bottom_left, bottom_right]
    """
    if torch.is_tensor(frame_4ch):
        x = frame_4ch.detach().cpu().numpy()
    else:
        x = np.asarray(frame_4ch)

    if x.ndim != 3 or x.shape[0] != 4:
        raise ValueError(f"Se esperaba (4,H,W). Llegó {x.shape}")

    if channel_order is None:
        channel_order = [0, 1, 2, 3]

    x = x[np.asarray(channel_order, dtype=np.int64)]

    top = np.concatenate([x[0], x[1]], axis=1)
    bottom = np.concatenate([x[2], x[3]], axis=1)
    return np.concatenate([top, bottom], axis=0).astype(np.float32)


def save_network_input_4ch_gif(
    tensor_all,
    out_path,
    cmap="hot",
    fps=15,
    range_min_max=True,
    channel_order=None,
):
    """
    Saves the NN input when it has 4 channels.

    tensor_all:
        (T,4,H,W), already normalized exactly as fed to the NN.
    """
    if tensor_all.ndim != 4 or tensor_all.shape[1] != 4:
        raise ValueError(f"Se esperaba (T,4,H,W). Llegó {tuple(tensor_all.shape)}")

    gif_indices = _get_gif_frame_indices(int(tensor_all.shape[0]), max_frames=500)

    if range_min_max:
        vmin = None
        vmax = None
    else:
        vmin = float(tensor_all.detach().min().item())
        vmax = float(tensor_all.detach().max().item())

    # Only the frames that will actually enter the GIF are copied to CPU.
    x = tensor_all[gif_indices].detach().cpu().float()

    cmap_obj = plt.get_cmap(cmap)
    frames = []

    for k in tqdm(range(x.shape[0]), desc=f"Saving GIF: {os.path.basename(out_path)}", leave=False):
        img = make_4channel_mosaic(x[k], channel_order=channel_order)

        if vmin is None or vmax is None:
            mn = np.min(img)
            mx = np.max(img)
        else:
            mn = vmin
            mx = vmax

        if np.isclose(mx, mn):
            norm = np.zeros_like(img, dtype=np.float32)
        else:
            norm = np.clip((img - mn) / (mx - mn), 0.0, 1.0)

        rgb = cmap_obj(norm)[..., :3]
        rgb = (255.0 * rgb).astype(np.uint8)
        frames.append(rgb)

    imageio.mimsave(out_path, frames, loop=0)


def save_z_estimation_heatmap(z_est_all, out_path, cl_sample=0, Name=""):
    """
    Saves a heatmap of the modal estimation sent by the NN.

    z_est_all:
        (T,n_modes). Values before the loop closes can still be useful
        because they correspond to open-loop input estimation.
    """
    z = z_est_all.detach().cpu().float().numpy()

    plt.figure(figsize=(13, 6))
    im = plt.imshow(z.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.axvline(cl_sample - 1, linestyle="--", linewidth=1.2, color="w", alpha=0.9, label="Loop closes")
    plt.xlabel("Sample")
    plt.ylabel("Estimated Zernike mode index")
    plt.title(f"{Name} | NN modal estimation")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Estimated coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

def save_metric_plot(y_open, y_closed, y_ideal, ylabel, out_path, Name="", cl_sample=0):
    x = np.arange(len(y_open))

    y_open   = y_open.detach().cpu().numpy()
    y_closed = y_closed.detach().cpu().numpy()
    y_ideal  = y_ideal.detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(x, y_open,   label="Open-loop residual", linewidth=1.8)
    plt.plot(x, y_closed, label="Closed-loop residual", linewidth=1.8)
    plt.plot(x, y_ideal,  label="Ideal residual", linewidth=1.8)
    plt.axvline(cl_sample-1, linestyle="--", linewidth=1.2, color="k", alpha=0.7, label="Loop closes")
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.title(f"{Name} | {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_comparative_plot(y_open, y_ideal, model_curves_dict, ylabel, out_path, cl_sample=0):
    x = np.arange(len(y_open))

    y_open  = y_open.detach().cpu().numpy()
    y_ideal = y_ideal.detach().cpu().numpy()

    plt.figure(figsize=(13, 6))
    plt.plot(x, y_open,  label="Open-loop residual", linewidth=2.2)
    plt.plot(x, y_ideal, label="Ideal residual", linewidth=2.2)

    for model_name, y_model in model_curves_dict.items():
        y_model = y_model.detach().cpu().numpy()
        plt.plot(x, y_model, label=model_name, linewidth=1.8)

    plt.axvline(cl_sample, linestyle="--", linewidth=1.2, color="k", alpha=0.7, label="Loop closes")
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.title(f"Model comparison | {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()



def save_comparative_plot_per_model_ideal(
    y_open,
    model_closed_curves_dict,
    model_ideal_curves_dict,
    ylabel,
    out_path,
    cl_sample=0,
):
    """
    Comparative plot where each model can have its own ideal curve because
    each model can use a different basis and a different number of modes.
    """
    x = np.arange(len(y_open))
    y_open = y_open.detach().cpu().numpy()

    plt.figure(figsize=(13, 6))
    plt.plot(x, y_open, label="Open-loop residual", linewidth=2.2)

    for model_name, y_ideal in model_ideal_curves_dict.items():
        y_ideal = y_ideal.detach().cpu().numpy()
        plt.plot(x, y_ideal, linestyle="--", linewidth=1.5, label=f"{model_name} ideal")

    for model_name, y_model in model_closed_curves_dict.items():
        y_model = y_model.detach().cpu().numpy()
        plt.plot(x, y_model, linewidth=1.8, label=f"{model_name} closed loop")

    plt.axvline(cl_sample, linestyle="--", linewidth=1.2, color="k", alpha=0.7, label="Loop closes")
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.title(f"Model comparison | {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def get_psf(telescope_pupil=None, phi=None, fovPx=None):
    phi_phasor = telescope_pupil * torch.exp(1j * phi)
    phi_phasor_padded = pad2size(phi_phasor, (fovPx, fovPx))
    psf = torch.fft.fftshift(torch.fft.fft2(phi_phasor_padded), dim=(-2, -1))
    I_psf = torch.abs(psf) ** 2
    return I_psf.cpu()


def build_frame_optical_pupil(
    telescope_pupil: torch.Tensor,
    atmospheric_amplitude: Optional[torch.Tensor],
) -> torch.Tensor:
    """Amplitude presented to the WFS/PSF for the current atmospheric frame."""
    if atmospheric_amplitude is None:
        return telescope_pupil
    return telescope_pupil * atmospheric_amplitude


def build_forward_pipe(WFS, NN, noise_pipe, norm_type, noise_flag):
    def forward_pipe(propagation_pupil=None, phi=None, return_net_input=False):
        # IMPORTANTE: propagation_pupil afecta únicamente la formación de la
        # imagen del WFS que ve la red. No se usa para métricas ni bases ideales.
        I_full, I_crop = WFS.propagate(pupil=propagation_pupil, phi=phi, return_both=True)

        if noise_flag:
            I_full = noise_pipe(I_full)
            I_crop = noise_pipe(I_crop)

        I_full = norm_I(I_full, norm=norm_type)
        I_crop = norm_I(I_crop, norm=norm_type)

        with torch.no_grad():
            zEst = NN(I_crop).detach()

        if return_net_input:
            # I_crop is exactly the normalized tensor entering the NN.
            return zEst.cpu(), I_full.cpu(), I_crop.detach().cpu()

        return zEst.cpu(), I_full.cpu()

    return forward_pipe


def clamp_integrator(x, limit):
    if limit is None:
        return x
    return torch.clamp(x, min=-limit, max=limit)


def run_closed_loop_single_model(
    model_info,
    bundle,
    model_basis: ModelBasisBundle,
    atmosphere_sequence: AtmosphereSequence,
    output_model_test_path,
    I_psf_ref_peak,
    noise_pipe,
    device,
    shared_precision,
    save_artifacts=True,
    current_noise_level="normal",
    propagation_pupil_enabled=True,
    propagation_obstruction_px=0.0,
    propagation_spider_count=0,
    propagation_spider_width_px=0.0,
    propagation_pupil_random_seed=None,
):
    Name = model_info["Name"]
    train_cfg = bundle["train_cfg"]
    WFS = bundle["WFS"]
    NN = bundle["NN"]
    telescope_cfg = bundle["telescope_cfg"]

    # Conservamos intactas las matrices de la base cargada. La máscara física
    # con obstrucción y spiders se construye una sola vez y se usa en TODO el
    # flujo óptico. Esto no añade pseudoinversas ni cálculos costosos al loop.
    base_pupil = _as_4d_pupil(
        model_basis.telescope_pupil,
        device=device,
        dtype=shared_precision.real,
    )
    telescope_pupil = build_wfs_propagation_pupil(
        ideal_pupil=base_pupil,
        enabled=propagation_pupil_enabled,
        central_obstruction_diam_px=propagation_obstruction_px,
        spiders=propagation_spider_count,
        spiders_px=propagation_spider_width_px,
        soft_edge_px=propagation_soft_edge_px,
        seed=propagation_pupil_random_seed,
    )
    propagation_pupil = telescope_pupil

    # Strehl de referencia correspondiente exactamente a la pupila física.
    phi_ref = torch.zeros(
        (1, 1, int(telescope_cfg["resolution"]), int(telescope_cfg["resolution"])),
        device=device,
        dtype=shared_precision.real,
    )
    I_psf_ref_peak = float(
        get_psf(
            telescope_pupil=telescope_pupil,
            phi=phi_ref,
            fovPx=int(telescope_cfg["resolution"]) * 4,
        ).max().item()
    )

    zDecomposeMat = model_basis.zDecomposeMat
    zComposeMat = model_basis.zComposeMat
    n_modes = int(model_basis.n_modes)

    # zernike_decompose_torch aplana phi a H*W y realiza
    # phi_flat @ zDecomposeMat.T. No recortar ni transponer esta matriz.
    expected_pixels = int(telescope_cfg["resolution"]) ** 2
    if zDecomposeMat.ndim != 2 or int(zDecomposeMat.shape[-1]) != expected_pixels:
        raise ValueError(
            f"Base incompatible para {Name}: zDecomposeMat debe tener shape "
            f"(n_modes, {expected_pixels}) y llegó {tuple(zDecomposeMat.shape)}. "
            "La matriz se conserva sin modificar; revisa el archivo de base cargado."
        )
    if int(zDecomposeMat.shape[0]) != n_modes:
        raise ValueError(
            f"Base incompatible para {Name}: zDecomposeMat contiene "
            f"{zDecomposeMat.shape[0]} modos y se esperaban {n_modes}."
        )

    os.makedirs(output_model_test_path, exist_ok=True)

    forward_pipe = build_forward_pipe(
        WFS=WFS,
        NN=NN,
        noise_pipe=noise_pipe,
        norm_type=train_cfg["norm_type"],
        noise_flag=noise_flag,
    )

    H = int(telescope_cfg["resolution"])
    W = int(telescope_cfg["resolution"])
    fovPx = int(telescope_cfg["resolution"]) * 4

    with torch.no_grad():
        phi_test_raw, amp_test = get_sequence_frame(
            atmosphere_sequence,
            0,
            device=device,
            dtype=shared_precision.real,
        )
        phi_test = phi_test_raw * telescope_pupil
        propagation_pupil_test = build_frame_optical_pupil(
            telescope_pupil,
            amp_test,
        )
        z_test, I_full_test, net_input_test = forward_pipe(
            propagation_pupil=propagation_pupil_test,
            phi=phi_test,
            return_net_input=True,
        )

    assert_network_output_matches_basis(z_test, model_basis, Name)

    IpropH = I_full_test.shape[-2]
    IpropW = I_full_test.shape[-1]

    save_network_input_flag = (net_input_test.ndim == 4 and net_input_test.shape[1] == 4)
    network_channel_order = get_four_channel_order_from_wfs(WFS) if save_network_input_flag else None

    psf_transition_all = torch.empty((n_samples, 1, fovPx, fovPx), dtype=shared_precision.real)
    phi_transition_all = torch.empty((n_samples, 1, H, W), dtype=shared_precision.real)
    I_transition_all   = torch.empty((n_samples, 1, IpropH, IpropW), dtype=shared_precision.real)
    amplitude_transition_all = None
    if atmosphere_sequence.amplitude is not None:
        amplitude_transition_all = torch.empty((n_samples, 1, H, W), dtype=shared_precision.real)

    # Stores exactly what enters the NN after noise + norm, only when C=4.
    # Shape: (T,4,Hcrop,Wcrop).
    net_input_transition_all = None
    if save_network_input_flag:
        net_input_transition_all = torch.empty(
            (n_samples, net_input_test.shape[1], net_input_test.shape[2], net_input_test.shape[3]),
            dtype=shared_precision.real,
        )

    # Stores the NN modal estimation associated with the network input.
    # This can differ per model because each model can have its own n_modes.
    z_est_transition_all = torch.empty((n_samples, n_modes), dtype=shared_precision.real)

    wfe_open   = torch.empty(n_samples, dtype=shared_precision.real)
    wfe_closed = torch.empty(n_samples, dtype=shared_precision.real)
    wfe_ideal  = torch.empty(n_samples, dtype=shared_precision.real)

    strehl_open   = torch.empty(n_samples, dtype=shared_precision.real)
    strehl_closed = torch.empty(n_samples, dtype=shared_precision.real)
    strehl_ideal  = torch.empty(n_samples, dtype=shared_precision.real)

    loop_closed_flag = torch.zeros(n_samples, dtype=torch.bool)

    # PI integrator state in the same coordinates as this model's training basis.
    u_integral = torch.zeros((1, n_modes), dtype=shared_precision.real, device=device)

    loop_pbar = tqdm(
        range(n_samples),
        desc=f"Closed loop | {Name} | {model_basis.basis_kind} | n_modes={n_modes} | noise={current_noise_level}",
        leave=True,
    )

    for n in loop_pbar:
        phi_atm_raw, atmospheric_amplitude = get_sequence_frame(
            atmosphere_sequence,
            n,
            device=device,
            dtype=shared_precision.real,
        )
        phi_atm = phi_atm_raw * telescope_pupil
        frame_propagation_pupil = build_frame_optical_pupil(
            telescope_pupil,
            atmospheric_amplitude,
        )

        # -------------------------
        # OPEN LOOP
        # -------------------------
        phi_open = remove_piston(phi_atm, telescope_pupil)
        psf_open = get_psf(telescope_pupil=frame_propagation_pupil, phi=phi_open, fovPx=fovPx)
        z_open, I_open, net_input_open = forward_pipe(
            propagation_pupil=frame_propagation_pupil,
            phi=phi_open,
            return_net_input=True,
        )

        z_open = z_open.to(dtype=shared_precision.real)
        wfe_open[n] = get_wfe_rms(phi_open.cpu(), telescope_pupil.cpu())[0]
        strehl_open[n] = get_strehl_from_psf(psf_open, I_psf_ref_peak)[0]

        # -------------------------
        # MODEL-SPECIFIC IDEAL LOOP
        # -------------------------
        # La fase ya está restringida por la pupila física con spiders.
        # Se usan las matrices originales sin recortes ni recomputación.
        z_ideal = zernike_decompose_torch(phi=phi_atm, zDecomposeMat=zDecomposeMat)

        if int(z_ideal.shape[-1]) != n_modes:
            raise ValueError(
                f"La decomposición ideal de {Name} entregó {z_ideal.shape[-1]} modos, "
                f"pero model_basis.n_modes={n_modes}. Revisa zDecomposeMat."
            )

        phi_ideal_cmd = zernike_compose_torch(
            zernike_phi_vector=z_ideal,
            zComposeMat=zComposeMat,
        )
        phi_ideal_res = remove_piston((phi_atm - phi_ideal_cmd) * telescope_pupil, telescope_pupil)
        psf_ideal = get_psf(telescope_pupil=frame_propagation_pupil, phi=phi_ideal_res, fovPx=fovPx)

        wfe_ideal[n] = get_wfe_rms(phi_ideal_res.cpu(), telescope_pupil.cpu())[0]
        strehl_ideal[n] = get_strehl_from_psf(psf_ideal, I_psf_ref_peak)[0]

        if n < cl_sample:
            phi_current = phi_open
            psf_current = psf_open
            I_current   = I_open
            z_current = z_open
            net_input_current = net_input_open

            wfe_closed[n] = wfe_open[n]
            strehl_closed[n] = strehl_open[n]
            loop_closed_flag[n] = False

        else:
            loop_closed_flag[n] = True

            # Current command applied to the system.
            u_command_current = u_integral
            phi_cmd_current = zernike_compose_torch(
                zernike_phi_vector=u_command_current,
                zComposeMat=zComposeMat,
            )

            # Residual measured by the sensor before updating the controller.
            phi_residual_before_update = remove_piston(
                (phi_atm + phi_cmd_current) * telescope_pupil,
                telescope_pupil,
            )

            z_est, _, net_input_residual = forward_pipe(
                propagation_pupil=frame_propagation_pupil,
                phi=phi_residual_before_update,
                return_net_input=True,
            )
            z_est = z_est.to(device=device, dtype=shared_precision.real)

            if int(z_est.shape[-1]) != n_modes:
                raise ValueError(
                    f"La red {Name} entregó {z_est.shape[-1]} modos, "
                    f"pero la base activa tiene {n_modes}."
                )

            z_current = z_est.detach().cpu().to(dtype=shared_precision.real)
            net_input_current = net_input_residual

            # ==========================================
            # REAL PI CONTROL IN MODEL-SPECIFIC BASIS
            # u_k = kp*e_k + I_k
            # I_k = I_{k-1} + ki*e_k
            # ==========================================
            u_integral = u_integral + loop_sign * ki * z_est

            if use_integrator_clamp:
                u_integral = clamp_integrator(u_integral, integrator_limit)

            u_command_new = u_integral + loop_sign * kp * z_est

            phi_cmd_new = zernike_compose_torch(
                zernike_phi_vector=u_command_new,
                zComposeMat=zComposeMat,
            )

            phi_current = remove_piston(
                (phi_atm + phi_cmd_new) * telescope_pupil,
                telescope_pupil,
            )

            psf_current = get_psf(telescope_pupil=frame_propagation_pupil, phi=phi_current, fovPx=fovPx)
            _, I_current = forward_pipe(propagation_pupil=frame_propagation_pupil, phi=phi_current)

            wfe_closed[n] = get_wfe_rms(phi_current.cpu(), telescope_pupil.cpu())[0]
            strehl_closed[n] = get_strehl_from_psf(psf_current, I_psf_ref_peak)[0]

        phi_transition_all[n] = phi_current.squeeze(0).cpu()
        psf_transition_all[n] = psf_current.squeeze(0)
        I_transition_all[n]   = I_current.squeeze(0).cpu()
        z_est_transition_all[n] = z_current.squeeze(0).detach().cpu()
        if amplitude_transition_all is not None and atmospheric_amplitude is not None:
            amplitude_transition_all[n] = atmospheric_amplitude.squeeze(0).detach().cpu()

        if net_input_transition_all is not None:
            # Save exactly the tensor entering the network: after noise and norm.
            # Only stored when it has 4 channels.
            if net_input_current.ndim == 4 and net_input_current.shape[1] == 4:
                net_input_transition_all[n] = net_input_current.squeeze(0).detach().cpu()

        loop_pbar.set_postfix({
            "iter": n + 1,
            "mode": "CL" if n >= cl_sample else "OL",
            "basis": model_basis.basis_kind,
            "modes": n_modes,
            "strehl": f"{strehl_closed[n].item():.4f}",
            "wfe": f"{wfe_closed[n].item():.4f}",
        })

    psf_integrated_start = min(cl_sample + 10, n_samples - 1)
    psf_integrated_closed_loop = psf_transition_all[psf_integrated_start:].mean(dim=0)

    results = {
        "psf_transition_all": psf_transition_all,
        "psf_integrated_closed_loop": psf_integrated_closed_loop,
        "psf_integrated_start": psf_integrated_start,
        "phi_transition_all": phi_transition_all,
        "I_transition_all": I_transition_all,
        "wfe_open": wfe_open,
        "wfe_closed": wfe_closed,
        "wfe_ideal": wfe_ideal,
        "strehl_open": strehl_open,
        "strehl_closed": strehl_closed,
        "strehl_ideal": strehl_ideal,
        "loop_closed_flag": loop_closed_flag,
        "z_est_transition_all": z_est_transition_all,
        "network_input_transition_all": net_input_transition_all,
        "network_channel_order": network_channel_order,
        "atmospheric_amplitude_transition_all": amplitude_transition_all,
        "atmosphere_info": atmosphere_sequence.info,
        "atmosphere_sampled_r0": atmosphere_sequence.sampled_r0,
        "cl_sample": cl_sample,
        "kp": kp,
        "ki": ki,
        "loop_sign": loop_sign,
        "model_info": model_info,
        "basis_info": model_basis.info,
        "basis_kind": model_basis.basis_kind,
        "basis_path": model_basis.basis_path,
        "n_modes": n_modes,
        "noise_level": current_noise_level,
        "propagation_pupil": propagation_pupil.detach().cpu(),
        "propagation_pupil_config": {
            "enabled": bool(propagation_pupil_enabled),
            "central_obstruction_diam_px": float(propagation_obstruction_px),
            "spiders": int(propagation_spider_count),
            "spiders_px": float(propagation_spider_width_px),
            "soft_edge_px": float(propagation_soft_edge_px),
            "seed": propagation_pupil_random_seed,
            "used_everywhere": True,
            "basis_matrices_left_unchanged": True,
        },
    }

    torch.save(results, os.path.join(output_model_test_path, "closed_loop_results.pt"))

    if save_artifacts:
        save_transition_gif_fast(
            tensor_all=psf_transition_all,
            loop_closed_flag=loop_closed_flag,
            out_path=os.path.join(output_model_test_path, "psf_open_to_closed_loop.gif"),
            cmap=psf_colormap,
            fps=15,
            use_log=False,
            range_min_max=True,
            visual_gamma=1.0,
            log_dynamic_range_db=60.0,  # PSF: -60..0 dB respecto del pico de cada frame
        )

        save_transition_gif_fast(
            tensor_all=phi_transition_all,
            loop_closed_flag=loop_closed_flag,
            out_path=os.path.join(output_model_test_path, "phi_open_to_closed_loop.gif"),
            cmap=phase_colormap,
            fps=15,
            use_log=False,
            range_min_max=True,
        )

        save_transition_gif_fast(
            tensor_all=I_transition_all,
            loop_closed_flag=loop_closed_flag,
            out_path=os.path.join(output_model_test_path, "I_propagated_open_to_closed_loop.gif"),
            cmap=propagation_colormap,
            fps=15,
            use_log=False,
            range_min_max=True,
        )

        if amplitude_transition_all is not None:
            save_transition_gif_fast(
                tensor_all=amplitude_transition_all,
                loop_closed_flag=loop_closed_flag,
                out_path=os.path.join(output_model_test_path, "atmospheric_amplitude.gif"),
                cmap=propagation_colormap,
                fps=15,
                use_log=False,
                range_min_max=True,
            )

        # This is exactly what enters the NN: noise + normalization included.
        # Saved only if the NN input has 4 channels.
        if net_input_transition_all is not None:
            save_network_input_4ch_gif(
                tensor_all=net_input_transition_all,
                out_path=os.path.join(output_model_test_path, "network_input_4pupils_exact.gif"),
                cmap=propagation_colormap,
                fps=15,
                range_min_max=True,
                channel_order=network_channel_order,
            )

        save_z_estimation_heatmap(
            z_est_all=z_est_transition_all,
            out_path=os.path.join(output_model_test_path, "nn_basis_estimation_heatmap.png"),
            cl_sample=cl_sample,
            Name=f"{Name} | {model_basis.basis_kind} | n_modes={n_modes}",
        )

        save_single_image(
            img=psf_integrated_closed_loop,
            out_path=os.path.join(output_model_test_path, "psf_integrated_closed_loop_after_cl_plus_10.png"),
            cmap=psf_colormap,
            title=f"{Name} | {model_basis.basis_kind} | Integrated PSF from sample {psf_integrated_start}",
            use_log=True,
            range_min_max=True,
        )

        save_metric_plot(
            y_open=wfe_open,
            y_closed=wfe_closed,
            y_ideal=wfe_ideal,
            ylabel="WFE RMS [rad]",
            out_path=os.path.join(output_model_test_path, "wfe_plot.png"),
            Name=f"{Name} | {model_basis.basis_kind} | n_modes={n_modes}",
            cl_sample=cl_sample,
        )

        save_metric_plot(
            y_open=strehl_open,
            y_closed=strehl_closed,
            y_ideal=strehl_ideal,
            ylabel="Strehl ratio",
            out_path=os.path.join(output_model_test_path, "strehl_plot.png"),
            Name=f"{Name} | {model_basis.basis_kind} | n_modes={n_modes}",
            cl_sample=cl_sample,
        )

    return results



# =========================================================
# LONG-DURATION STABILITY TEST
# =========================================================
#
# Design goals:
#   - The controller still runs at EVERY camera/atmosphere frame.
#   - The atmosphere evolves on-the-fly; no temporal sequence is stored.
#   - Only WFE and Strehl are retained.
#   - At most stability_max_metric_points are retained, uniformly distributed
#     over the complete physical duration.
#   - Time is reported in minutes using the configured frame_rate.
#   - No GIFs, PSF cubes, phase cubes, propagated-intensity cubes, network-input
#     cubes or per-frame modal histories are allocated.
# =========================================================


def _stability_sample_indices(
    total_frames: int,
    max_points: int,
    cl_frame: int,
) -> np.ndarray:
    """
    Return <= max_points monotonically increasing frame indices.

    The points are spread across the entire run. The first/last frame and the
    samples immediately around loop closure are forced into the selection when
    they exist.
    """
    total_frames = int(total_frames)
    max_points = int(max_points)
    cl_frame = int(cl_frame)

    if total_frames <= 0:
        raise ValueError("total_frames debe ser > 0.")
    if max_points < 4:
        raise ValueError("max_points debe ser >= 4.")

    if total_frames <= max_points:
        return np.arange(total_frames, dtype=np.int64)

    mandatory = {0, total_frames - 1}

    if 0 <= cl_frame - 1 < total_frames:
        mandatory.add(cl_frame - 1)
    if 0 <= cl_frame < total_frames:
        mandatory.add(cl_frame)

    mandatory = np.array(sorted(mandatory), dtype=np.int64)

    n_uniform = max(0, max_points - len(mandatory))
    if n_uniform > 0:
        uniform = np.linspace(
            0,
            total_frames - 1,
            n_uniform,
            dtype=np.int64,
        )
        selected = np.unique(np.concatenate([uniform, mandatory]))
    else:
        selected = mandatory

    # Rounding/duplicate behavior from linspace normally keeps this <= max_points.
    # Keep a strict final guard without ever removing mandatory points.
    if selected.size > max_points:
        mandatory_set = set(int(v) for v in mandatory.tolist())
        optional = np.array(
            [v for v in selected if int(v) not in mandatory_set],
            dtype=np.int64,
        )
        keep_optional = max_points - len(mandatory)
        if keep_optional > 0 and optional.size > keep_optional:
            pick = np.linspace(
                0,
                optional.size - 1,
                keep_optional,
                dtype=np.int64,
            )
            optional = optional[pick]
        selected = np.unique(np.concatenate([mandatory, optional]))

    return np.sort(selected.astype(np.int64))


def _build_streaming_atmosphere(
    *,
    ref_bundle: Dict[str, Any],
    resolution: int,
    model_telescope_diameter: float,
    precision,
    device: str,
) -> Atmosphere:
    """
    Build the same Torch ATMOSPHERE configuration as the normal closed-loop test,
    but without allocating/storing a temporal sequence.
    """
    D_atm = (
        float(model_telescope_diameter)
        if atmosphere_telescope_diameter is None
        else float(atmosphere_telescope_diameter)
    )

    wavelength = (
        _source_wavelength_from_bundle(ref_bundle)
        if atmosphere_wavelength is None
        else float(atmosphere_wavelength)
    )

    if atmosphere_scintillation and atmosphere_propagation_mode != "asm_delta":
        print(
            "[ATMOSPHERE] WARNING: scintillation=True with "
            "propagation_mode!='asm_delta'. The geometric field has unit "
            "amplitude, so no scintillation will be present."
        )

    atm = Atmosphere(
        batch_size=1,
        resolution=int(resolution),
        telescope_diameter=D_atm,
        frame_rate=float(frame_rate),
        r0=r0,
        L0=float(L0),
        l0=float(l0),
        wind_speed=wind_speed,
        fractional_r0=fractional_r0,
        altitude=altitude,
        wind_direction=wind_direction,
        n_subharmonic_levels=int(atmosphere_n_subharmonic_levels),
        subharmonic_mode=str(atmosphere_subharmonic_mode),
        direction_convention=str(atmosphere_direction_convention),
        normalize_fractional_r0=bool(atmosphere_normalize_fractional_r0),
        remove_piston=bool(atmosphere_remove_piston),
        store_components=False,
        device=device,
        dtype=precision.real,
        seed=int(seed),
        wavelength=wavelength,
        r0_reference_wavelength=float(atmosphere_r0_reference_wavelength),
        propagation_mode=str(atmosphere_propagation_mode),
        asm_extra_pixels=atmosphere_asm_extra_pixels,
        asm_min_physical_margin=float(atmosphere_asm_min_physical_margin),
        asm_padding_factor=float(atmosphere_asm_padding_factor),
        delta_mode=str(atmosphere_delta_mode),
        delta_wrap_warning_threshold=float(atmosphere_delta_wrap_warning_threshold),
        temporal_reanchor_interval=int(atmosphere_temporal_reanchor_interval),
        frozen_flow_mode=str(atmosphere_frozen_flow_mode),
    )

    return atm


def build_stability_forward_pipe(WFS, NN, noise_pipe, norm_type, noise_flag):
    """
    Minimal WFS -> noise -> normalization -> NN path for the stability test.

    It intentionally preserves the same camera-noise ordering as the normal
    forward pipe: I_full is augmented first and I_crop second. Only zEst is
    returned because no propagated images are stored.
    """
    def forward_pipe(propagation_pupil=None, phi=None):
        I_full, I_crop = WFS.propagate(
            pupil=propagation_pupil,
            phi=phi,
            return_both=True,
        )

        if noise_flag:
            # Preserve the original random-number consumption order.
            I_full = noise_pipe(I_full)
            I_crop = noise_pipe(I_crop)

        # Preserve the same normalization operations.
        I_full = norm_I(I_full, norm=norm_type)
        I_crop = norm_I(I_crop, norm=norm_type)

        with torch.no_grad():
            zEst = NN(I_crop).detach()

        return zEst

    return forward_pipe


@torch.no_grad()
def run_stability_single_model(
    *,
    model_info,
    bundle,
    model_basis: ModelBasisBundle,
    ref_bundle_for_atmosphere: Dict[str, Any],
    noise_pipe,
    device: str,
    shared_precision,
    total_frames: int,
    metric_indices: np.ndarray,
    current_noise_level,
    propagation_pupil_enabled=True,
    propagation_obstruction_px=0.0,
    propagation_spider_count=0,
    propagation_spider_width_px=0.0,
    propagation_pupil_random_seed=None,
):
    """
    Streaming closed-loop stability test.

    The AO loop advances every frame, but WFE/Strehl are evaluated only at
    metric_indices. Memory therefore depends on max_points, not total_frames.
    """
    Name = model_info["Name"]
    train_cfg = bundle["train_cfg"]
    WFS = bundle["WFS"]
    NN = bundle["NN"]
    telescope_cfg = bundle["telescope_cfg"]

    base_pupil = _as_4d_pupil(
        model_basis.telescope_pupil,
        device=device,
        dtype=shared_precision.real,
    )

    telescope_pupil = build_wfs_propagation_pupil(
        ideal_pupil=base_pupil,
        enabled=propagation_pupil_enabled,
        central_obstruction_diam_px=propagation_obstruction_px,
        spiders=propagation_spider_count,
        spiders_px=propagation_spider_width_px,
        soft_edge_px=propagation_soft_edge_px,
        seed=propagation_pupil_random_seed,
    )
    propagation_pupil = telescope_pupil

    resolution = int(telescope_cfg["resolution"])
    fovPx = resolution * 4

    # Diffraction-limited Strehl reference for the exact physical pupil.
    phi_ref = torch.zeros(
        (1, 1, resolution, resolution),
        device=device,
        dtype=shared_precision.real,
    )
    I_psf_ref_peak = float(
        get_psf(
            telescope_pupil=telescope_pupil,
            phi=phi_ref,
            fovPx=fovPx,
        ).max().item()
    )

    zDecomposeMat = model_basis.zDecomposeMat
    zComposeMat = model_basis.zComposeMat
    n_modes = int(model_basis.n_modes)

    expected_pixels = resolution ** 2
    if zDecomposeMat.ndim != 2 or int(zDecomposeMat.shape[-1]) != expected_pixels:
        raise ValueError(
            f"Base incompatible para {Name}: zDecomposeMat debe tener shape "
            f"(n_modes, {expected_pixels}) y llegó {tuple(zDecomposeMat.shape)}."
        )
    if int(zDecomposeMat.shape[0]) != n_modes:
        raise ValueError(
            f"Base incompatible para {Name}: zDecomposeMat contiene "
            f"{zDecomposeMat.shape[0]} modos y se esperaban {n_modes}."
        )

    forward_pipe = build_stability_forward_pipe(
        WFS=WFS,
        NN=NN,
        noise_pipe=noise_pipe,
        norm_type=train_cfg["norm_type"],
        noise_flag=noise_flag,
    )

    # The complete atmosphere lives only as ONE evolving frame/state.
    atm = _build_streaming_atmosphere(
        ref_bundle=ref_bundle_for_atmosphere,
        resolution=resolution,
        model_telescope_diameter=float(telescope_cfg["diameter"]),
        precision=shared_precision,
        device=device,
    )

    phase = atm.gen(seed=int(seed))
    sampled_r0 = atm.r0_batch.detach().cpu().clone()

    metric_indices = np.asarray(metric_indices, dtype=np.int64)
    n_metric = int(metric_indices.size)

    metric_frame = np.empty(n_metric, dtype=np.int64)
    metric_time_min = np.empty(n_metric, dtype=np.float64)
    metric_wfe = np.empty(n_metric, dtype=np.float32)
    metric_strehl = np.empty(n_metric, dtype=np.float32)

    metric_ptr = 0
    next_metric_frame = (
        int(metric_indices[metric_ptr])
        if metric_ptr < n_metric
        else None
    )

    # PI integrator state in the exact model-specific training basis.
    u_integral = torch.zeros(
        (1, n_modes),
        dtype=shared_precision.real,
        device=device,
    )

    last_wfe = None
    last_strehl = None
    progress_refresh = max(1, total_frames // 1000)

    loop_pbar = tqdm(
        range(total_frames),
        desc=(
            f"Stability | {Name} | {model_basis.basis_kind} | "
            f"n_modes={n_modes} | noise={current_noise_level}"
        ),
        leave=True,
    )

    for n in loop_pbar:
        if n > 0:
            phase = atm.update()

        # Atmosphere public shape: [1,B,H,W], B=1.
        phi_atm_raw = phase[0:1, 0:1]
        phi_atm = phi_atm_raw * telescope_pupil

        atmospheric_amplitude = None
        if atmosphere_scintillation:
            atmospheric_amplitude = atm.field.abs()[0:1, 0:1]

        frame_propagation_pupil = build_frame_optical_pupil(
            telescope_pupil,
            atmospheric_amplitude,
        )

        # -------------------------------------------------
        # OPEN LOOP PREFIX
        # -------------------------------------------------
        if n < cl_sample:
            phi_metric = remove_piston(
                phi_atm,
                telescope_pupil,
            )

            # Keep one WFS/NN acquisition per physical frame, as in a real camera
            # stream, even before loop closure. Its estimate is intentionally not
            # applied until cl_sample.
            z_probe = forward_pipe(
                propagation_pupil=frame_propagation_pupil,
                phi=phi_metric,
            )

            if int(z_probe.shape[-1]) != n_modes:
                raise ValueError(
                    f"La red {Name} entregó {z_probe.shape[-1]} modos, "
                    f"pero la base activa tiene {n_modes}."
                )

        # -------------------------------------------------
        # CLOSED LOOP
        # -------------------------------------------------
        else:
            # Command physically present before the current camera acquisition.
            phi_cmd_current = zernike_compose_torch(
                zernike_phi_vector=u_integral,
                zComposeMat=zComposeMat,
            )

            phi_residual_before_update = remove_piston(
                (phi_atm + phi_cmd_current) * telescope_pupil,
                telescope_pupil,
            )

            z_est = forward_pipe(
                propagation_pupil=frame_propagation_pupil,
                phi=phi_residual_before_update,
            ).to(device=device, dtype=shared_precision.real)

            if int(z_est.shape[-1]) != n_modes:
                raise ValueError(
                    f"La red {Name} entregó {z_est.shape[-1]} modos, "
                    f"pero la base activa tiene {n_modes}."
                )

            # Same PI law used by the normal closed-loop test.
            u_integral = u_integral + loop_sign * ki * z_est

            if use_integrator_clamp:
                u_integral = clamp_integrator(
                    u_integral,
                    integrator_limit,
                )

            # Only build the post-update phase when this frame must be measured.
            # For unsaved frames, the next iteration only needs u_integral.
            if next_metric_frame is not None and n == next_metric_frame:
                u_command_new = u_integral + loop_sign * kp * z_est
                phi_cmd_new = zernike_compose_torch(
                    zernike_phi_vector=u_command_new,
                    zComposeMat=zComposeMat,
                )
                phi_metric = remove_piston(
                    (phi_atm + phi_cmd_new) * telescope_pupil,
                    telescope_pupil,
                )
            else:
                phi_metric = None

        # -------------------------------------------------
        # SPARSE METRIC SAMPLING
        # -------------------------------------------------
        if next_metric_frame is not None and n == next_metric_frame:
            if phi_metric is None:
                raise RuntimeError(
                    f"phi_metric no fue construido para el frame muestreado {n}."
                )

            # WFE and PSF/Strehl are evaluated ONLY for selected frames.
            wfe_value = float(
                get_wfe_rms(
                    phi_metric.detach().cpu(),
                    telescope_pupil.detach().cpu(),
                )[0].item()
            )

            psf_metric = get_psf(
                telescope_pupil=frame_propagation_pupil,
                phi=phi_metric,
                fovPx=fovPx,
            )
            strehl_value = float(
                get_strehl_from_psf(
                    psf_metric,
                    I_psf_ref_peak,
                )[0].item()
            )

            metric_frame[metric_ptr] = int(n)
            metric_time_min[metric_ptr] = float(n) / float(frame_rate) / 60.0
            metric_wfe[metric_ptr] = wfe_value
            metric_strehl[metric_ptr] = strehl_value

            last_wfe = wfe_value
            last_strehl = strehl_value

            metric_ptr += 1
            next_metric_frame = (
                int(metric_indices[metric_ptr])
                if metric_ptr < n_metric
                else None
            )

        if (
            n == 0
            or (n + 1) % progress_refresh == 0
            or n == total_frames - 1
        ):
            postfix = {
                "frame": n + 1,
                "time_min": f"{float(n) / float(frame_rate) / 60.0:.3f}",
                "mode": "CL" if n >= cl_sample else "OL",
            }
            if last_wfe is not None:
                postfix["WFE"] = f"{last_wfe:.4f}"
                postfix["SR"] = f"{last_strehl:.4f}"
            loop_pbar.set_postfix(postfix)

    if metric_ptr != n_metric:
        raise RuntimeError(
            f"Se esperaban {n_metric} muestras métricas y se guardaron {metric_ptr}."
        )

    return {
        "sample_indices": torch.from_numpy(metric_frame.copy()),
        "time_minutes": torch.from_numpy(metric_time_min.copy()),
        "wfe_rms_rad": torch.from_numpy(metric_wfe.copy()),
        "strehl_ratio": torch.from_numpy(metric_strehl.copy()),
        "sampled_r0": sampled_r0,
        "atmosphere_info": atm.info(),
        "n_total_frames": int(total_frames),
        "duration_minutes_requested": float(total_frames) / float(frame_rate) / 60.0,
        "frame_rate_hz": float(frame_rate),
        "cl_sample": int(cl_sample),
        "cl_time_minutes": float(cl_sample) / float(frame_rate) / 60.0,
        "kp": float(kp),
        "ki": float(ki),
        "loop_sign": float(loop_sign),
        "noise_level": int(current_noise_level),
        "basis_info": dict(model_basis.info),
        "model_info": dict(model_info),
    }


def save_stability_comparison_plot(
    *,
    results_by_model: Dict[str, Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    out_path: str,
    title: str,
    cl_time_minutes: float,
):
    """
    Save one time-series plot versus physical time in minutes.
    """
    plt.figure(figsize=(13, 6))

    for model_name, results in results_by_model.items():
        x = results["time_minutes"].detach().cpu().numpy()
        y = results[metric_key].detach().cpu().numpy()
        plt.plot(
            x,
            y,
            linewidth=1.5,
            label=model_name,
        )

    plt.axvline(
        float(cl_time_minutes),
        linestyle="--",
        linewidth=1.2,
        color="k",
        alpha=0.7,
        label="Loop closes",
    )

    plt.xlabel("Time [min]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if len(results_by_model) > 1:
        plt.legend()
    else:
        # Keep the loop-close marker visible in the single-model case.
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()




def main():
    args = parse_args()

    test_name = args.test_name
    device = args.device
    selected_noise_level = args.noise_level
    selected_noise_output_mode = args.noise_output_mode
    selected_noise_param_jitter = args.noise_param_jitter
    duration_minutes = float(args.duration_minutes)
    max_points = int(args.max_points)

    if duration_minutes <= 0.0:
        raise ValueError("--duration_minutes debe ser > 0.")
    if max_points < 4:
        raise ValueError("--max_points debe ser >= 4.")
    if float(frame_rate) <= 0.0:
        raise ValueError("frame_rate debe ser > 0.")

    selected_propagation_enabled = (
        use_telescope_pupil_in_propagation
        and not args.disable_propagation_obstruction
    )
    selected_propagation_obstruction_px = float(args.propagation_obstruction_px)
    selected_propagation_spiders = int(args.propagation_spiders)
    selected_propagation_spiders_px = float(args.propagation_spiders_px)

    # Physical duration -> number of camera/atmosphere updates.
    total_frames = max(
        1,
        int(round(duration_minutes * 60.0 * float(frame_rate))),
    )

    if cl_sample >= total_frames:
        raise ValueError(
            f"cl_sample={cl_sample} debe ser menor que total_frames={total_frames}. "
            "Aumenta --duration_minutes o reduce cl_sample."
        )

    metric_indices = _stability_sample_indices(
        total_frames=total_frames,
        max_points=max_points,
        cl_frame=cl_sample,
    )

    actual_duration_minutes = float(total_frames) / float(frame_rate) / 60.0
    cl_time_minutes = float(cl_sample) / float(frame_rate) / 60.0

    output_dir = os.path.join("./TEST/CL_STABILITY", test_name)
    os.makedirs(output_dir, exist_ok=True)

    if len(MODELS_TO_TEST) == 0:
        raise ValueError("MODELS_TO_TEST está vacío. Debes definir al menos un modelo.")

    # -----------------------------------------------------
    # CPU-only reference configuration.
    # No NN/WFS/basis is loaded here.
    # -----------------------------------------------------
    first_model_info = MODELS_TO_TEST[0]
    first_train_path = first_model_info["train_path"]
    first_stage = first_model_info["stage"]
    first_config_path = os.path.join(
        first_train_path,
        "config",
        "all_cfg.pt",
    )

    print(f"Loading reference config on CPU: {first_model_info['Name']}")
    ref_cfg = torch.load(first_config_path, map_location="cpu")

    ref_telescope_cfg = ref_cfg["telescope_cfg"]
    ref_stage_cfg = ref_cfg["stages_cfg"][
        0 if first_stage is None else first_stage
    ]
    ref_train_cfg = ref_stage_cfg["train"]
    ref_precision = get_precision(ref_train_cfg["precision"])

    ref_resolution = int(ref_telescope_cfg["resolution"])
    ref_diameter = float(ref_telescope_cfg["diameter"])
    ref_precision_name = ref_train_cfg["precision"]

    ref_bundle_for_atmosphere = {
        "cfg": ref_cfg,
    }

    if selected_noise_level == "all":
        noise_levels_to_run = list(range(0, 11))
    else:
        noise_levels_to_run = [
            validate_noise_level(selected_noise_level)
        ]

    print("\n" + "=" * 80)
    print("LONG-DURATION CLOSED-LOOP STABILITY TEST")
    print("=" * 80)
    print(f"Frame rate           : {float(frame_rate):.3f} Hz")
    print(f"Requested duration   : {duration_minutes:.6f} min")
    print(f"Total frames         : {total_frames}")
    print(f"Loop closes at frame : {cl_sample}")
    print(f"Loop closes at       : {cl_time_minutes:.6f} min")
    print(f"Stored metric points : {len(metric_indices)} / {total_frames}")
    print("Stored data           : WFE + Strehl only")
    print("Atmosphere sequence   : streaming / not stored")
    print("=" * 80)

    global_summary = {
        "test_name": test_name,
        "device": device,
        "frame_rate_hz": float(frame_rate),
        "duration_minutes_requested": duration_minutes,
        "duration_minutes_actual": actual_duration_minutes,
        "total_frames": int(total_frames),
        "max_metric_points": int(max_points),
        "stored_metric_points": int(len(metric_indices)),
        "metric_sample_indices": torch.from_numpy(metric_indices.copy()),
        "cl_sample": int(cl_sample),
        "cl_time_minutes": cl_time_minutes,
        "kp": float(kp),
        "ki": float(ki),
        "loop_sign": float(loop_sign),
        "noise_flag": bool(noise_flag),
        "r0": r0,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "altitude": altitude,
        "fractional_r0": fractional_r0,
        "L0": L0,
        "models_to_test": MODELS_TO_TEST,
        "noise_levels": {},
    }

    for current_noise_level in noise_levels_to_run:
        current_noise_level = int(current_noise_level)
        current_noise_name = f"noise_{current_noise_level:02d}"

        print(
            f"\nRunning LONG stability test with "
            f"noise_level={current_noise_level}/10"
        )

        output_noise_dir = os.path.join(
            output_dir,
            current_noise_name,
        )
        os.makedirs(output_noise_dir, exist_ok=True)

        # Same noise configuration logic as the normal test.
        noise_pipe = build_noise_pipe(
            noise_level=current_noise_level,
            output_mode=selected_noise_output_mode,
            return_metadata=False,
            param_jitter=selected_noise_param_jitter,
        )

        results_by_model = {}

        # One model resident on GPU at a time.
        for model_index, model_info in enumerate(MODELS_TO_TEST):
            Name = model_info["Name"]

            print(
                f"\nLoading model {model_index + 1}/{len(MODELS_TO_TEST)}: "
                f"{Name}"
            )

            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = None
            if torch.cuda.is_available():
                cuda_rng_states = torch.cuda.get_rng_state_all()

            bundle = load_model_bundle(
                model_info,
                device=device,
            )

            validate_shared_model_compatibility(
                model_info,
                bundle,
                ref_resolution=ref_resolution,
                ref_diameter=ref_diameter,
                ref_precision_name=ref_precision_name,
            )

            model_basis = build_model_basis_bundle(
                model_info,
                bundle,
                device=device,
                dtype=ref_precision.real,
            )

            # Keep sequential model loading from perturbing the stochastic test.
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

            print(
                f"Basis loaded | {Name} | "
                f"basis={model_basis.basis_kind} | "
                f"n_modes={model_basis.n_modes} | "
                f"path={model_basis.basis_path}"
            )

            results = run_stability_single_model(
                model_info=model_info,
                bundle=bundle,
                model_basis=model_basis,
                ref_bundle_for_atmosphere=ref_bundle_for_atmosphere,
                noise_pipe=noise_pipe,
                device=device,
                shared_precision=ref_precision,
                total_frames=total_frames,
                metric_indices=metric_indices,
                current_noise_level=current_noise_level,
                propagation_pupil_enabled=selected_propagation_enabled,
                propagation_obstruction_px=selected_propagation_obstruction_px,
                propagation_spider_count=selected_propagation_spiders,
                propagation_spider_width_px=selected_propagation_spiders_px,
                propagation_pupil_random_seed=propagation_pupil_seed,
            )

            output_model_dir = os.path.join(
                output_noise_dir,
                Name,
            )
            os.makedirs(output_model_dir, exist_ok=True)

            metrics_path = os.path.join(
                output_model_dir,
                "stability_metrics.pt",
            )
            torch.save(results, metrics_path)

            # Only sparse scalar metric arrays remain on CPU.
            results_by_model[Name] = results

            del model_basis
            del bundle
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Released model from GPU: {Name}")

        # -------------------------------------------------
        # ONLY TWO PLOTS: WFE and Strehl versus minutes.
        # -------------------------------------------------
        wfe_plot_path = os.path.join(
            output_noise_dir,
            "wfe_stability_minutes.png",
        )
        strehl_plot_path = os.path.join(
            output_noise_dir,
            "strehl_stability_minutes.png",
        )

        save_stability_comparison_plot(
            results_by_model=results_by_model,
            metric_key="wfe_rms_rad",
            ylabel="WFE RMS [rad]",
            out_path=wfe_plot_path,
            title=(
                f"Closed-loop stability | WFE | "
                f"noise={current_noise_level} | "
                f"{actual_duration_minutes:.3f} min"
            ),
            cl_time_minutes=cl_time_minutes,
        )

        save_stability_comparison_plot(
            results_by_model=results_by_model,
            metric_key="strehl_ratio",
            ylabel="Strehl ratio",
            out_path=strehl_plot_path,
            title=(
                f"Closed-loop stability | Strehl ratio | "
                f"noise={current_noise_level} | "
                f"{actual_duration_minutes:.3f} min"
            ),
            cl_time_minutes=cl_time_minutes,
        )

        noise_summary = {
            "noise_level": current_noise_level,
            "noise_output_mode": selected_noise_output_mode,
            "noise_adc_headroom": noise_adc_headroom,
            "noise_param_jitter": selected_noise_param_jitter,
            "wfe_plot": wfe_plot_path,
            "strehl_plot": strehl_plot_path,
            "model_metric_paths": {
                name: os.path.join(
                    output_noise_dir,
                    name,
                    "stability_metrics.pt",
                )
                for name in results_by_model.keys()
            },
        }

        torch.save(
            noise_summary,
            os.path.join(
                output_noise_dir,
                "stability_summary.pt",
            ),
        )

        global_summary["noise_levels"][
            current_noise_level
        ] = noise_summary

        print(f"\nFinished noise_level={current_noise_level}/10")
        print(f" - {wfe_plot_path}")
        print(f" - {strehl_plot_path}")

        del results_by_model
        del noise_pipe
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(
        global_summary,
        os.path.join(
            output_dir,
            "stability_summary_all_noise_levels.pt",
        ),
    )

    print("\nLong-duration stability test finished successfully.")
    print(f"General output folder: {output_dir}")

if __name__ == "__main__":
    main()