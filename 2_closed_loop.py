from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm

import GENERAL_FUNCTIONS.camera_noise as camera_noise
from GENERAL_FUNCTIONS.functions_torch import get_precision, norm_I, pad2size
from ATMOSPHERE.atmosphere import Atmosphere
from MODAL_BASIS.Zernike import (
    get_zernike_on_pupil,
    zernike_compose_torch,
    zernike_decompose_torch,
)
from NN.model_manager import ModelManager


# =============================================================================
# USER CONFIG
# =============================================================================

RUN_MODE = "closed_loop"  # "closed_loop" | "stability"

DEFAULT_DEVICE = "cuda:4"
DEFAULT_TEST_NAME_CLOSED_LOOP = "Test"
DEFAULT_TEST_NAME_STABILITY = "Stability"

SEED = 400

# Normal closed loop
N_SAMPLES = 250
CL_SAMPLE = 100

# Stability
STABILITY_DURATION_MINUTES = 10.0
STABILITY_MAX_METRIC_POINTS = 1000

# PI
KP = 0.0
KI = 1.0
LOOP_SIGN = -1.0
USE_INTEGRATOR_CLAMP = False
INTEGRATOR_LIMIT = 5.0

# Camera
NOISE_FLAG = True
NOISE_LEVEL = "all"  # 0..10 | "all" | "training"
NOISE_OUTPUT_MODE = "Mono8"
NOISE_PARAM_JITTER = 0.10

# Recommended: exact domain used by current training.
# Alternatives preserve capabilities from the old closed-loop code.
CAMERA_NOISE_DOMAIN = "network_input"  # network_input | full_frame | legacy_independent

# Physical pupil
USE_TELESCOPE_PUPIL_IN_PROPAGATION = True
PHYSICAL_PUPIL_MODE = "model"  # model | analytic | tensor
PROPAGATION_CENTRAL_OBSTRUCTION_DIAM_PX = 0.0
PROPAGATION_SPIDERS = 0
PROPAGATION_SPIDERS_PX = 0.0
PROPAGATION_SOFT_EDGE_PX = 0.0
PROPAGATION_PUPIL_SEED = SEED

CUSTOM_PUPIL_PATH = None
CUSTOM_PUPIL_INTERSECT_MODEL_SUPPORT = True
CUSTOM_PUPIL_NORMALIZE = True
CUSTOM_OBSTRUCTION_OFFSET_X_PX = 0.0
CUSTOM_OBSTRUCTION_OFFSET_Y_PX = 0.0
CUSTOM_SPIDER_ANGLES_DEG = None
CUSTOM_SPIDER_ORIGIN_OFFSET_X_PX = None
CUSTOM_SPIDER_ORIGIN_OFFSET_Y_PX = None
CUSTOM_SPIDER_BIDIRECTIONAL = False
SAVE_WFS_CROP_BOXES_DIAGNOSTIC = True

# Atmosphere
R0 = 0.02
FRACTIONAL_R0 = [0.5, 0.3, 0.2]
WIND_DIRECTION = [90.0, 240.0, 120.0]
WIND_SPEED = [8.0, 10.0, 10.0]
ALTITUDE = [0.0, 1000.0, 2000.0]
L0 = 15.0
LITTLE_L0 = 1e-10
FRAME_RATE = 1000.0

ATMOSPHERE_TELESCOPE_DIAMETER = 0.6  # None -> model diameter
ATMOSPHERE_WAVELENGTH = None         # None -> source wavelength in first model
ATMOSPHERE_R0_REFERENCE_WAVELENGTH = 500e-9
ATMOSPHERE_R0_SAMPLING = "uniform_r0"
ATMOSPHERE_SCINTILLATION = False
ATMOSPHERE_PROPAGATION_MODE = "geometric"
ATMOSPHERE_DELTA_MODE = "final"
ATMOSPHERE_FROZEN_FLOW_MODE = "analytic"
ATMOSPHERE_DIRECTION_CONVENTION = "oopao"
ATMOSPHERE_N_SUBHARMONIC_LEVELS = 3
ATMOSPHERE_SUBHARMONIC_MODE = "full"
ATMOSPHERE_NORMALIZE_FRACTIONAL_R0 = True
ATMOSPHERE_REMOVE_PISTON = False
ATMOSPHERE_TEMPORAL_REANCHOR_INTERVAL = 256
ATMOSPHERE_ASM_EXTRA_PIXELS = "auto"
ATMOSPHERE_ASM_MIN_PHYSICAL_MARGIN = 0.75
ATMOSPHERE_ASM_PADDING_FACTOR = 2.0
ATMOSPHERE_DELTA_WRAP_WARNING_THRESHOLD = 1.9 * np.pi
ATMOSPHERE_WARMUP = True
ATMOSPHERE_WARMUP_UPDATES = 2
ATMOSPHERE_SEQUENCE_STORAGE = "device"  # normal mode only
SAVE_ATMOSPHERE_SEQUENCE_PT = False

# Visualization
PHASE_COLORMAP = "viridis"
PROPAGATION_COLORMAP = "hot"
PSF_COLORMAP = "hot"
MAX_GIF_FRAMES = 250
GIF_FPS = 15

# Models used by normal 2-like mode
MODELS_TO_TEST_CLOSED_LOOP = [
    {
        "train_path": "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS/TRAIN/single/phiRes_128/nnRes_36/DM_BAX370_MRS/ACTUATOR/nModes_97/MODEL_ConvNeXtTiny/RAMA_zscore_global_source",
        "basis_path": "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS/MODAL_BASIS/DEFORMABLE_MIRROR_BASIS/BAX370_MRS/ACTUATOR_BASIS_RES_128.pt",
        "stage": 0,
        "Name": "convNext_zscore_global_source",
    },

]

# Models used by old 3-like stability mode
MODELS_TO_TEST_STABILITY = [
    {
        "train_path": "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS_EVOLVE_V2/TRAIN/single/phiRes_128/nnRes_32/DM_IDEAL/ZERNIKE/nModes_209/MODEL_ConvNeXtTiny/PYR4_32_std_grad_local",
        "stage": 0,
        "Name": "ConvNeXTiny_32_std_grad_local_209",
    },
]


def parse_args():
    p = argparse.ArgumentParser("Unified normal/stability closed-loop evaluator")
    p.add_argument("--mode", default=RUN_MODE, choices=["closed_loop", "stability"])
    p.add_argument("--test_name", default=None)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--noise_level", default=str(NOISE_LEVEL), help="0..10 | all | training")
    p.add_argument("--noise_output_mode", default=NOISE_OUTPUT_MODE, choices=["Mono8", "Mono12", "Mono16"])
    p.add_argument("--noise_param_jitter", type=float, default=NOISE_PARAM_JITTER)
    p.add_argument("--duration_minutes", type=float, default=STABILITY_DURATION_MINUTES)
    p.add_argument("--max_points", type=int, default=STABILITY_MAX_METRIC_POINTS)
    p.add_argument("--propagation_obstruction_px", type=float, default=PROPAGATION_CENTRAL_OBSTRUCTION_DIAM_PX)
    p.add_argument("--propagation_spiders", type=int, default=PROPAGATION_SPIDERS)
    p.add_argument("--propagation_spiders_px", type=float, default=PROPAGATION_SPIDERS_PX)
    p.add_argument("--disable_propagation_obstruction", action="store_true")
    return p.parse_args()


# =============================================================================
# BASIC HELPERS / PUPIL
# =============================================================================

def safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def as_4d_pupil(pupil, device, dtype):
    pupil = torch.as_tensor(pupil, device=device, dtype=dtype)
    if pupil.ndim == 2:
        pupil = pupil[None, None]
    elif pupil.ndim == 3:
        pupil = pupil[None]
    if pupil.ndim != 4 or pupil.shape[:2] != (1, 1):
        raise ValueError(f"Pupil must be [1,1,H,W], got {tuple(pupil.shape)}")
    return pupil.contiguous()


def circular_pupil(n, *, device, dtype, soft_edge_px=0.0):
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    c = (n - 1) / 2.0
    q = torch.arange(n, device=device, dtype=work_dtype)
    Y, X = torch.meshgrid(q, q, indexing="ij")
    R = torch.sqrt((X - c) ** 2 + (Y - c) ** 2)
    if soft_edge_px <= 0:
        out = (R <= c).to(dtype)
    else:
        t = ((R - c) / float(soft_edge_px)).clamp(0, 1)
        out = (0.5 * (1 + torch.cos(torch.pi * t))).to(dtype)
    return out[None, None].contiguous()


def circular_pupil_telescope(
    n, *, device, dtype, soft_edge_px=0.0, spiders=0, spiders_px=0.0,
    central_obstruction_diam_px=0.0, central_obstruction_offset_x_px=0.0,
    central_obstruction_offset_y_px=0.0, spider_angles_deg=None,
    spider_origin_offset_x_px=None, spider_origin_offset_y_px=None,
    spider_bidirectional=False, theta0=None,
):
    c = (n - 1) / 2.0
    q = torch.arange(n, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(q, q, indexing="ij")
    Rout = torch.sqrt((X - c) ** 2 + (Y - c) ** 2)
    pupil = circular_pupil(n, device=device, dtype=torch.float32, soft_edge_px=soft_edge_px)[0, 0]

    ox = c + float(central_obstruction_offset_x_px)
    oy = c + float(central_obstruction_offset_y_px)
    if central_obstruction_diam_px > 0:
        Robs = torch.sqrt((X - ox) ** 2 + (Y - oy) ** 2)
        pupil *= (Robs >= float(central_obstruction_diam_px) / 2.0).float()

    if spider_angles_deg is not None:
        angles = torch.tensor(np.deg2rad([float(v) for v in spider_angles_deg]), device=device, dtype=torch.float32)
    elif spiders > 0:
        if theta0 is None:
            theta0 = torch.rand((), device=device) * (2 * torch.pi / spiders)
        theta0 = torch.as_tensor(theta0, device=device, dtype=torch.float32)
        angles = torch.stack([theta0 + k * 2 * torch.pi / spiders for k in range(spiders)])
    else:
        angles = torch.empty(0, device=device)

    if angles.numel() and spiders_px > 0:
        sx = float(central_obstruction_offset_x_px if spider_origin_offset_x_px is None else spider_origin_offset_x_px)
        sy = float(central_obstruction_offset_y_px if spider_origin_offset_y_px is None else spider_origin_offset_y_px)
        Xs, Ys = X - (c + sx), Y - (c + sy)
        smask = torch.ones((n, n), device=device)
        hw = float(spiders_px) / 2
        for theta in angles:
            ux, uy = torch.cos(theta), torch.sin(theta)
            dist = torch.abs(-uy * Xs + ux * Ys)
            radial = ux * Xs + uy * Ys
            arm = torch.ones_like(radial, dtype=torch.bool) if spider_bidirectional else radial >= 0
            region = (dist <= hw) & arm & (Rout <= c)
            smask = torch.where(region, torch.zeros_like(smask), smask)
        pupil *= smask

    return pupil.to(dtype)[None, None]


def load_custom_pupil(path, reference_pupil):
    if path is None:
        raise ValueError("CUSTOM_PUPIL_PATH is required for PHYSICAL_PUPIL_MODE='tensor'")
    path = os.path.expanduser(str(path))
    if path.lower().endswith(".npy"):
        data = np.load(path)
    else:
        data = safe_load(path)
        if isinstance(data, dict):
            for key in ("pupil", "telescope_pupil", "mask"):
                if key in data:
                    data = data[key]
                    break
    pupil = torch.as_tensor(data, dtype=reference_pupil.dtype).squeeze()
    if pupil.ndim != 2 or tuple(pupil.shape) != tuple(reference_pupil.shape[-2:]):
        raise ValueError(f"Invalid custom pupil shape {tuple(pupil.shape)}")
    pupil = pupil.to(reference_pupil.device)[None, None]
    if CUSTOM_PUPIL_NORMALIZE and float(pupil.max()) > 0:
        pupil = pupil / pupil.max()
    if CUSTOM_PUPIL_INTERSECT_MODEL_SUPPORT:
        pupil = pupil * reference_pupil.clamp_min(0)
    return pupil


def build_physical_pupil(model_pupil, *, enabled, obstruction_px, spiders, spiders_px):
    if not enabled or PHYSICAL_PUPIL_MODE == "model":
        return model_pupil.clone()
    if PHYSICAL_PUPIL_MODE == "tensor":
        return load_custom_pupil(CUSTOM_PUPIL_PATH, model_pupil)
    if PHYSICAL_PUPIL_MODE != "analytic":
        raise ValueError("PHYSICAL_PUPIL_MODE must be model|analytic|tensor")

    n = int(model_pupil.shape[-1])
    theta0 = None
    if CUSTOM_SPIDER_ANGLES_DEG is None and spiders > 0 and spiders_px > 0:
        gen = torch.Generator(device=model_pupil.device)
        gen.manual_seed(int(PROPAGATION_PUPIL_SEED))
        theta0 = torch.rand((), device=model_pupil.device, generator=gen) * (2 * torch.pi / spiders)

    mask = circular_pupil_telescope(
        n, device=model_pupil.device, dtype=model_pupil.dtype,
        soft_edge_px=PROPAGATION_SOFT_EDGE_PX,
        spiders=spiders, spiders_px=spiders_px,
        central_obstruction_diam_px=obstruction_px,
        central_obstruction_offset_x_px=CUSTOM_OBSTRUCTION_OFFSET_X_PX,
        central_obstruction_offset_y_px=CUSTOM_OBSTRUCTION_OFFSET_Y_PX,
        spider_angles_deg=CUSTOM_SPIDER_ANGLES_DEG,
        spider_origin_offset_x_px=CUSTOM_SPIDER_ORIGIN_OFFSET_X_PX,
        spider_origin_offset_y_px=CUSTOM_SPIDER_ORIGIN_OFFSET_Y_PX,
        spider_bidirectional=CUSTOM_SPIDER_BIDIRECTIONAL,
        theta0=theta0,
    )
    return model_pupil * mask


# =============================================================================
# CAMERA: current training API + historical 0..10 scale
# =============================================================================

NoiseSelection = Union[int, Literal["training"]]


def validate_noise_level(value):
    try:
        level = int(value)
    except Exception as exc:
        raise ValueError("noise_level must be 0..10, all, or training") from exc
    if not 0 <= level <= 10:
        raise ValueError("noise_level must be in [0,10]")
    return level


def parse_noise_levels(value):
    value = str(value).strip().lower()
    if value == "all":
        return list(range(11))
    if value == "training":
        return ["training"]
    return [validate_noise_level(value)]


def range_from_dict(cfg):
    return camera_noise.Range(float(cfg["low"]), float(cfg["high"]), bool(cfg.get("log", False)))


def signal_domain_from_dict(cfg):
    return camera_noise.CameraSignalDomain(
        signal_e=range_from_dict(cfg["signal_e"]),
        bg_e=range_from_dict(cfg["bg_e"]),
    )


def build_training_camera_config(cfg):
    if cfg is None:
        raise RuntimeError("noise_level='training' requires camera_cfg in all_cfg.pt")
    e = cfg["electronics"]
    fw = e.get("full_well_e", None)
    electronics = camera_noise.CameraElectronics(
        gain_e_per_dn=range_from_dict(e["gain_e_per_dn"]),
        read_sigma_e=range_from_dict(e["read_sigma_e"]),
        bias_dn=range_from_dict(e["bias_dn"]),
        full_well_e=None if fw is None else range_from_dict(fw),
        output_mode=e["output_mode"],
        mono16_align=e["mono16_align"],
        use_ste_adc=bool(e["use_ste_adc"]),
        add_prnu=bool(e["add_prnu"]),
        prnu_sigma=float(e["prnu_sigma"]),
        add_dsnu=bool(e["add_dsnu"]),
        dsnu_sigma_e=float(e["dsnu_sigma_e"]),
    )
    return camera_noise.CameraNoiseAugmentConfig(
        low=signal_domain_from_dict(cfg["low"]),
        normal=signal_domain_from_dict(cfg["normal"]),
        good=signal_domain_from_dict(cfg["good"]),
        electronics=electronics,
        p_low=float(cfg["p_low"]), p_normal=float(cfg["p_normal"]), p_good=float(cfg["p_good"]),
        signal_scaling=cfg["signal_scaling"], signal_scope=cfg["signal_scope"],
        signal_parameter_mode=cfg["signal_parameter_mode"],
        electronics_parameter_mode=cfg["electronics_parameter_mode"],
        shot_noise=cfg["shot_noise"], return_metadata=False,
    )


def interp_log(a, b, t):
    return float(np.exp(np.log(a) * (1 - t) + np.log(b) * t))


def range_around(base, jitter, *, log, min_value=1e-12):
    j = max(0.0, float(jitter))
    lo = max(base * (1 - j), min_value)
    hi = max(base * (1 + j), lo + min_value)
    return camera_noise.Range(lo, hi, log)


def build_level_camera_config(level, saved_cfg, output_mode, jitter):
    """Historical 0..10 severity, executed through the current camera model."""
    L = validate_noise_level(level)
    t = L / 10.0
    signal = interp_log(9000.0, 35.0, t)
    bg = interp_log(0.03, 55.0, t)
    read = 0.35 * (1 - t) + 8.0 * t
    bias = 0.5 * (1 - t) + 4.0 * t
    prnu = 0.002 * (1 - t) + 0.012 * t
    dsnu = 0.05 * (1 - t) + 0.60 * t

    domain = camera_noise.CameraSignalDomain(
        signal_e=range_around(signal, jitter, log=True, min_value=1e-6),
        bg_e=range_around(bg, jitter, log=True, min_value=1e-6),
    )

    if saved_cfg is not None:
        se = saved_cfg["electronics"]
        gain = range_from_dict(se["gain_e_per_dn"])
        fw_cfg = se.get("full_well_e", None)
        fw = None if fw_cfg is None else range_from_dict(fw_cfg)
        align = se.get("mono16_align", "lsb")
        scaling = saved_cfg.get("signal_scaling", "peak")
        scope = saved_cfg.get("signal_scope", "sample")
        shot = saved_cfg.get("shot_noise", "poisson")
    else:
        # Legacy fallback without auto-gain.
        gain = camera_noise.Range(30.0, 30.0, False)
        fw, align, scaling, scope, shot = None, "lsb", "peak", "sample", "poisson"

    electronics = camera_noise.CameraElectronics(
        gain_e_per_dn=gain,
        read_sigma_e=range_around(read, jitter, log=False, min_value=1e-6),
        bias_dn=range_around(bias, jitter, log=False, min_value=0.0),
        full_well_e=fw,
        output_mode=output_mode,
        mono16_align=align,
        use_ste_adc=False,
        add_prnu=True, prnu_sigma=prnu,
        add_dsnu=True, dsnu_sigma_e=dsnu,
    )
    return camera_noise.CameraNoiseAugmentConfig(
        low=domain, normal=domain, good=domain, electronics=electronics,
        p_low=0.0, p_normal=1.0, p_good=0.0,
        signal_scaling=scaling, signal_scope=scope,
        signal_parameter_mode="per_sample",
        electronics_parameter_mode="per_batch",
        shot_noise=shot, return_metadata=False,
    )


@dataclass
class CameraRuntime:
    mode: str
    crop: Optional[camera_noise.CameraNoiseAugmenter]
    full: Optional[camera_noise.CameraNoiseAugmenter]
    seed: int
    selection: NoiseSelection
    meta: Dict[str, Any] = field(default_factory=dict)


def new_augmenter(cfg, device, seed):
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(seed))
    return camera_noise.CameraNoiseAugmenter(cfg, generator=gen)


def build_camera_runtime(bundle, selection, device, output_mode, jitter):
    saved_cfg = bundle.get("camera_cfg", None)
    if selection == "training":
        cfg = build_training_camera_config(saved_cfg)
        kind = "training_distribution"
    else:
        cfg = build_level_camera_config(int(selection), saved_cfg, output_mode, jitter)
        kind = f"severity_{int(selection):02d}"

    seed_offset = 0 if saved_cfg is None else int(saved_cfg.get("seed_offset", 0))
    camera_seed = int(SEED) + seed_offset
    mode = str(CAMERA_NOISE_DOMAIN).strip().lower()
    if mode not in ("network_input", "full_frame", "legacy_independent"):
        raise ValueError("CAMERA_NOISE_DOMAIN must be network_input|full_frame|legacy_independent")

    crop = new_augmenter(cfg, device, camera_seed) if mode in ("network_input", "legacy_independent") else None
    full = new_augmenter(cfg, device, camera_seed + 1_000_003) if mode in ("full_frame", "legacy_independent") else None

    return CameraRuntime(
        mode=mode, crop=crop, full=full, seed=camera_seed, selection=selection,
        meta={"kind": kind, "has_saved_camera_cfg": saved_cfg is not None},
    )


def apply_camera(augmenter, x):
    if augmenter is None:
        return x
    if getattr(augmenter, "_sequence_shape", None) is None:
        augmenter.start_sequence(x)
    return augmenter(x)


def camera_snapshot(runtime):
    if runtime is None:
        return None

    def one(aug):
        if aug is None:
            return None
        params = getattr(aug, "_sequence_params", None)
        return {
            "signal_mode": getattr(aug, "_signal_mode", None),
            "sequence_shape": getattr(aug, "_sequence_shape", None),
            "params": None if params is None else {
                k: None if v is None else v.detach().cpu()
                for k, v in params.items()
            },
        }

    return {
        "mode": runtime.mode, "seed": runtime.seed, "selection": runtime.selection,
        **runtime.meta, "crop": one(runtime.crop), "full": one(runtime.full),
    }


# =============================================================================
# MODEL / CONFIG / BASIS
# =============================================================================

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


def source_wavelength_from_cfg(cfg):
    source = cfg.get("source_cfg", cfg.get("source", None))
    if source is None:
        # Compatibility with pre-SourceCfg experiments (e.g. old stability runs).
        # The repository historically used 635 nm.
        print("[WARNING] Legacy all_cfg.pt has no source_cfg; assuming 635 nm.")
        return 635e-9
    return float(source["wavelength"] if isinstance(source, dict) else source.wavelength)


def load_model_bundle(model_info, device):
    train_path = Path(model_info["train_path"])
    stage = model_info.get("stage", 0)
    stage_idx = 0 if stage is None else int(stage)

    nn_path = train_path / f"stage_{stage_idx}" / "wts" / "model_full.pt"
    cfg_path = train_path / "config" / "all_cfg.pt"
    wfs_path = train_path / "WFS" / "WFS.pt"

    cfg = safe_load(cfg_path, "cpu")
    telescope_cfg = cfg["telescope_cfg"]
    wfs_cfg = cfg["wfs_cfg"]
    model_cfg = cfg["model_cfg"]
    stage_cfg = cfg["stages_cfg"][stage_idx]
    atmosphere_cfg = stage_cfg["atmosphere"]
    train_cfg = stage_cfg["train"]
    runtime_cfg = cfg.get("runtime_cfg", None)
    modal_basis_cfg = cfg.get("modal_basis_cfg", None)
    camera_cfg = cfg.get("camera_cfg", None)

    if runtime_cfg is not None:
        precision_name = str(runtime_cfg["precision"]).lower()
    else:
        if "precision" not in train_cfg:
            raise KeyError("No runtime precision and no legacy train precision")
        precision_name = str(train_cfg["precision"]).lower()

    precision = get_precision(precision_name)

    WFS = safe_load(wfs_path, device)
    if hasattr(WFS, "device"):
        WFS.device = device
    if hasattr(WFS, "crop_pos_noise"):
        WFS.crop_pos_noise = 0
    if hasattr(WFS, "crop_size_noise"):
        WFS.crop_size_noise = 0

    NN = ModelManager.load_full(str(nn_path), device=device, dtype=precision.real).eval()

    return {
        "cfg": cfg,
        "telescope_cfg": telescope_cfg,
        "wfs_cfg": wfs_cfg,
        "model_cfg": model_cfg,
        "stage_cfg": stage_cfg,
        "atmosphere_cfg": atmosphere_cfg,
        "train_cfg": train_cfg,
        "runtime_cfg": runtime_cfg,
        "precision_name": precision_name,
        "modal_basis_cfg": modal_basis_cfg,
        "camera_cfg": camera_cfg,
        "precision": precision,
        "WFS": WFS,
        "NN": NN,
        "source_wavelength": source_wavelength_from_cfg(cfg),
    }


def get_model_n_modes(model_info, bundle):
    if model_info.get("n_modes", None) is not None:
        n = int(model_info["n_modes"])
        if n <= 0:
            raise ValueError("n_modes must be > 0")
        return n

    mb = bundle.get("modal_basis_cfg", None)
    if mb is not None and "n_modes" in mb:
        n = int(mb["n_modes"])
        if n <= 0:
            raise ValueError("modal_basis_cfg['n_modes'] must be > 0")
        return n

    legacy = bundle["atmosphere_cfg"]
    if "n_modes" not in legacy:
        raise KeyError("Could not determine n_modes")
    n = int(legacy["n_modes"])
    if n <= 0:
        raise ValueError("Legacy n_modes must be > 0")
    return n


def resolve_dm_basis_path(model_info, bundle):
    repo_root = Path(__file__).resolve().parent
    for key in ("basis_path", "dm_basis_path"):
        value = model_info.get(key, None)
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = repo_root / path
            return str(path.resolve())

    mb = bundle.get("modal_basis_cfg", None)
    if mb is None:
        raise KeyError("No modal_basis_cfg and no explicit basis_path")
    for key in ("root", "name", "basis_type"):
        if key not in mb:
            raise KeyError(f"modal_basis_cfg missing {key!r}")

    root = Path(mb["root"]).expanduser()
    if not root.is_absolute():
        root = repo_root / root
    root = root.resolve()

    resolution = int(bundle["telescope_cfg"]["resolution"])
    return str(root / str(mb["name"]) / f"{str(mb['basis_type']).upper()}_BASIS_RES_{resolution}.pt")


def build_model_basis_bundle(model_info, bundle, device, dtype):
    name = model_info.get("Name", "<unknown>")
    telescope_cfg = bundle["telescope_cfg"]
    atmosphere_cfg = bundle["atmosphere_cfg"]
    mb = bundle.get("modal_basis_cfg", None)
    WFS = bundle["WFS"]

    resolution = int(telescope_cfg["resolution"])
    diameter = float(telescope_cfg["diameter"])

    if not hasattr(WFS, "telescope_pupil"):
        raise AttributeError("Loaded WFS has no telescope_pupil")

    trained_pupil = as_4d_pupil(WFS.telescope_pupil, device, dtype)
    if tuple(trained_pupil.shape[-2:]) != (resolution, resolution):
        raise ValueError("WFS pupil resolution mismatch")

    if mb is not None:
        dm_basis = bool(mb.get("use_dm_basis", False))
        dm_basis_type = str(mb.get("basis_type", "UNKNOWN")).upper()
        dm_name = str(mb.get("name", "UNKNOWN"))
    else:
        dm_basis = bool(atmosphere_cfg.get("dm_basis", False))
        dm_basis_type = str(atmosphere_cfg.get("dm_basis_type", "UNKNOWN")).upper()
        dm_name = str(atmosphere_cfg.get("dm_name", "UNKNOWN"))

    if dm_basis:
        basis_path = resolve_dm_basis_path(model_info, bundle)
        if not os.path.isfile(basis_path):
            raise FileNotFoundError(f"DM basis not found for {name}: {basis_path}")

        data = safe_load(basis_path, "cpu")
        required = {"zComposeMat", "zDecomposeMat", "telescope_pupil"}
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"DM basis missing keys: {sorted(missing)}")

        zComposeMat = data["zComposeMat"]
        zDecomposeMat = data["zDecomposeMat"]

        expected_pixels = resolution * resolution
        if zDecomposeMat.ndim != 2 or int(zDecomposeMat.shape[1]) != expected_pixels:
            raise ValueError(
                f"zDecomposeMat must be [n_modes,{expected_pixels}], got {tuple(zDecomposeMat.shape)}"
            )

        n_modes = int(zDecomposeMat.shape[0])
        expected_compose = (resolution, resolution, n_modes)
        if tuple(zComposeMat.shape) != expected_compose:
            raise ValueError(
                f"zComposeMat must be {expected_compose}, got {tuple(zComposeMat.shape)}"
            )

        configured_n = None if mb is None else mb.get("n_modes", None)
        if configured_n is not None and int(configured_n) != n_modes:
            print(
                f"[WARNING] modal_basis_cfg n_modes={configured_n}, "
                f"but DM basis contains {n_modes}; basis file takes precedence."
            )

        dm_pupil = as_4d_pupil(data["telescope_pupil"], device, dtype)
        if tuple(dm_pupil.shape[-2:]) != (resolution, resolution):
            raise ValueError("DM basis pupil resolution mismatch")

        zComposeMat = zComposeMat.to(device=device, dtype=dtype)
        zDecomposeMat = zDecomposeMat.to(device=device, dtype=dtype)

        return ModelBasisBundle(
            telescope_pupil=trained_pupil,
            zDecomposeMat=zDecomposeMat,
            zComposeMat=zComposeMat,
            n_modes=n_modes,
            basis_kind=f"DM_{dm_basis_type}",
            dm_basis=True,
            dm_basis_type=dm_basis_type,
            dm_name=dm_name,
            basis_path=basis_path,
            info={
                "Name": name,
                "resolution": resolution,
                "diameter": diameter,
                "n_modes": n_modes,
                "zComposeMat_shape": tuple(zComposeMat.shape),
                "zDecomposeMat_shape": tuple(zDecomposeMat.shape),
                "dm_pupil_shape": tuple(dm_pupil.shape),
                "telescope_pupil_shape": tuple(trained_pupil.shape),
            },
        )

    n_modes = get_model_n_modes(model_info, bundle)
    ideal_pupil = circular_pupil(resolution, device=device, dtype=dtype)
    zDecomposeMat, zComposeMat = get_zernike_on_pupil(
        ideal_pupil=ideal_pupil.squeeze().cpu(),
        physical_pupil=trained_pupil.squeeze().cpu(),
        diameter=diameter,
        nModes=n_modes,
        type="torch",
    )

    return ModelBasisBundle(
        telescope_pupil=trained_pupil,
        zDecomposeMat=zDecomposeMat.to(device=device, dtype=dtype),
        zComposeMat=zComposeMat.to(device=device, dtype=dtype),
        n_modes=n_modes,
        basis_kind="ZERNIKE",
        dm_basis=False,
        info={
            "Name": name,
            "resolution": resolution,
            "diameter": diameter,
            "n_modes": n_modes,
            "zComposeMat_shape": tuple(zComposeMat.shape),
            "zDecomposeMat_shape": tuple(zDecomposeMat.shape),
            "telescope_pupil_shape": tuple(trained_pupil.shape),
        },
    )


def validate_shared_model_compatibility(
    model_info, bundle, ref_resolution, ref_diameter, ref_precision_name, ref_wavelength
):
    name = model_info.get("Name", "<unknown>")
    tel = bundle["telescope_cfg"]

    if int(tel["resolution"]) != int(ref_resolution):
        raise ValueError(f"{name}: resolution mismatch")
    if not math.isclose(float(tel["diameter"]), float(ref_diameter), abs_tol=1e-12):
        raise ValueError(f"{name}: telescope diameter mismatch")
    if str(bundle["precision_name"]).lower() != str(ref_precision_name).lower():
        raise ValueError(f"{name}: precision mismatch")

    if ATMOSPHERE_WAVELENGTH is None and not math.isclose(
        float(bundle["source_wavelength"]), float(ref_wavelength), rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError(f"{name}: source wavelength mismatch")


def assert_network_output_matches_basis(NN, WFS, basis, device, dtype, name):
    # Shape-only test: no WFS propagation and no camera RNG consumption.
    dummy = torch.zeros(tuple(WFS.output_shape), device=device, dtype=dtype)
    with torch.no_grad():
        y = NN(dummy)
    if y.ndim != 2 or int(y.shape[-1]) != int(basis.n_modes):
        raise ValueError(
            f"{name}: NN output {tuple(y.shape)} incompatible with basis n_modes={basis.n_modes}"
        )


# =============================================================================
# ATMOSPHERE
# =============================================================================

@dataclass
class AtmosphereSequence:
    phase: torch.Tensor
    amplitude: Optional[torch.Tensor]
    sampled_r0: torch.Tensor
    info: Dict[str, Any]


def build_atmosphere(ref_cfg, resolution, telescope_diameter, precision, device):
    D = float(telescope_diameter if ATMOSPHERE_TELESCOPE_DIAMETER is None else ATMOSPHERE_TELESCOPE_DIAMETER)
    wavelength = (
        source_wavelength_from_cfg(ref_cfg)
        if ATMOSPHERE_WAVELENGTH is None
        else float(ATMOSPHERE_WAVELENGTH)
    )

    if ATMOSPHERE_SCINTILLATION and ATMOSPHERE_PROPAGATION_MODE != "asm_delta":
        print("[ATMOSPHERE] WARNING: scintillation=True with geometric propagation gives unit amplitude.")

    atm = Atmosphere(
        batch_size=1,
        resolution=int(resolution),
        telescope_diameter=D,
        frame_rate=float(FRAME_RATE),
        r0=R0,
        L0=float(L0),
        l0=float(LITTLE_L0),
        wind_speed=WIND_SPEED,
        fractional_r0=FRACTIONAL_R0,
        altitude=ALTITUDE,
        wind_direction=WIND_DIRECTION,
        n_subharmonic_levels=int(ATMOSPHERE_N_SUBHARMONIC_LEVELS),
        subharmonic_mode=str(ATMOSPHERE_SUBHARMONIC_MODE),
        direction_convention=str(ATMOSPHERE_DIRECTION_CONVENTION),
        normalize_fractional_r0=bool(ATMOSPHERE_NORMALIZE_FRACTIONAL_R0),
        remove_piston=bool(ATMOSPHERE_REMOVE_PISTON),
        store_components=False,
        device=device,
        dtype=precision.real,
        seed=int(SEED),
        wavelength=wavelength,
        r0_reference_wavelength=float(ATMOSPHERE_R0_REFERENCE_WAVELENGTH),
        propagation_mode=str(ATMOSPHERE_PROPAGATION_MODE),
        asm_extra_pixels=ATMOSPHERE_ASM_EXTRA_PIXELS,
        asm_min_physical_margin=float(ATMOSPHERE_ASM_MIN_PHYSICAL_MARGIN),
        asm_padding_factor=float(ATMOSPHERE_ASM_PADDING_FACTOR),
        delta_mode=str(ATMOSPHERE_DELTA_MODE),
        delta_wrap_warning_threshold=float(ATMOSPHERE_DELTA_WRAP_WARNING_THRESHOLD),
        temporal_reanchor_interval=int(ATMOSPHERE_TEMPORAL_REANCHOR_INTERVAL),
        frozen_flow_mode=str(ATMOSPHERE_FROZEN_FLOW_MODE),
        r0_sampling=str(ATMOSPHERE_R0_SAMPLING),
    )

    if ATMOSPHERE_WARMUP and hasattr(atm, "warmup"):
        atm.warmup(n_updates=int(ATMOSPHERE_WARMUP_UPDATES))
    return atm


@torch.no_grad()
def generate_atmosphere_sequence(ref_cfg, resolution, telescope_diameter, precision, device, output_dir):
    if ATMOSPHERE_SEQUENCE_STORAGE not in ("device", "cpu"):
        raise ValueError("ATMOSPHERE_SEQUENCE_STORAGE must be device|cpu")

    atm = build_atmosphere(ref_cfg, resolution, telescope_diameter, precision, device)
    store_device = device if ATMOSPHERE_SEQUENCE_STORAGE == "device" else "cpu"

    phase_all = torch.empty(
        (N_SAMPLES, 1, resolution, resolution), dtype=precision.real, device=store_device
    )
    amplitude_all = torch.empty_like(phase_all) if ATMOSPHERE_SCINTILLATION else None

    phase = atm.gen(seed=int(SEED))
    pbar = tqdm(range(N_SAMPLES), desc="Generating ATMOSPHERE sequence")
    for n in pbar:
        if n > 0:
            phase = atm.update()
        phase_all[n, 0].copy_(phase[0, 0].detach().to(store_device))
        if amplitude_all is not None:
            amplitude_all[n, 0].copy_(atm.field.abs()[0, 0].detach().to(store_device))

    sampled_r0 = atm.r0_batch.detach().cpu().clone()
    info = atm.info()
    info["sampled_r0_batch_m"] = sampled_r0
    info["sequence_frames"] = int(N_SAMPLES)
    info["sequence_storage_device"] = str(phase_all.device)

    seq = AtmosphereSequence(phase_all, amplitude_all, sampled_r0, info)
    if SAVE_ATMOSPHERE_SEQUENCE_PT:
        torch.save(
            {
                "phase": phase_all.cpu(),
                "amplitude": None if amplitude_all is None else amplitude_all.cpu(),
                "sampled_r0": sampled_r0,
                "atmosphere_info": info,
            },
            os.path.join(output_dir, "atmosphere_sequence_torch.pt"),
        )
    return seq


def get_sequence_frame(seq, index, device, dtype):
    phase = seq.phase[index:index + 1].to(device=device, dtype=dtype, non_blocking=True)
    amplitude = None
    if seq.amplitude is not None:
        amplitude = seq.amplitude[index:index + 1].to(device=device, dtype=dtype, non_blocking=True)
    return phase, amplitude


# =============================================================================
# OPTICS / WFS
# =============================================================================

@torch.no_grad()
def remove_piston(phi, pupil):
    mask = (pupil > 0).to(phi.dtype)
    denom = mask.sum((-2, -1), keepdim=True).clamp_min(1.0)
    piston = (phi * mask).sum((-2, -1), keepdim=True) / denom
    return (phi - piston) * mask


@torch.no_grad()
def get_wfe_rms(phi, pupil):
    phi0 = remove_piston(phi, pupil)
    mask = (pupil > 0).to(phi0.dtype)
    denom = mask.sum((-2, -1)).clamp_min(1.0)
    return torch.sqrt((phi0.square() * mask).sum((-2, -1)) / denom).squeeze(1)


@torch.no_grad()
def get_psf(telescope_pupil, phi, fovPx):
    phasor = telescope_pupil * torch.exp(1j * phi)
    padded = pad2size(phasor, (fovPx, fovPx))
    field = torch.fft.fftshift(torch.fft.fft2(padded), dim=(-2, -1))
    return torch.abs(field) ** 2


@torch.no_grad()
def get_strehl(psf, reference_peak):
    return psf.amax((-2, -1)).squeeze(1) / max(float(reference_peak), 1e-12)


def build_frame_optical_pupil(telescope_pupil, atmospheric_amplitude):
    return telescope_pupil if atmospheric_amplitude is None else telescope_pupil * atmospheric_amplitude


def clamp_integrator(x, limit):
    return x if limit is None else torch.clamp(x, -limit, limit)


def build_forward_pipe(WFS, NN, runtime, norm_type, noise_flag):
    mode = "network_input" if runtime is None else runtime.mode

    def forward(propagation_pupil, phi, return_boxes=False, infer_nn=True):
        boxes = None

        if noise_flag and mode == "full_frame":
            I_full = WFS.propagate(pupil=propagation_pupil, phi=phi, no_crop=True)
            I_full = apply_camera(runtime.full, I_full)
            if return_boxes:
                I_crop, boxes = WFS.crop_pyr(I_full, return_boxes=True)
            else:
                I_crop = WFS.crop_pyr(I_full, return_boxes=False)
        else:
            if return_boxes:
                I_full, I_crop, boxes = WFS.propagate(
                    pupil=propagation_pupil, phi=phi, return_both=True, return_boxes=True
                )
            else:
                I_full, I_crop = WFS.propagate(
                    pupil=propagation_pupil, phi=phi, return_both=True
                )

            if noise_flag and mode == "network_input":
                I_crop = apply_camera(runtime.crop, I_crop)
            elif noise_flag and mode == "legacy_independent":
                I_full = apply_camera(runtime.full, I_full)
                I_crop = apply_camera(runtime.crop, I_crop)

        I_full = norm_I(I_full, norm=norm_type)
        I_crop = norm_I(I_crop, norm=norm_type)

        z = None
        if infer_nn:
            with torch.no_grad():
                z = NN(I_crop).detach()

        return z, I_full, I_crop, boxes

    return forward



# =============================================================================
# VISUALIZATION
# =============================================================================

def gif_indices(n, max_frames=MAX_GIF_FRAMES):
    if n <= max_frames:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_frames, dtype=np.int64)


def image2d(x, use_log=False):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if use_log:
            x = torch.log10(x.clamp_min(1e-12))
        x = x.squeeze().numpy()
    else:
        x = np.asarray(x).squeeze()
        if use_log:
            x = np.log10(np.clip(x, 1e-12, None))
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {x.shape}")
    return x.astype(np.float32)


def save_single_image(img, out_path, cmap="hot", title=None, use_log=False):
    x = image2d(img, use_log)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(x, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_crop_boxes(I_full, boxes, out_path):
    if boxes is None:
        return
    img = image2d(I_full)
    boxes = np.asarray(boxes)
    if boxes.ndim == 3:
        boxes = boxes[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(img, cmap=PROPAGATION_COLORMAP)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        ax.add_patch(Rectangle((float(x0), float(y0)), float(x1 - x0), float(y1 - y0), fill=False, linewidth=1.5))
        ax.text(float(x0), float(y0), f"P{i+1}", fontsize=8)
    ax.set_title("WFS full frame + exact NN crops")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def get_four_channel_order(WFS):
    try:
        coords = getattr(WFS, "coords", None)
        if coords is None:
            return [0, 1, 2, 3]
        coords = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        if coords.ndim != 2 or coords.shape[0] != 4:
            return [0, 1, 2, 3]
        idx = np.arange(4)
        top = idx[np.argsort(coords[:, 1])[:2]]
        bottom = idx[np.argsort(coords[:, 1])[2:]]
        top = top[np.argsort(coords[top, 0])]
        bottom = bottom[np.argsort(coords[bottom, 0])]
        return [int(top[0]), int(top[1]), int(bottom[0]), int(bottom[1])]
    except Exception:
        return [0, 1, 2, 3]


def mosaic4(frame, order):
    x = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    x = x[np.asarray(order)]
    return np.concatenate([
        np.concatenate([x[0], x[1]], axis=1),
        np.concatenate([x[2], x[3]], axis=1),
    ], axis=0).astype(np.float32)


def save_network_input_gif(tensor_all, out_path, order):
    idx = gif_indices(int(tensor_all.shape[0]))
    x = tensor_all[idx].detach().cpu().float()
    cmap = plt.get_cmap(PROPAGATION_COLORMAP)
    frames = []
    for k in tqdm(range(x.shape[0]), desc=f"Saving {os.path.basename(out_path)}", leave=False):
        img = mosaic4(x[k], order)
        mn, mx = float(img.min()), float(img.max())
        norm = np.zeros_like(img) if np.isclose(mn, mx) else (img - mn) / (mx - mn)
        frames.append((255 * cmap(np.clip(norm, 0, 1))[..., :3]).astype(np.uint8))
    imageio.mimsave(out_path, frames, duration=1000.0 / GIF_FPS, loop=0)


def save_transition_gif(tensor_all, out_path, cmap, *, symmetric=False, psf_db=None):
    idx = gif_indices(int(tensor_all.shape[0]))
    x = tensor_all[idx].detach().cpu().float().squeeze(1).numpy()
    cmap_obj = plt.get_cmap(cmap)
    vmin, vmax = float(x.min()), float(x.max())
    if symmetric:
        a = max(abs(vmin), abs(vmax))
        vmin, vmax = -a, a

    frames = []
    for img in tqdm(x, desc=f"Saving {os.path.basename(out_path)}", leave=False):
        if psf_db is not None:
            peak = max(float(img.max()), 1e-30)
            db = 10 * np.log10(np.clip(img / peak, 1e-30, None))
            db = np.clip(db, -float(psf_db), 0)
            norm = (db + float(psf_db)) / float(psf_db)
        else:
            norm = np.zeros_like(img) if np.isclose(vmin, vmax) else (img - vmin) / (vmax - vmin)
        frames.append((255 * cmap_obj(np.clip(norm, 0, 1))[..., :3]).astype(np.uint8))
    imageio.mimsave(out_path, frames, duration=1000.0 / GIF_FPS, loop=0)


def save_heatmap(data, out_path, title, cl_sample):
    z = data.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(z.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.axvline(cl_sample, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Mode / actuator")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_estimation_vs_control(z_est, u_cmd, out_path, title, cl_sample):
    zr = torch.sqrt(torch.mean(z_est.float() ** 2, dim=1)).numpy()
    ur = torch.sqrt(torch.mean(u_cmd.float() ** 2, dim=1)).numpy()
    x = np.arange(len(zr))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, zr, label="NN estimation RMS")
    ax.plot(x, ur, label="Accumulated control RMS")
    ax.axvline(cl_sample, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Modal RMS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(y_open, y_closed, y_ideal, ylabel, out_path, title, cl_sample):
    x = np.arange(len(y_open))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y_open.numpy(), label="Open loop")
    ax.plot(x, y_closed.numpy(), label="Closed loop")
    ax.plot(x, y_ideal.numpy(), label="Ideal projection")
    ax.axvline(cl_sample, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(open_curve, closed_by_model, ideal_by_model, ylabel, out_path, cl_sample):
    x = np.arange(len(open_curve))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, open_curve.numpy(), linewidth=2.2, label="Open loop")
    for name, y in ideal_by_model.items():
        ax.plot(x, y.numpy(), "--", linewidth=1.4, label=f"{name} ideal")
    for name, y in closed_by_model.items():
        ax.plot(x, y.numpy(), linewidth=1.8, label=f"{name} closed")
    ax.axvline(cl_sample, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Model comparison | {ylabel}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# NORMAL CLOSED LOOP (mode 2)
# =============================================================================

@torch.no_grad()
def run_closed_loop_single_model(
    model_info, bundle, basis, atmosphere_sequence, output_model_dir,
    camera_runtime, device, precision, noise_selection,
    propagation_enabled, propagation_obstruction_px,
    propagation_spider_count, propagation_spider_width_px,
):
    name = model_info["Name"]
    train_cfg = bundle["train_cfg"]
    WFS, NN = bundle["WFS"], bundle["NN"]
    resolution = int(bundle["telescope_cfg"]["resolution"])

    telescope_pupil = build_physical_pupil(
        as_4d_pupil(basis.telescope_pupil, device, precision.real),
        enabled=propagation_enabled,
        obstruction_px=propagation_obstruction_px,
        spiders=propagation_spider_count,
        spiders_px=propagation_spider_width_px,
    )

    assert_network_output_matches_basis(NN, WFS, basis, device, precision.real, name)

    zD, zC = basis.zDecomposeMat, basis.zComposeMat
    n_modes = int(basis.n_modes)
    fov = resolution * 4

    phi_ref = torch.zeros((1, 1, resolution, resolution), device=device, dtype=precision.real)
    reference_peak = float(get_psf(telescope_pupil, phi_ref, fov).max().item())

    forward = build_forward_pipe(WFS, NN, camera_runtime, train_cfg["norm_type"], NOISE_FLAG)

    os.makedirs(output_model_dir, exist_ok=True)

    # Clean geometry diagnostic, deliberately outside camera RNG.
    phi0_raw, amp0 = get_sequence_frame(atmosphere_sequence, 0, device, precision.real)
    phi0 = phi0_raw * telescope_pupil
    pupil0 = build_frame_optical_pupil(telescope_pupil, amp0)
    try:
        I_full_diag, _, boxes = WFS.propagate(
            pupil=pupil0, phi=phi0, return_both=True, return_boxes=True
        )
    except Exception:
        I_full_diag, _ = WFS.propagate(pupil=pupil0, phi=phi0, return_both=True)
        boxes = None

    save_single_image(
        telescope_pupil,
        os.path.join(output_model_dir, "physical_pupil_used.png"),
        cmap="gray",
        title=f"{name} | physical pupil | {PHYSICAL_PUPIL_MODE}",
    )
    if SAVE_WFS_CROP_BOXES_DIAGNOSTIC:
        save_crop_boxes(
            I_full_diag,
            boxes,
            os.path.join(output_model_dir, "wfs_full_with_exact_network_crops.png"),
        )

    IpropH, IpropW = int(I_full_diag.shape[-2]), int(I_full_diag.shape[-1])

    # CPU history keeps GPU memory bounded while preserving 2-closed-loop outputs.
    psf_hist = torch.empty((N_SAMPLES, 1, fov, fov), dtype=precision.real)
    phi_hist = torch.empty((N_SAMPLES, 1, resolution, resolution), dtype=precision.real)
    Ifull_hist = torch.empty((N_SAMPLES, 1, IpropH, IpropW), dtype=precision.real)
    z_hist = torch.empty((N_SAMPLES, n_modes), dtype=precision.real)
    u_hist = torch.empty((N_SAMPLES, n_modes), dtype=precision.real)
    phi_cmd_hist = torch.empty((N_SAMPLES, 1, resolution, resolution), dtype=precision.real)

    amp_hist = (
        torch.empty((N_SAMPLES, 1, resolution, resolution), dtype=precision.real)
        if atmosphere_sequence.amplitude is not None else None
    )
    net_hist = None
    channel_order = None

    wfe_open = torch.empty(N_SAMPLES, dtype=precision.real)
    wfe_closed = torch.empty_like(wfe_open)
    wfe_ideal = torch.empty_like(wfe_open)
    sr_open = torch.empty_like(wfe_open)
    sr_closed = torch.empty_like(wfe_open)
    sr_ideal = torch.empty_like(wfe_open)
    loop_closed = torch.zeros(N_SAMPLES, dtype=torch.bool)

    u_integral = torch.zeros((1, n_modes), device=device, dtype=precision.real)
    u_applied = torch.zeros_like(u_integral)

    pbar = tqdm(
        range(N_SAMPLES),
        desc=f"Closed loop | {name} | {basis.basis_kind} | n={n_modes} | noise={noise_selection}",
    )

    for n in pbar:
        phi_atm_raw, amp = get_sequence_frame(atmosphere_sequence, n, device, precision.real)
        phi_atm = phi_atm_raw * telescope_pupil
        frame_pupil = build_frame_optical_pupil(telescope_pupil, amp)

        # Open-loop physical metrics only. No extra noisy WFS call after CL closure.
        phi_open = remove_piston(phi_atm, telescope_pupil)
        psf_open = get_psf(frame_pupil, phi_open, fov)
        wfe_open[n] = get_wfe_rms(phi_open, telescope_pupil)[0].cpu()
        sr_open[n] = get_strehl(psf_open, reference_peak)[0].cpu()

        # Ideal projection on the exact model basis.
        z_ideal = zernike_decompose_torch(phi=phi_atm, zDecomposeMat=zD)
        phi_ideal_cmd = zernike_compose_torch(zernike_phi_vector=z_ideal, zComposeMat=zC)
        phi_ideal = remove_piston((phi_atm - phi_ideal_cmd) * telescope_pupil, telescope_pupil)
        psf_ideal = get_psf(frame_pupil, phi_ideal, fov)
        wfe_ideal[n] = get_wfe_rms(phi_ideal, telescope_pupil)[0].cpu()
        sr_ideal[n] = get_strehl(psf_ideal, reference_peak)[0].cpu()

        # Exactly ONE noisy NN acquisition per physical frame.
        if n < CL_SAMPLE:
            measurement_phase = phi_open
        else:
            phi_cmd_before = zernike_compose_torch(
                zernike_phi_vector=u_applied, zComposeMat=zC
            )
            measurement_phase = remove_piston(
                (phi_atm + phi_cmd_before) * telescope_pupil, telescope_pupil
            )

        z_est, I_full_measurement, net_input, _ = forward(
            frame_pupil, measurement_phase, return_boxes=False, infer_nn=True
        )
        z_est = z_est.to(device=device, dtype=precision.real)

        if int(z_est.shape[-1]) != n_modes:
            raise ValueError(f"{name}: NN returned {z_est.shape[-1]} modes, basis has {n_modes}")

        if net_hist is None and net_input.ndim == 4 and int(net_input.shape[1]) == 4:
            net_hist = torch.empty(
                (N_SAMPLES, int(net_input.shape[1]), int(net_input.shape[2]), int(net_input.shape[3])),
                dtype=precision.real,
            )
            channel_order = get_four_channel_order(WFS)

        if n < CL_SAMPLE:
            loop_closed[n] = False
            phi_current = phi_open
            psf_current = psf_open
            u_store = u_applied
            phi_cmd_store = zernike_compose_torch(zernike_phi_vector=u_store, zComposeMat=zC)
            wfe_closed[n] = wfe_open[n]
            sr_closed[n] = sr_open[n]
        else:
            loop_closed[n] = True
            u_integral = u_integral + LOOP_SIGN * KI * z_est
            if USE_INTEGRATOR_CLAMP:
                u_integral = clamp_integrator(u_integral, INTEGRATOR_LIMIT)

            u_new = u_integral + LOOP_SIGN * KP * z_est
            u_applied = u_new.detach()

            phi_cmd_new = zernike_compose_torch(zernike_phi_vector=u_new, zComposeMat=zC)
            phi_current = remove_piston(
                (phi_atm + phi_cmd_new) * telescope_pupil, telescope_pupil
            )
            psf_current = get_psf(frame_pupil, phi_current, fov)
            wfe_closed[n] = get_wfe_rms(phi_current, telescope_pupil)[0].cpu()
            sr_closed[n] = get_strehl(psf_current, reference_peak)[0].cpu()
            u_store = u_new
            phi_cmd_store = phi_cmd_new

        phi_hist[n].copy_(phi_current[0].cpu())
        psf_hist[n].copy_(psf_current[0].cpu())
        Ifull_hist[n].copy_(I_full_measurement[0].cpu())
        z_hist[n].copy_(z_est[0].cpu())
        u_hist[n].copy_(u_store[0].cpu())
        phi_cmd_hist[n].copy_(phi_cmd_store[0].cpu())

        if net_hist is not None:
            net_hist[n].copy_(net_input[0].cpu())
        if amp_hist is not None and amp is not None:
            amp_hist[n].copy_(amp[0].cpu())

        pbar.set_postfix(
            mode="CL" if n >= CL_SAMPLE else "OL",
            WFE=f"{float(wfe_closed[n]):.4f}",
            SR=f"{float(sr_closed[n]):.4f}",
        )

    integrated_start = min(CL_SAMPLE + 10, N_SAMPLES - 1)
    psf_integrated = psf_hist[integrated_start:].mean(0)

    results = {
        "psf_transition_all": psf_hist,
        "psf_integrated_closed_loop": psf_integrated,
        "psf_integrated_start": integrated_start,
        "phi_transition_all": phi_hist,
        "I_transition_all": Ifull_hist,
        "z_est_transition_all": z_hist,
        "u_command_transition_all": u_hist,
        "phi_cmd_transition_all": phi_cmd_hist,
        "network_input_transition_all": net_hist,
        "network_channel_order": channel_order,
        "atmospheric_amplitude_transition_all": amp_hist,
        "wfe_open": wfe_open,
        "wfe_closed": wfe_closed,
        "wfe_ideal": wfe_ideal,
        "strehl_open": sr_open,
        "strehl_closed": sr_closed,
        "strehl_ideal": sr_ideal,
        "loop_closed_flag": loop_closed,
        "cl_sample": CL_SAMPLE,
        "kp": KP,
        "ki": KI,
        "loop_sign": LOOP_SIGN,
        "noise_selection": noise_selection,
        "camera_runtime": camera_snapshot(camera_runtime),
        "model_info": dict(model_info),
        "basis_info": dict(basis.info),
        "basis_kind": basis.basis_kind,
        "basis_path": basis.basis_path,
        "n_modes": n_modes,
        "atmosphere_info": atmosphere_sequence.info,
        "atmosphere_sampled_r0": atmosphere_sequence.sampled_r0,
        "physical_pupil": telescope_pupil.cpu(),
    }
    torch.save(results, os.path.join(output_model_dir, "closed_loop_results.pt"))

    # Preserve the normal mode artifacts.
    save_transition_gif(phi_cmd_hist, os.path.join(output_model_dir, "dm_accumulated_command.gif"), PHASE_COLORMAP, symmetric=True)
    save_heatmap(u_hist, os.path.join(output_model_dir, "accumulated_control_heatmap.png"), f"{name} | accumulated control", CL_SAMPLE)
    save_estimation_vs_control(z_hist, u_hist, os.path.join(output_model_dir, "estimation_vs_control_plot.png"), f"{name} | estimation vs control", CL_SAMPLE)
    save_transition_gif(psf_hist, os.path.join(output_model_dir, "psf_open_to_closed_loop.gif"), PSF_COLORMAP, psf_db=60.0)
    save_transition_gif(phi_hist, os.path.join(output_model_dir, "phi_open_to_closed_loop.gif"), PHASE_COLORMAP)
    save_transition_gif(Ifull_hist, os.path.join(output_model_dir, "I_propagated_open_to_closed_loop.gif"), PROPAGATION_COLORMAP)

    if amp_hist is not None:
        save_transition_gif(amp_hist, os.path.join(output_model_dir, "atmospheric_amplitude.gif"), PROPAGATION_COLORMAP)
    if net_hist is not None:
        save_network_input_gif(net_hist, os.path.join(output_model_dir, "network_input_4pupils_exact.gif"), channel_order)

    save_heatmap(z_hist, os.path.join(output_model_dir, "nn_basis_estimation_heatmap.png"), f"{name} | NN estimation", CL_SAMPLE)
    save_single_image(
        psf_integrated,
        os.path.join(output_model_dir, "psf_integrated_closed_loop_after_cl_plus_10.png"),
        cmap=PSF_COLORMAP,
        title=f"{name} | integrated PSF from sample {integrated_start}",
        use_log=True,
    )
    save_metric_plot(wfe_open, wfe_closed, wfe_ideal, "WFE RMS [rad]", os.path.join(output_model_dir, "wfe_plot.png"), f"{name} | WFE", CL_SAMPLE)
    save_metric_plot(sr_open, sr_closed, sr_ideal, "Strehl ratio", os.path.join(output_model_dir, "strehl_plot.png"), f"{name} | Strehl", CL_SAMPLE)

    return results


def run_closed_loop_mode(
    args, models, ref_cfg, ref_precision, ref_precision_name,
    ref_resolution, ref_diameter, ref_wavelength, noise_levels,
    output_dir, propagation_enabled,
):
    atmosphere_sequence = generate_atmosphere_sequence(
        ref_cfg, ref_resolution, ref_diameter, ref_precision, args.device, output_dir
    )

    global_summary = {
        "mode": "closed_loop",
        "test_name": args.test_name,
        "device": args.device,
        "models": models,
        "noise_levels": {},
        "atmosphere_info": atmosphere_sequence.info,
    }

    for selection in noise_levels:
        noise_name = "noise_training" if selection == "training" else f"noise_{int(selection):02d}"
        noise_dir = os.path.join(output_dir, noise_name)
        os.makedirs(noise_dir, exist_ok=True)
        print(f"\n{'=' * 80}\nCLOSED LOOP | noise={selection}\n{'=' * 80}")

        result_paths = {}
        cwfe, csr, ciwfe, cisr = {}, {}, {}, {}
        ref_wfe_open = ref_sr_open = None

        for i, model_info in enumerate(models):
            name = model_info["Name"]
            print(f"\nLoading model {i + 1}/{len(models)}: {name}")
            bundle = load_model_bundle(model_info, args.device)
            validate_shared_model_compatibility(
                model_info, bundle, ref_resolution, ref_diameter,
                ref_precision_name, ref_wavelength
            )
            basis = build_model_basis_bundle(model_info, bundle, args.device, ref_precision.real)
            runtime = None if not NOISE_FLAG else build_camera_runtime(
                bundle, selection, args.device, args.noise_output_mode, args.noise_param_jitter
            )

            results = run_closed_loop_single_model(
                model_info, bundle, basis, atmosphere_sequence,
                os.path.join(noise_dir, name), runtime, args.device,
                ref_precision, selection, propagation_enabled,
                args.propagation_obstruction_px, args.propagation_spiders,
                args.propagation_spiders_px,
            )

            result_paths[name] = os.path.join(noise_dir, name, "closed_loop_results.pt")
            cwfe[name] = results["wfe_closed"]
            csr[name] = results["strehl_closed"]
            ciwfe[name] = results["wfe_ideal"]
            cisr[name] = results["strehl_ideal"]
            if ref_wfe_open is None:
                ref_wfe_open = results["wfe_open"]
                ref_sr_open = results["strehl_open"]

            del results, runtime, basis, bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_model_comparison(
            ref_wfe_open, cwfe, ciwfe, "WFE RMS [rad]",
            os.path.join(noise_dir, "wfe_compare_all_models.png"), CL_SAMPLE
        )
        save_model_comparison(
            ref_sr_open, csr, cisr, "Strehl ratio",
            os.path.join(noise_dir, "strehl_compare_all_models.png"), CL_SAMPLE
        )

        summary = {
            "noise_selection": selection,
            "results_paths": dict(result_paths),
        }
        torch.save(summary, os.path.join(noise_dir, "comparison_summary.pt"))
        global_summary["noise_levels"][str(selection)] = summary

    torch.save(
        global_summary,
        os.path.join(output_dir, "comparison_summary_all_noise_levels.pt"),
    )



# =============================================================================
# STABILITY MODE (mode 3)
# =============================================================================

def stability_sample_indices(total_frames, max_points, cl_frame):
    total_frames = int(total_frames)
    max_points = int(max_points)
    cl_frame = int(cl_frame)

    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if max_points < 4:
        raise ValueError("max_points must be >= 4")
    if total_frames <= max_points:
        return np.arange(total_frames, dtype=np.int64)

    mandatory = {0, total_frames - 1}
    if 0 <= cl_frame - 1 < total_frames:
        mandatory.add(cl_frame - 1)
    if 0 <= cl_frame < total_frames:
        mandatory.add(cl_frame)

    n_uniform = max(0, max_points - len(mandatory))
    uniform = (
        np.linspace(0, total_frames - 1, n_uniform, dtype=np.int64)
        if n_uniform > 0 else np.empty(0, dtype=np.int64)
    )
    selected = np.unique(np.concatenate([uniform, np.asarray(sorted(mandatory), dtype=np.int64)]))

    if selected.size > max_points:
        mandatory_arr = np.asarray(sorted(mandatory), dtype=np.int64)
        mandatory_set = set(mandatory_arr.tolist())
        optional = np.asarray([v for v in selected if int(v) not in mandatory_set], dtype=np.int64)
        keep = max_points - mandatory_arr.size
        if keep > 0 and optional.size > keep:
            optional = optional[np.linspace(0, optional.size - 1, keep, dtype=np.int64)]
        selected = np.unique(np.concatenate([mandatory_arr, optional]))

    return np.sort(selected)


@torch.no_grad()
def run_stability_single_model(
    model_info, bundle, basis, ref_cfg, camera_runtime, device, precision,
    total_frames, metric_indices, noise_selection,
    propagation_enabled, propagation_obstruction_px,
    propagation_spider_count, propagation_spider_width_px,
):
    name = model_info["Name"]
    WFS, NN = bundle["WFS"], bundle["NN"]
    train_cfg = bundle["train_cfg"]
    resolution = int(bundle["telescope_cfg"]["resolution"])
    fov = resolution * 4

    telescope_pupil = build_physical_pupil(
        as_4d_pupil(basis.telescope_pupil, device, precision.real),
        enabled=propagation_enabled,
        obstruction_px=propagation_obstruction_px,
        spiders=propagation_spider_count,
        spiders_px=propagation_spider_width_px,
    )

    assert_network_output_matches_basis(NN, WFS, basis, device, precision.real, name)

    zC = basis.zComposeMat
    n_modes = int(basis.n_modes)

    phi_ref = torch.zeros((1, 1, resolution, resolution), device=device, dtype=precision.real)
    reference_peak = float(get_psf(telescope_pupil, phi_ref, fov).max().item())

    forward = build_forward_pipe(WFS, NN, camera_runtime, train_cfg["norm_type"], NOISE_FLAG)

    # Streaming atmosphere: no temporal cube is allocated.
    atm = build_atmosphere(
        ref_cfg,
        resolution,
        float(bundle["telescope_cfg"]["diameter"]),
        precision,
        device,
    )
    phase = atm.gen(seed=int(SEED))
    sampled_r0 = atm.r0_batch.detach().cpu().clone()

    metric_indices = np.asarray(metric_indices, dtype=np.int64)
    n_metric = int(metric_indices.size)

    metric_frame = np.empty(n_metric, dtype=np.int64)
    metric_time_min = np.empty(n_metric, dtype=np.float64)
    metric_wfe = np.empty(n_metric, dtype=np.float32)
    metric_sr = np.empty(n_metric, dtype=np.float32)

    metric_ptr = 0
    next_metric = int(metric_indices[0]) if n_metric else None

    u_integral = torch.zeros((1, n_modes), device=device, dtype=precision.real)
    u_applied = torch.zeros_like(u_integral)

    last_wfe = None
    last_sr = None
    refresh = max(1, total_frames // 1000)

    pbar = tqdm(
        range(total_frames),
        desc=f"Stability | {name} | {basis.basis_kind} | noise={noise_selection}",
    )

    for n in pbar:
        if n > 0:
            phase = atm.update()

        phi_atm = phase[0:1, 0:1] * telescope_pupil
        amp = atm.field.abs()[0:1, 0:1] if ATMOSPHERE_SCINTILLATION else None
        frame_pupil = build_frame_optical_pupil(telescope_pupil, amp)

        if n < CL_SAMPLE:
            measurement_phase = remove_piston(phi_atm, telescope_pupil)
        else:
            phi_cmd_before = zernike_compose_torch(
                zernike_phi_vector=u_applied, zComposeMat=zC
            )
            measurement_phase = remove_piston(
                (phi_atm + phi_cmd_before) * telescope_pupil, telescope_pupil
            )

        # Exactly one noisy WFS/NN acquisition at every physical frame.
        z_est, _, _, _ = forward(
            frame_pupil, measurement_phase, return_boxes=False, infer_nn=True
        )
        z_est = z_est.to(device=device, dtype=precision.real)

        if int(z_est.shape[-1]) != n_modes:
            raise ValueError(f"{name}: NN output/basis mismatch")

        phi_metric = None

        if n < CL_SAMPLE:
            if next_metric is not None and n == next_metric:
                phi_metric = measurement_phase
        else:
            u_integral = u_integral + LOOP_SIGN * KI * z_est
            if USE_INTEGRATOR_CLAMP:
                u_integral = clamp_integrator(u_integral, INTEGRATOR_LIMIT)

            u_new = u_integral + LOOP_SIGN * KP * z_est
            u_applied = u_new.detach()

            if next_metric is not None and n == next_metric:
                phi_cmd_new = zernike_compose_torch(
                    zernike_phi_vector=u_new, zComposeMat=zC
                )
                phi_metric = remove_piston(
                    (phi_atm + phi_cmd_new) * telescope_pupil, telescope_pupil
                )

        if next_metric is not None and n == next_metric:
            wfe = float(get_wfe_rms(phi_metric, telescope_pupil)[0].item())
            psf = get_psf(frame_pupil, phi_metric, fov)
            sr = float(get_strehl(psf, reference_peak)[0].item())

            metric_frame[metric_ptr] = n
            metric_time_min[metric_ptr] = float(n) / float(FRAME_RATE) / 60.0
            metric_wfe[metric_ptr] = wfe
            metric_sr[metric_ptr] = sr
            last_wfe, last_sr = wfe, sr

            metric_ptr += 1
            next_metric = int(metric_indices[metric_ptr]) if metric_ptr < n_metric else None

        if n == 0 or (n + 1) % refresh == 0 or n == total_frames - 1:
            post = {
                "frame": n + 1,
                "time_min": f"{float(n) / FRAME_RATE / 60.0:.3f}",
                "mode": "CL" if n >= CL_SAMPLE else "OL",
            }
            if last_wfe is not None:
                post["WFE"] = f"{last_wfe:.4f}"
                post["SR"] = f"{last_sr:.4f}"
            pbar.set_postfix(post)

    if metric_ptr != n_metric:
        raise RuntimeError(f"Expected {n_metric} metrics, stored {metric_ptr}")

    return {
        "sample_indices": torch.from_numpy(metric_frame.copy()),
        "time_minutes": torch.from_numpy(metric_time_min.copy()),
        "wfe_rms_rad": torch.from_numpy(metric_wfe.copy()),
        "strehl_ratio": torch.from_numpy(metric_sr.copy()),
        "sampled_r0": sampled_r0,
        "atmosphere_info": atm.info(),
        "n_total_frames": int(total_frames),
        "duration_minutes": float(total_frames) / float(FRAME_RATE) / 60.0,
        "frame_rate_hz": float(FRAME_RATE),
        "cl_sample": int(CL_SAMPLE),
        "cl_time_minutes": float(CL_SAMPLE) / float(FRAME_RATE) / 60.0,
        "kp": float(KP),
        "ki": float(KI),
        "loop_sign": float(LOOP_SIGN),
        "noise_selection": noise_selection,
        "camera_runtime": camera_snapshot(camera_runtime),
        "physical_pupil_mode": PHYSICAL_PUPIL_MODE,
        "basis_info": dict(basis.info),
        "model_info": dict(model_info),
    }


def save_stability_plot(results_by_model, metric_key, ylabel, out_path, title, cl_time_minutes):
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, result in results_by_model.items():
        x = result["time_minutes"].numpy()
        y = result[metric_key].numpy()
        ax.plot(x, y, linewidth=1.5, label=name)

    ax.axvline(cl_time_minutes, linestyle="--", linewidth=1.2, label="Loop closes")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_stability_mode(
    args, models, ref_cfg, ref_precision, ref_precision_name,
    ref_resolution, ref_diameter, ref_wavelength, noise_levels,
    output_dir, propagation_enabled,
):
    duration = float(args.duration_minutes)
    max_points = int(args.max_points)

    if duration <= 0:
        raise ValueError("duration_minutes must be > 0")
    if FRAME_RATE <= 0:
        raise ValueError("FRAME_RATE must be > 0")

    total_frames = max(1, int(round(duration * 60.0 * float(FRAME_RATE))))
    if CL_SAMPLE >= total_frames:
        raise ValueError("CL_SAMPLE must be < total_frames")

    metric_indices = stability_sample_indices(total_frames, max_points, CL_SAMPLE)
    actual_duration = float(total_frames) / float(FRAME_RATE) / 60.0
    cl_time = float(CL_SAMPLE) / float(FRAME_RATE) / 60.0

    print("\n" + "=" * 80)
    print("LONG-DURATION STABILITY")
    print("=" * 80)
    print(f"Frame rate       : {FRAME_RATE:.3f} Hz")
    print(f"Duration         : {actual_duration:.6f} min")
    print(f"Total frames     : {total_frames}")
    print(f"Metric points    : {len(metric_indices)}")
    print(f"Loop closes      : frame {CL_SAMPLE} ({cl_time:.6f} min)")
    print("=" * 80)

    global_summary = {
        "mode": "stability",
        "test_name": args.test_name,
        "device": args.device,
        "frame_rate_hz": float(FRAME_RATE),
        "duration_minutes": actual_duration,
        "total_frames": int(total_frames),
        "metric_sample_indices": torch.from_numpy(metric_indices.copy()),
        "models": models,
        "noise_levels": {},
    }

    for selection in noise_levels:
        noise_name = "noise_training" if selection == "training" else f"noise_{int(selection):02d}"
        noise_dir = os.path.join(output_dir, noise_name)
        os.makedirs(noise_dir, exist_ok=True)
        print(f"\nSTABILITY | noise={selection}")

        results_by_model = {}

        for i, model_info in enumerate(models):
            name = model_info["Name"]
            print(f"Loading model {i + 1}/{len(models)}: {name}")

            bundle = load_model_bundle(model_info, args.device)
            validate_shared_model_compatibility(
                model_info, bundle, ref_resolution, ref_diameter,
                ref_precision_name, ref_wavelength
            )
            basis = build_model_basis_bundle(model_info, bundle, args.device, ref_precision.real)
            runtime = None if not NOISE_FLAG else build_camera_runtime(
                bundle, selection, args.device, args.noise_output_mode, args.noise_param_jitter
            )

            result = run_stability_single_model(
                model_info, bundle, basis, ref_cfg, runtime, args.device,
                ref_precision, total_frames, metric_indices, selection,
                propagation_enabled, args.propagation_obstruction_px,
                args.propagation_spiders, args.propagation_spiders_px,
            )

            model_dir = os.path.join(noise_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(result, os.path.join(model_dir, "stability_metrics.pt"))
            results_by_model[name] = result

            del runtime, basis, bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        wfe_path = os.path.join(noise_dir, "wfe_stability_minutes.png")
        sr_path = os.path.join(noise_dir, "strehl_stability_minutes.png")

        save_stability_plot(
            results_by_model, "wfe_rms_rad", "WFE RMS [rad]",
            wfe_path, f"Closed-loop stability | WFE | noise={selection}", cl_time
        )
        save_stability_plot(
            results_by_model, "strehl_ratio", "Strehl ratio",
            sr_path, f"Closed-loop stability | Strehl | noise={selection}", cl_time
        )

        summary = {
            "noise_selection": selection,
            "wfe_plot": wfe_path,
            "strehl_plot": sr_path,
            "model_metric_paths": {
                name: os.path.join(noise_dir, name, "stability_metrics.pt")
                for name in results_by_model
            },
        }
        torch.save(summary, os.path.join(noise_dir, "stability_summary.pt"))
        global_summary["noise_levels"][str(selection)] = summary

        del results_by_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(
        global_summary,
        os.path.join(output_dir, "stability_summary_all_noise_levels.pt"),
    )


# =============================================================================
# MAIN
# =============================================================================

def load_reference_configuration(models):
    if not models:
        raise ValueError("Model list is empty")

    first = models[0]
    train_path = Path(first["train_path"])
    stage = first.get("stage", 0)
    stage_idx = 0 if stage is None else int(stage)

    cfg = safe_load(train_path / "config" / "all_cfg.pt", "cpu")
    tel = cfg["telescope_cfg"]
    train_cfg = cfg["stages_cfg"][stage_idx]["train"]
    runtime_cfg = cfg.get("runtime_cfg", None)

    if runtime_cfg is not None:
        precision_name = str(runtime_cfg["precision"]).lower()
    else:
        precision_name = str(train_cfg["precision"]).lower()

    precision = get_precision(precision_name)
    return (
        cfg,
        precision,
        precision_name,
        int(tel["resolution"]),
        float(tel["diameter"]),
        source_wavelength_from_cfg(cfg),
    )


def main():
    args = parse_args()
    mode = str(args.mode).lower()

    if args.test_name is None:
        args.test_name = (
            DEFAULT_TEST_NAME_CLOSED_LOOP
            if mode == "closed_loop"
            else DEFAULT_TEST_NAME_STABILITY
        )

    if mode == "closed_loop":
        models = MODELS_TO_TEST_CLOSED_LOOP
        output_dir = os.path.join("./TEST/CL", args.test_name)
    else:
        models = MODELS_TO_TEST_STABILITY
        output_dir = os.path.join("./TEST/CL_STABILITY", args.test_name)

    os.makedirs(output_dir, exist_ok=True)

    (
        ref_cfg, ref_precision, ref_precision_name,
        ref_resolution, ref_diameter, ref_wavelength,
    ) = load_reference_configuration(models)

    noise_levels = parse_noise_levels(args.noise_level)
    propagation_enabled = (
        USE_TELESCOPE_PUPIL_IN_PROPAGATION
        and not args.disable_propagation_obstruction
    )

    print("\n" + "=" * 80)
    print("UNIFIED DEEP PYRAMID WFS CLOSED LOOP")
    print("=" * 80)
    print(f"Mode             : {mode}")
    print(f"Device           : {args.device}")
    print(f"Test             : {args.test_name}")
    print(f"Precision        : {ref_precision_name}")
    print(f"Resolution       : {ref_resolution}")
    print(f"Diameter         : {ref_diameter}")
    print(f"Wavelength       : {ref_wavelength:.6e} m")
    print(f"Noise selection  : {noise_levels}")
    print(f"Camera domain    : {CAMERA_NOISE_DOMAIN}")
    print(f"Physical pupil   : {PHYSICAL_PUPIL_MODE}")
    print("=" * 80)

    if mode == "closed_loop":
        run_closed_loop_mode(
            args, models, ref_cfg, ref_precision, ref_precision_name,
            ref_resolution, ref_diameter, ref_wavelength, noise_levels,
            output_dir, propagation_enabled,
        )
    else:
        run_stability_mode(
            args, models, ref_cfg, ref_precision, ref_precision_name,
            ref_resolution, ref_diameter, ref_wavelength, noise_levels,
            output_dir, propagation_enabled,
        )

    print("\nFinished successfully.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
