# 1-main.py
from __future__ import annotations
import argparse
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ATMOSPHERE.atmosphere import Atmosphere
from MODAL_BASIS.Zernike import *
from GENERAL_FUNCTIONS.camera_noise import *
from GENERAL_FUNCTIONS.functions_torch import *
from NN.model_manager import ModelManager
from PYRAMID.Pyramid_WFS import Pyramid
from PYRAMID.functions_pyr_torch import *
from CONFIG.config import get_config, save_run_info
from GENERAL_FUNCTIONS.losses import CoefLoss


@dataclass(frozen=True)
class AtmosphereBatchProfile:
    """Batch-shared parameters plus the per-realization r0 interval."""

    dr0_range: Tuple[float, float]
    r0: Tuple[float, float]
    fractional_r0: Tuple[float, ...]
    wind_speed: Tuple[float, ...]
    wind_direction: Tuple[float, ...]
    altitude: Tuple[float, ...]


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_curve(
    stage_path: Path,
    *,
    train_open,
    val_open,
    train_last,
    val_last,
    title: str,
) -> None:
    stage_path.mkdir(parents=True, exist_ok=True)
    figure_path = stage_path / "loss_curve.png"

    plt.figure(figsize=(9, 5))
    plt.plot(train_open, label="train_open_std")
    plt.plot(val_open, label="val_open_std")
    plt.plot(train_last, label="train_last_std")
    plt.plot(val_last, label="val_last_std")
    plt.xlabel("epoch")
    plt.ylabel("phase STD [rad]")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

def _get_wfs_display_geometry(
    I_full: torch.Tensor,
    boxes,
    network_resolution: int,
):
    """
    Calcula el escalado necesario para que cada bbox tenga
    network_resolution x network_resolution en la representación
    escalada.

    Ejemplo:
        full frame = 512x512
        crop real  = 148x148
        NN         = 36x36

        scale = 36 / 148
    """

    if len(boxes) == 0:
        raise ValueError(
            "boxes no puede estar vacío."
        )

    H = int(I_full.shape[-2])
    W = int(I_full.shape[-1])

    x0, y0, x1, y1 = boxes[0]

    crop_w = float(x1 - x0)
    crop_h = float(y1 - y0)

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(
            "Bounding box inválido."
        )

    # Todos los crops actualmente tienen el mismo tamaño.
    # Lo validamos para evitar errores silenciosos.
    for box in boxes:

        bx0, by0, bx1, by1 = box

        this_w = float(bx1 - bx0)
        this_h = float(by1 - by0)

        if this_w != crop_w or this_h != crop_h:
            raise ValueError(
                "Todos los bounding boxes deben tener "
                "el mismo tamaño para usar un único escalado."
            )

    scale_x = (
        float(network_resolution)
        / crop_w
    )

    scale_y = (
        float(network_resolution)
        / crop_h
    )

    display_w = max(
        1,
        int(round(W * scale_x)),
    )

    display_h = max(
        1,
        int(round(H * scale_y)),
    )

    return {
        "H": H,
        "W": W,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "display_h": display_h,
        "display_w": display_w,
    }


def _plot_wfs_with_boxes(
    ax,
    I_full: torch.Tensor,
    boxes,
    *,
    network_resolution: int,
    title: str,
):
    """
    Escala el full frame con el mismo factor con que cada crop
    termina en la resolución de entrada de la NN.

    Los boxes se dibujan en el orden EXACTO de los canales
    de entrada de la red.
    """

    geometry = _get_wfs_display_geometry(
        I_full,
        boxes,
        network_resolution,
    )

    display_h = geometry["display_h"]
    display_w = geometry["display_w"]

    scale_x = geometry["scale_x"]
    scale_y = geometry["scale_y"]

    # ============================================================
    # Escalar full frame
    # ============================================================

    I_scaled = F.interpolate(
        I_full.float(),
        size=(
            display_h,
            display_w,
        ),
        mode="bilinear",
        align_corners=False,
    )

    img = (
        I_scaled[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # ============================================================
    # Mostrar full frame escalado
    # ============================================================

    ax.imshow(
        img,
        origin="upper",
    )

    # ============================================================
    # Bounding boxes
    # ============================================================

    for idx, box in enumerate(boxes):

        x0, y0, x1, y1 = box

        x0 = x0 * scale_x
        x1 = x1 * scale_x

        y0 = y0 * scale_y
        y1 = y1 * scale_y

        width = x1 - x0
        height = y1 - y0

        rectangle = Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            linewidth=2.0,
        )

        ax.add_patch(rectangle)

        # Número = canal de entrada NN
        ax.text(
            x0 + 1,
            y0 + 4,
            f"{idx}",
            fontsize=10,
            bbox=dict(
                alpha=0.75,
                edgecolor="none",
            ),
        )

    ax.set_title(title)
    ax.axis("off")

    return I_scaled

def _save_debug_val_open_last_plot(
    stage_path: Path,
    *,
    open_phi: torch.Tensor,
    open_amplitude: torch.Tensor,
    open_I: torch.Tensor,
    open_boxes,
    open_pred: torch.Tensor,
    last_phi: torch.Tensor,
    last_amplitude: torch.Tensor,
    last_I: torch.Tensor,
    last_boxes,
    last_pred: torch.Tensor,
    network_resolution: int,
    stage_idx: int,
    filename: str = "val_open_last_debug.png",
) -> None:

    stage_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        stage_path
        / filename
    )

    # ============================================================
    # CPU
    # ============================================================

    open_phi = (
        open_phi
        .detach()
        .float()
        .cpu()
    )

    open_amplitude = (
        open_amplitude
        .detach()
        .float()
        .cpu()
    )

    open_I = (
        open_I
        .detach()
        .float()
        .cpu()
    )

    open_pred = (
        open_pred
        .detach()
        .float()
        .cpu()
    )

    last_phi = (
        last_phi
        .detach()
        .float()
        .cpu()
    )

    last_amplitude = (
        last_amplitude
        .detach()
        .float()
        .cpu()
    )

    last_I = (
        last_I
        .detach()
        .float()
        .cpu()
    )

    last_pred = (
        last_pred
        .detach()
        .float()
        .cpu()
    )

    # ============================================================
    # Figure
    # ============================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(19, 8),
        dpi=160,
    )

    fig.suptitle(
        f"Stage {stage_idx} - online atmosphere validation"
    )

    # ============================================================
    # OPEN LOOP
    # ============================================================

    axes[0, 0].set_title(
        "OPEN LOOP | phi target"
    )

    axes[0, 0].imshow(
        open_phi[0, 0].numpy()
    )

    axes[0, 0].axis("off")

    # ------------------------------------------------------------

    axes[0, 1].set_title(
        "OPEN LOOP | atmospheric amplitude"
    )

    axes[0, 1].imshow(
        open_amplitude[0, 0].numpy()
    )

    axes[0, 1].axis("off")

    # ------------------------------------------------------------
    # FULL WFS + BBOXES
    # ------------------------------------------------------------

    _plot_wfs_with_boxes(
        axes[0, 2],
        open_I,
        open_boxes,
        network_resolution=network_resolution,
        title="OPEN LOOP | WFS + NN crops",
    )

    # ------------------------------------------------------------

    axes[0, 3].set_title(
        "OPEN LOOP | predicted coefficients"
    )

    axes[0, 3].plot(
        open_pred[0].numpy(),
        label="Prediction",
    )

    axes[0, 3].set_xlabel(
        "mode / actuator"
    )

    axes[0, 3].set_ylabel(
        "coefficient"
    )

    axes[0, 3].grid(
        True,
        alpha=0.3,
    )

    axes[0, 3].legend()

    # ============================================================
    # CLOSED LOOP
    # ============================================================

    axes[1, 0].set_title(
        "CLOSED LOOP | residual phi"
    )

    axes[1, 0].imshow(
        last_phi[0, 0].numpy()
    )

    axes[1, 0].axis("off")

    # ------------------------------------------------------------

    axes[1, 1].set_title(
        "CLOSED LOOP | atmospheric amplitude"
    )

    axes[1, 1].imshow(
        last_amplitude[0, 0].numpy()
    )

    axes[1, 1].axis("off")

    # ------------------------------------------------------------
    # FULL WFS + BBOXES
    # ------------------------------------------------------------

    _plot_wfs_with_boxes(
        axes[1, 2],
        last_I,
        last_boxes,
        network_resolution=network_resolution,
        title="CLOSED LOOP | WFS + NN crops",
    )

    # ------------------------------------------------------------

    axes[1, 3].set_title(
        "CLOSED LOOP | predicted residual"
    )

    axes[1, 3].plot(
        last_pred[0].numpy(),
        label="Prediction",
    )

    axes[1, 3].set_xlabel(
        "mode / actuator"
    )

    axes[1, 3].set_ylabel(
        "coefficient"
    )

    axes[1, 3].grid(
        True,
        alpha=0.3,
    )

    axes[1, 3].legend()

    # ============================================================
    # Save
    # ============================================================

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

def _uniform_scalar(
    generator: torch.Generator,
    low: float,
    high: float,
) -> float:
    low = float(low)
    high = float(high)

    if high < low:
        raise ValueError(f"Invalid interval [{low}, {high}].")
    if high == low:
        return low

    random_value = torch.rand(
        (),
        generator=generator,
        dtype=torch.float64,
    ).item()
    return low + (high - low) * random_value


def _sample_stratified_layer_values(
    interval: Tuple[float, float],
    n_layers: int,
    generator: torch.Generator,
) -> Tuple[float, ...]:
    """Divide one global interval into N layer intervals and sample each.

    Example for [0, 15] and three layers:
        layer 0 -> U(0, 5)
        layer 1 -> U(5, 10)
        layer 2 -> U(10, 15)
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    low, high = map(float, interval)
    if high < low:
        raise ValueError(f"Invalid interval [{low}, {high}].")

    step = (high - low) / n_layers
    values = []

    for layer_index in range(n_layers):
        layer_low = low + layer_index * step
        layer_high = (
            high
            if layer_index == n_layers - 1
            else low + (layer_index + 1) * step
        )
        values.append(
            _uniform_scalar(
                generator,
                layer_low,
                layer_high,
            )
        )

    return tuple(values)


def _sample_atmosphere_profile(
    *,
    atmosphere_cfg,
    telescope_diameter: float,
    seed: int,
) -> AtmosphereBatchProfile:
    """Sample one batch-shared physical profile deterministically."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    # Keep the old dataset distribution exactly: r0 itself is sampled
    # uniformly. Atmosphere receives [r0_min, r0_max] and draws one value per
    # realization when gen() is called. Wind/altitude/direction remain shared
    # inside the batch and are sampled only once here for the next batch.
    dr0_min, dr0_max = map(float, atmosphere_cfg.dr0_range)
    r0 = (
        float(telescope_diameter) / dr0_max,
        float(telescope_diameter) / dr0_min,
    )

    fractional_r0 = tuple(
        float(value) for value in atmosphere_cfg.fractional_r0
    )
    n_layers = len(fractional_r0)

    wind_speed = _sample_stratified_layer_values(
        atmosphere_cfg.wind_speed_range,
        n_layers,
        generator,
    )
    wind_direction = _sample_stratified_layer_values(
        atmosphere_cfg.wind_direction_range,
        n_layers,
        generator,
    )
    altitude = _sample_stratified_layer_values(
        atmosphere_cfg.altitude_range,
        n_layers,
        generator,
    )

    return AtmosphereBatchProfile(
        dr0_range=(dr0_min, dr0_max),
        r0=r0,
        fractional_r0=fractional_r0,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        altitude=altitude,
    )


def _atmosphere_to_network_batch(
    value: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Convert Atmosphere [1,B,H,W] output into WFS [B,1,H,W]."""
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(
            f"Atmosphere {name} must have shape [1,B,H,W]. "
            f"Received {tuple(value.shape)}."
        )
    return value.permute(1, 0, 2, 3).contiguous()


def _configure_atmosphere(
    atmosphere: Atmosphere,
    profile: AtmosphereBatchProfile,
    *,
    realization_seed: int,
) -> torch.Tensor:
    """Configure shared batch physics and generate frame zero once.

    ``profile.r0`` is [min,max], so Atmosphere samples one independent r0 per
    realization while wind/direction/altitude are shared by the batch.
    """
    return atmosphere.configure_profile(
        r0=profile.r0,
        wind_speed=profile.wind_speed,
        wind_direction=profile.wind_direction,
        altitude=profile.altitude,
        fractional_r0=profile.fractional_r0,
        seed=int(realization_seed),
    )


def _prepare_atmosphere_frame(
    *,
    atmosphere: Atmosphere,
    phase: torch.Tensor,
    telescope_pupil: torch.Tensor,
    scintillation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare phase, atmospheric amplitude and effective WFS pupil.

    The atmospheric complex field is

        E_atm = A_atm * exp(1j * phi_atm)

    Therefore scintillation modifies the WFS input amplitude, not its phase:

        pupil_effective = pupil_telescope * A_atm

    The closed-loop correction remains purely phase-only.
    """
    phase_batch = _atmosphere_to_network_batch(
        phase,
        name="phase",
    )
    phase_batch = phase_batch * telescope_pupil

    if scintillation:
        amplitude = _atmosphere_to_network_batch(
            atmosphere.field.abs(),
            name="amplitude",
        )
    else:
        amplitude = torch.ones_like(phase_batch)

    effective_pupil = telescope_pupil * amplitude

    return phase_batch, amplitude, effective_pupil


def _seed_global_torch(seed: int, device: str) -> None:
    """Make noise augmentation and random loop gain reproducible."""
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _online_batch_counts(
    *,
    n_samples: int,
    train_fraction: float,
    batch_size: int,
) -> Tuple[int, int, int, int]:
    """Return train/val batch counts and actual generated sample counts."""
    train_target = max(1, int(round(n_samples * train_fraction)))
    val_target = max(1, n_samples - train_target)

    train_batches = max(1, math.ceil(train_target / batch_size))
    val_batches = max(1, math.ceil(val_target / batch_size))

    return (
        train_batches,
        val_batches,
        train_batches * batch_size,
        val_batches * batch_size,
    )


def _build_atmosphere(
    *,
    initial_profile: AtmosphereBatchProfile,
    atmosphere_cfg,
    telescope_cfg,
    source_cfg,
    train_cfg,
    precision,
    device: str,
) -> Atmosphere:
    return Atmosphere(
        batch_size=train_cfg.batch_size,
        resolution=telescope_cfg.resolution,
        telescope_diameter=telescope_cfg.diameter,
        frame_rate=atmosphere_cfg.frame_rate,
        r0=initial_profile.r0,
        L0=atmosphere_cfg.L0,
        l0=atmosphere_cfg.l0,
        wind_speed=tuple(reversed(initial_profile.wind_speed)),
        wind_direction=tuple(reversed(initial_profile.wind_direction)),
        altitude=tuple(reversed(initial_profile.altitude)),
        fractional_r0=tuple(reversed(initial_profile.fractional_r0)),
        n_subharmonic_levels=(
            atmosphere_cfg.n_subharmonic_levels
        ),
        subharmonic_mode=atmosphere_cfg.subharmonic_mode,
        direction_convention=(
            atmosphere_cfg.direction_convention
        ),
        normalize_fractional_r0=(
            atmosphere_cfg.normalize_fractional_r0
        ),
        remove_piston=atmosphere_cfg.remove_piston,
        store_components=atmosphere_cfg.store_components,
        wavelength=source_cfg.wavelength,
        r0_reference_wavelength=(
            atmosphere_cfg.r0_reference_wavelength
        ),
        propagation_mode=atmosphere_cfg.propagation_mode,
        delta_mode=atmosphere_cfg.delta_mode,
        asm_extra_pixels=atmosphere_cfg.asm_extra_pixels,
        asm_min_physical_margin=(
            atmosphere_cfg.asm_min_physical_margin
        ),
        asm_padding_factor=atmosphere_cfg.asm_padding_factor,
        delta_wrap_warning_threshold=(
            atmosphere_cfg.delta_wrap_warning_threshold
        ),
        frozen_flow_mode=atmosphere_cfg.frozen_flow_mode,
        temporal_reanchor_interval=(
            atmosphere_cfg.temporal_reanchor_interval
        ),
        dtype=precision.real,
        device=device,
        seed=atmosphere_cfg.seed,
    )


def _run_open_closed_loop_batch(
    *,
    atmosphere: Atmosphere,
    initial_phase: torch.Tensor,
    realization_seed: int,
    scintillation: bool,
    WFS,
    NN,
    coef_loss,
    zComposeMat: torch.Tensor,
    telescope_pupil: torch.Tensor,
    train_cfg,
    noise_pipe,
    device: str,
):
    """Process one already generated online atmosphere batch.

    ``configure_profile(..., seed=...)`` has already generated frame zero.
    This function uses that returned phase directly. Every later atmospheric
    frame is produced only with ``atmosphere.update()``.
    """
    _seed_global_torch(realization_seed, device)

    # -------------------------------------------------
    # STEP 0: OPEN LOOP — use the existing frame zero
    # -------------------------------------------------
    phi0, amplitude0, effective_pupil0 = _prepare_atmosphere_frame(
        atmosphere=atmosphere,
        phase=initial_phase,
        telescope_pupil=telescope_pupil,
        scintillation=scintillation,
    )

    intensity, open_boxes = WFS.propagate(
        phi=phi0,
        pupil=effective_pupil0,
        return_boxes=True,
    )
    if train_cfg.noise:
        intensity = noise_pipe(intensity)
    intensity = norm_I(intensity, train_cfg.norm_type)

    prediction = NN(intensity)
    loss0, logs0 = coef_loss(
        pred=prediction,
        target=None,
        phi_target=phi0,
    )

    total_loss = loss0
    step_logs = [logs0]

    batch_size = int(phi0.shape[0])
    gain_low, gain_high = train_cfg.cl_gain_range

    gain_generator = torch.Generator(device=phi0.device)
    gain_generator.manual_seed(int(realization_seed) + 17)

    loop_gain = torch.empty(
        batch_size,
        1,
        device=phi0.device,
        dtype=phi0.dtype,
    ).uniform_(
        gain_low,
        gain_high,
        generator=gain_generator,
    )

    with torch.no_grad():
        command = loop_gain * prediction.detach()

    open_debug = (
        phi0,
        amplitude0,
        effective_pupil0,
        intensity,
        prediction,
        open_boxes,
    )
    last_debug = open_debug

    # -------------------------------------------------
    # TEMPORAL CLOSED LOOP
    # -------------------------------------------------
    for _ in range(int(train_cfg.cl_iter)):
        with torch.no_grad():
            (
                phi_atmosphere,
                amplitude_atmosphere,
                effective_pupil_atmosphere,
            ) = _prepare_atmosphere_frame(
                atmosphere=atmosphere,
                phase=atmosphere.update(),
                telescope_pupil=telescope_pupil,
                scintillation=scintillation,
            )

            phi_correction = zernike_compose_torch(
                zernike_phi_vector=command,
                zComposeMat=zComposeMat,
            )
            phi_state = (
                phi_atmosphere - phi_correction
            ) * telescope_pupil

        intensity_closed, closed_boxes = WFS.propagate(
            phi=phi_state,
            pupil=effective_pupil_atmosphere,
            return_boxes=True,
        )
        if train_cfg.noise:
            intensity_closed = noise_pipe(intensity_closed)
        intensity_closed = norm_I(
            intensity_closed,
            train_cfg.norm_type,
        )

        residual_prediction = NN(intensity_closed)
        loss_closed, logs_closed = coef_loss(
            pred=residual_prediction,
            target=None,
            phi_target=phi_state,
        )

        total_loss = total_loss + loss_closed
        step_logs.append(logs_closed)

        with torch.no_grad():
            command = (
                command
                + loop_gain * residual_prediction.detach()
            )

        last_debug = (
            phi_state,
            amplitude_atmosphere,
            effective_pupil_atmosphere,
            intensity_closed,
            residual_prediction,
            closed_boxes,
        )

    total_loss = total_loss / (int(train_cfg.cl_iter) + 1)

    return {
        "total_loss": total_loss,
        "step_logs": step_logs,
        "open_debug": open_debug,
        "last_debug": last_debug,
    }

def _build_telescope_pupil(
    *,
    telescope_cfg,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:

    resolution = int(
        telescope_cfg.resolution
    )

    # Centro geométrico de la pupila
    cx = (
        resolution - 1
    ) / 2.0

    cy = (
        resolution - 1
    ) / 2.0

    # Offset de la obstrucción
    dx, dy = (
        telescope_cfg.central_obstruction_offset_px
    )

    obstruction_center = (
        cx + float(dx),
        cy + float(dy),
    )

    pupil = circular_pupil_telescope(
        n=resolution,

        spiders=telescope_cfg.spiders,

        spiders_px=(
            telescope_cfg.spiders_px
        ),

        spider_angles_deg=(
            telescope_cfg.spider_angles_deg
        ),

        central_obstruction_diam_px=(
            telescope_cfg.central_obstruction_px
        ),

        central_obstruction_center_px=(
            obstruction_center
        ),

        device=device,
        dtype=dtype,
        soft_edge_px=0.0,
    )

    return pupil

def _as_4d_pupil(
    pupil,
    *,
    device,
    dtype,
    resolution,
):

    if not torch.is_tensor(pupil):
        pupil = torch.as_tensor(pupil)

    pupil = pupil.to(
        device=device,
        dtype=dtype,
    )

    if pupil.ndim == 2:
        pupil = pupil[
            None,
            None,
            :,
            :
        ]

    elif pupil.ndim == 3:

        if pupil.shape[0] != 1:
            raise ValueError(
                "Pupila 3-D debe tener shape [1,H,W]."
            )

        pupil = pupil.unsqueeze(0)

    elif pupil.ndim != 4:
        raise ValueError(
            "La pupila debe ser 2-D, 3-D o 4-D."
        )

    expected_shape = (
        1,
        1,
        resolution,
        resolution,
    )

    if tuple(pupil.shape) != expected_shape:
        raise ValueError(
            f"Pupila esperada {expected_shape}, "
            f"recibida {tuple(pupil.shape)}."
        )

    return pupil.contiguous()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online atmospheric training"
    )
    parser.add_argument(
        "--expName",
        default="DEBUG",
        type=str,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    arguments = parser.parse_args()

    cfg = get_config()
    source_cfg = cfg.source
    telescope_cfg = cfg.telescope
    wfs_cfg = cfg.wfs
    model_cfg = cfg.model
    stages_cfg = cfg.stages

    phase_rad_to_wfe_nm = (
        source_cfg.wavelength
        * 1e9
        / (2.0 * math.pi)
    )

    device = arguments.device
    experiment_name = arguments.expName
    pid = os.getpid()

    precision = get_precision(stages_cfg[0].train.precision)

    # ============================================================
    # PUPILAS
    # ============================================================

    # Pupila física definida por el telescopio:
    # obstrucción + offset + spiders + ángulos.
    configured_telescope_pupil = _build_telescope_pupil(
        telescope_cfg=telescope_cfg,
        device=device,
        dtype=precision.real,
    )

    # Pupila circular ideal.
    # Se usa exclusivamente para definir los Zernikes clásicos.
    ideal_telescope_pupil = circular_pupil(
        n=telescope_cfg.resolution,
        device=device,
        dtype=precision.real,
        soft_edge_px=0.0,
    )

    # ============================================================
    # ZERNIKE OR ACTUATOR BASIS
    # ============================================================

    dm_basis = stages_cfg[0].atmosphere.dm_basis
    dm_basis_type = stages_cfg[0].atmosphere.dm_basis_type
    dm_name = stages_cfg[0].atmosphere.dm_name

    if dm_basis:

        basis_root = Path(
            "/data2/rmunoz/DEEP_WFS/DEEP_PYRAMID_WFS/"
            "MODAL_BASIS/DEFORMABLE_MIRROR_BASIS"
        )

        basis_directory = (
            basis_root
            / dm_name
        )

        if dm_basis_type == "ZERNIKE":

            print("DM ZERNIKE BASIS TYPE")

            dm_basis_path = (
                basis_directory
                / (
                    f"ZERNIKE_BASIS_RES_"
                    f"{telescope_cfg.resolution}.pt"
                )
            )

        else:

            print("DM ACTUATOR BASIS TYPE")

            dm_basis_path = (
                basis_directory
                / (
                    f"ACTUATOR_BASIS_RES_"
                    f"{telescope_cfg.resolution}.pt"
                )
            )

        dm_data = torch.load(
            dm_basis_path,
            map_location="cpu",
        )

        # --------------------------------------------------------
        # Base DM ORIGINAL
        # --------------------------------------------------------

        zComposeMat = dm_data[
            "zComposeMat"
        ]

        zDecomposeMat = dm_data[
            "zDecomposeMat"
        ]

        # --------------------------------------------------------
        # Pupila propia de la base DM
        # --------------------------------------------------------

        if "telescope_pupil" not in dm_data:
            raise KeyError(
                "La base DM debe contener 'telescope_pupil'."
            )

        dm_pupil = _as_4d_pupil(
            dm_data["telescope_pupil"],
            device=device,
            dtype=precision.real,
            resolution=telescope_cfg.resolution,
        )

        # --------------------------------------------------------
        # Pupila física final
        #
        # DM pupil × telescope pupil
        # --------------------------------------------------------

        telescope_pupil = (
            dm_pupil
            * configured_telescope_pupil
        )

        if not torch.any(
            telescope_pupil > 0
        ):
            raise ValueError(
                "La intersección entre la pupila DM y "
                "la pupila física está vacía."
            )

    else:

        dm_basis_path = None
        dm_basis_type = "ZERNIKE"
        dm_name = "IDEAL"

        print("IDEAL ZERNIKE BASIS TYPE")

        # Pupila física del sistema.
        telescope_pupil = (
            configured_telescope_pupil
        )

        # --------------------------------------------------------
        # Zernikes:
        #
        # 1. círculo ideal
        # 2. aplicar pupila física
        # 3. pseudoinversa sobre pupila física
        # --------------------------------------------------------

        (
            zDecomposeMat,
            zComposeMat,
        ) = get_zernike_on_pupil(
            ideal_pupil=(
                ideal_telescope_pupil
                .squeeze()
                .cpu()
            ),
            physical_pupil=(
                telescope_pupil
                .squeeze()
                .cpu()
            ),
            diameter=telescope_cfg.diameter,
            nModes=(
                stages_cfg[0]
                .atmosphere
                .n_modes
            ),
            type="torch",
        )

    zDecomposeMat = zDecomposeMat.to(
        device=device,
        dtype=precision.real,
    )
    zComposeMat = zComposeMat.to(
        device=device,
        dtype=precision.real,
    )

    n_output_modes = int(zDecomposeMat.shape[0])
    for stage in stages_cfg:
        stage.atmosphere.n_modes = n_output_modes

    # ============================================================
    # WFS
    # ============================================================
    WFS = Pyramid(
        telescope_resolution=telescope_cfg.resolution,
        telescope_diameter=telescope_cfg.diameter,
        telescope_samp=telescope_cfg.samp,
        output_resolution=model_cfg.resolution,
        filter_ratio=wfs_cfg.filter_ratio,
        nHeads=wfs_cfg.heads,
        alpha=wfs_cfg.alpha,
        wavelength=source_cfg.wavelength,
        pixel_pitch=source_cfg.pixel_pitch,
        crop_mode=wfs_cfg.return_type,
        crop_pos_noise=wfs_cfg.crop_pos_noise,
        crop_size_noise=wfs_cfg.crop_size_noise,
        offset=wfs_cfg.offset,
        precision=precision,
        device=device,
        telescope_pupil=telescope_pupil
    )
    wfs_output_shape = WFS.piston_wfs.shape

    # ============================================================
    # CAMERA NOISE
    # ============================================================
    camera_noise_cfg = CameraNoiseAugmentConfig(
        low=CameraNoiseDomain(
            peak_e=Range(30.0, 150.0, log=True),
            bg_e=Range(0.5, 5.0, log=True),
            read_sigma_e=Range(1.5, 3.5),
            bias_dn=Range(0.0, 3.0),
        ),
        normal=CameraNoiseDomain(
            peak_e=Range(150.0, 1500.0, log=True),
            bg_e=Range(0.2, 3.0, log=True),
            read_sigma_e=Range(0.8, 2.0),
            bias_dn=Range(0.0, 3.0),
        ),
        good=CameraNoiseDomain(
            peak_e=Range(1500.0, 7000.0, log=True),
            bg_e=Range(0.05, 1.0, log=True),
            read_sigma_e=Range(0.5, 1.2),
            bias_dn=Range(0.0, 3.0),
        ),
        p_low=0.30,
        p_normal=0.55,
        p_good=0.15,
        parameter_mode="per_channel",
        shot_noise="poisson",
        output_mode="Mono8",
        mono16_align="lsb",
        use_ste_adc=False,
        auto_gain=True,
        adc_headroom=0.90,
        min_gain_e_per_dn=1e-6,
        add_prnu=True,
        prnu_sigma=0.005,
        add_dsnu=True,
        dsnu_sigma_e=0.2,
        return_metadata=False,
    )
    noise_pipe = CameraNoiseAugmenter(camera_noise_cfg)

    # ============================================================
    # MODEL
    # ============================================================
    NN = ModelManager(
        device=device,
        dtype=precision.real,
        nnModel=model_cfg.name,
        input_shape=wfs_output_shape,
        output_dim=n_output_modes,
        wts=model_cfg.weights,
    )

    # ============================================================
    # EXPERIMENT DIRECTORY
    # ============================================================
    experiment_path = Path(
        f"./TRAIN/{stages_cfg[0].train.precision}/"
        f"phiRes_{telescope_cfg.resolution}/"
        f"nnRes_{model_cfg.resolution}/"
        f"DM_{dm_name}/"
        f"{dm_basis_type}/"
        f"nModes_{n_output_modes}/"
        f"MODEL_{model_cfg.name}/"
        f"{experiment_name}/"
    )

    wfs_path = experiment_path / "WFS"
    wfs_figures_path = wfs_path / "figures"
    wfs_zernike_figures_path = (
        wfs_path / "figures_zernike"
    )

    wfs_path.mkdir(parents=True, exist_ok=True)
    wfs_figures_path.mkdir(parents=True, exist_ok=True)
    wfs_zernike_figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(6, 6),
    )

    ax.imshow(
        telescope_pupil[
            0,
            0,
        ].detach().cpu(),
        origin="upper",
    )

    ax.set_title(
        "TELESCOPE PUPIL"
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        wfs_figures_path
        / "telescope_pupil.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    save_run_info(
        cfg=cfg,
        out_dir=str(experiment_path),
        run_name="config",
    )
    _save_text(
        experiment_path / "config" / "pid.txt",
        str(pid),
    )

    torch.save(
        {
            "source_cfg": asdict(cfg.source),
            "telescope_cfg": asdict(cfg.telescope),
            "wfs_cfg": asdict(cfg.wfs),
            "model_cfg": asdict(cfg.model),
            "stages_cfg": [
                asdict(stage) for stage in cfg.stages
            ],
        },
        experiment_path / "config" / "all_cfg.pt",
    )
    torch.save(WFS, wfs_path / "WFS.pt")

    # ============================================================
    # FLAT WFS + NETWORK BOUNDING BOXES
    # ============================================================

    with torch.no_grad():

        flat_phi = torch.zeros(
            (
                1,
                1,
                telescope_cfg.resolution,
                telescope_cfg.resolution,
            ),
            device=device,
            dtype=precision.real,
        )

        (
            flat_full,
            flat_network,
            flat_boxes,
        ) = WFS.propagate(
            phi=flat_phi,
            pupil=telescope_pupil,
            return_both=True,
            return_boxes=True,
        )

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(6, 6),
            dpi=160,
        )

        _plot_wfs_with_boxes(
            ax,
            flat_full,
            flat_boxes[0],
            network_resolution=model_cfg.resolution,
            title=(
                "FLAT WFS | "
                f"NN input: {model_cfg.resolution}x"
                f"{model_cfg.resolution}"
            ),
        )

        fig.tight_layout()

        fig.savefig(
            wfs_figures_path / "wfs_flat.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ============================================================
    # BASIS SANITY PLOTS
    # ============================================================
    basis_amplitude = 0.1 if dm_basis else 5.0

    for mode_index in range(min(12, n_output_modes)):
        coefficient = torch.zeros(
            (1, 1, n_output_modes),
            dtype=precision.real,
            device=device,
        )
        coefficient[0, 0, mode_index] = basis_amplitude

        phi_positive = zernike_compose_torch(
            coefficient,
            zComposeMat,
        )
        phi_negative = zernike_compose_torch(
            -coefficient,
            zComposeMat,
        )

        (
            intensity_positive_full,
            intensity_positive_network,
            boxes_positive,
        ) = WFS.propagate(
            pupil=telescope_pupil,
            phi=phi_positive,
            return_both=True,
            return_boxes=True,
        )

        (
            intensity_negative_full,
            intensity_negative_network,
            boxes_negative,
        ) = WFS.propagate(
            pupil=telescope_pupil,
            phi=phi_negative,
            return_both=True,
            return_boxes=True,
        )


        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        axes[0, 0].imshow(phi_positive.squeeze().cpu())
        axes[0, 0].set_title(
            f"Mode {mode_index} | amp {basis_amplitude}"
        )
        axes[0, 0].axis("off")

        _plot_wfs_with_boxes(
            axes[0, 1],
            intensity_positive_full,
            boxes_positive[0],
            network_resolution=model_cfg.resolution,
            title="Pyramid propagation + NN crops",
        )

        axes[1, 0].imshow(phi_negative.squeeze().cpu())
        axes[1, 0].set_title(
            f"Mode {mode_index} | amp {-basis_amplitude}"
        )
        axes[1, 0].axis("off")

        _plot_wfs_with_boxes(
            axes[1, 1],
            intensity_negative_full,
            boxes_negative[0],
            network_resolution=model_cfg.resolution,
            title="Pyramid propagation + NN crops",
        )

        fig.tight_layout()
        fig.savefig(
            wfs_zernike_figures_path
            / f"mode_{mode_index}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # ============================================================
    # TRAINING STAGES
    # ============================================================
    for stage_idx, stage_cfg in enumerate(stages_cfg):
        train_cfg = stage_cfg.train
        atmosphere_cfg = stage_cfg.atmosphere
        loss_cfg = train_cfg.coef_loss

        
        (
            train_batches,
            val_batches,
            actual_train_samples,
            actual_val_samples,
        ) = _online_batch_counts(
            n_samples=atmosphere_cfg.n_samples,
            train_fraction=train_cfg.train_frac,
            batch_size=train_cfg.batch_size,
        )

        initial_seed = (
            atmosphere_cfg.seed
            + stage_idx * 100_000_000
        )
        initial_profile = _sample_atmosphere_profile(
            atmosphere_cfg=atmosphere_cfg,
            telescope_diameter=telescope_cfg.diameter,
            seed=initial_seed,
        )

        atmosphere = _build_atmosphere(
            initial_profile=initial_profile,
            atmosphere_cfg=atmosphere_cfg,
            telescope_cfg=telescope_cfg,
            source_cfg=source_cfg,
            train_cfg=train_cfg,
            precision=precision,
            device=device,
        )

        # --------------------------------------------------------
        # Stage directories
        # --------------------------------------------------------
        stage_path = experiment_path / f"stage_{stage_idx}"
        stage_figures_path = stage_path / "figures"
        stage_model_path = stage_path / "wts"

        stage_path.mkdir(parents=True, exist_ok=True)
        stage_figures_path.mkdir(parents=True, exist_ok=True)
        stage_model_path.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Spatial phi loss
        # --------------------------------------------------------
        coef_loss = CoefLoss(
            cfg=loss_cfg,
            n_modes=n_output_modes,
            tel_diameter=telescope_cfg.diameter,
            tel_resolution=telescope_cfg.resolution,
            tel_pupil=telescope_pupil,
            device=device,
            dtype=precision.real,
            dm_basis=dm_basis,
            dm_basis_path=(
                None
                if dm_basis_path is None
                else str(dm_basis_path)
            ),
            zComposeMat=zComposeMat,
        )

        # --------------------------------------------------------
        # Stage 0 sanity plots with an online validation sample
        # --------------------------------------------------------
        if stage_idx == 0:
            sanity_seed = (
                atmosphere_cfg.seed
                + atmosphere_cfg.validation_seed_offset
            )
            sanity_profile = _sample_atmosphere_profile(
                atmosphere_cfg=atmosphere_cfg,
                telescope_diameter=telescope_cfg.diameter,
                seed=sanity_seed,
            )
            sanity_phase = _configure_atmosphere(
                atmosphere,
                sanity_profile,
                realization_seed=sanity_seed,
            )

            with torch.no_grad():
                (
                    phi_test_full_batch,
                    amplitude_test_full_batch,
                    effective_pupil_test_full_batch,
                ) = _prepare_atmosphere_frame(
                    atmosphere=atmosphere,
                    phase=sanity_phase,
                    telescope_pupil=telescope_pupil,
                    scintillation=atmosphere_cfg.scintillation,
                )
                phi_test = phi_test_full_batch[0:1]
                amplitude_test = amplitude_test_full_batch[0:1]
                effective_pupil_test = (
                    effective_pupil_test_full_batch[0:1]
                )

                (
                    intensity_test_full,
                    intensity_test_network,
                    intensity_test_boxes,
                ) = WFS.propagate(
                    phi=phi_test,
                    pupil=effective_pupil_test,
                    return_both=True,
                    return_boxes=True,
                )

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15, 5),
                )
                fig.suptitle("ONLINE ATMOSPHERE WFS")
                axes[0].set_title("phase")
                axes[0].imshow(
                    phi_test[0, 0].detach().cpu()
                )
                axes[0].axis("off")
                axes[1].set_title("atmospheric amplitude")
                axes[1].imshow(
                    amplitude_test[0, 0].detach().cpu()
                )
                axes[1].axis("off")
                _plot_wfs_with_boxes(
                    axes[2],
                    intensity_test_full,
                    intensity_test_boxes[0],
                    network_resolution=model_cfg.resolution,
                    title="WFS intensity + NN crops",
                )
                fig.tight_layout()
                fig.savefig(
                    wfs_figures_path
                    / "wfs_atmosphere.jpeg",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

                mask = WFS.mask.detach().squeeze().cpu()
                plt.figure(figsize=(5, 5))
                plt.title("FOURIER PYRAMID MASK")
                plt.imshow(mask)
                plt.axis("off")
                plt.savefig(
                    wfs_figures_path / "mask_e0.jpeg",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close()

        # --------------------------------------------------------
        # Optimizer + scheduler
        # --------------------------------------------------------
        learning_rate = (
            1e-4
            if train_cfg.lr is None
            else float(train_cfg.lr)
        )
        weight_decay = float(train_cfg.weight_decay)
        epochs = int(train_cfg.epochs)

        optimizer = torch.optim.Adam(
            NN.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = None
        if train_cfg.lr_gamma is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=float(train_cfg.lr_gamma),
            )

        # --------------------------------------------------------
        # Histories
        # --------------------------------------------------------
        train_loss_history = []
        val_loss_history = []

        train_open_std_history = []
        val_open_std_history = []

        train_last_std_history = []
        val_last_std_history = []

        last_val_pack = None
        best_val_loss = math.inf

        print(
            f"Stage {stage_idx}: online generation | "
            f"train batches={train_batches}, "
            f"val batches={val_batches}, "
            f"batch={train_cfg.batch_size}, "
            f"layers={atmosphere_cfg.n_layers}, "
            f"scintillation={atmosphere_cfg.scintillation}"
        )

        for epoch in range(epochs):
            # ====================================================
            # TRAIN
            # ====================================================
            NN.train()

            running_loss = 0.0
            running_open_std = 0.0
            running_last_std = 0.0

            progress = tqdm(
                range(train_batches),
                desc=(
                    f"[stage {stage_idx}] "
                    f"train epoch {epoch + 1}/{epochs}"
                ),
                leave=False,
            )

            for batch_idx in progress:
                batch_seed = (
                    atmosphere_cfg.seed
                    + stage_idx * 100_000_000
                    + epoch * train_batches
                    + batch_idx
                )

                profile = _sample_atmosphere_profile(
                    atmosphere_cfg=atmosphere_cfg,
                    telescope_diameter=telescope_cfg.diameter,
                    seed=batch_seed,
                )
                initial_phase = _configure_atmosphere(
                    atmosphere,
                    profile,
                    realization_seed=batch_seed,
                )

                optimizer.zero_grad(set_to_none=True)

                output = _run_open_closed_loop_batch(
                    atmosphere=atmosphere,
                    initial_phase=initial_phase,
                    realization_seed=batch_seed,
                    scintillation=atmosphere_cfg.scintillation,
                    WFS=WFS,
                    NN=NN,
                    coef_loss=coef_loss,
                    zComposeMat=zComposeMat,
                    telescope_pupil=telescope_pupil,
                    train_cfg=train_cfg,
                    noise_pipe=noise_pipe,
                    device=device,
                )

                total_loss = output["total_loss"]
                step_logs = output["step_logs"]

                total_loss.backward()
                optimizer.step()

                running_loss += float(
                    total_loss.detach().cpu()
                )
                running_open_std += step_logs[0][
                    "std_spatial"
                ]
                running_last_std += step_logs[-1][
                    "std_spatial"
                ]

                completed = batch_idx + 1
                dr0_batch = (
                    telescope_cfg.diameter / atmosphere.r0_batch
                )
                std0 = (
                    running_open_std
                    / completed
                )

                std_last = (
                    running_last_std
                    / completed
                )

                progress.set_postfix(
                    loss=running_loss / completed,

                    std0=std0,
                    std_last=std_last,

                    wfe0_nm=(
                        std0
                        * phase_rad_to_wfe_nm
                    ),

                    wfe_last_nm=(
                        std_last
                        * phase_rad_to_wfe_nm
                    ),

                    dr0_min=float(
                        dr0_batch.min().item()
                    ),

                    dr0_max=float(
                        dr0_batch.max().item()
                    ),
                )

            train_loss_epoch = (
                running_loss / train_batches
            )
            train_open_std_epoch = (
                running_open_std / train_batches
            )
            train_last_std_epoch = (
                running_last_std / train_batches
            )

            train_loss_history.append(train_loss_epoch)
            train_open_std_history.append(
                train_open_std_epoch
            )
            train_last_std_history.append(
                train_last_std_epoch
            )

            # ====================================================
            # VALIDATION
            # ====================================================
            NN.eval()

            running_loss = 0.0
            running_open_std = 0.0
            running_last_std = 0.0

            progress = tqdm(
                range(val_batches),
                desc=(
                    f"[stage {stage_idx}] "
                    f"val epoch {epoch + 1}/{epochs}"
                ),
                leave=False,
            )

            with torch.no_grad():
                for batch_idx in progress:
                    # No epoch term: validation is regenerated, but exactly
                    # the same profiles/realizations are used every epoch.
                    batch_seed = (
                        atmosphere_cfg.seed
                        + atmosphere_cfg.validation_seed_offset
                        + stage_idx * 100_000_000
                        + batch_idx
                    )

                    profile = _sample_atmosphere_profile(
                        atmosphere_cfg=atmosphere_cfg,
                        telescope_diameter=telescope_cfg.diameter,
                        seed=batch_seed,
                    )
                    initial_phase = _configure_atmosphere(
                        atmosphere,
                        profile,
                        realization_seed=batch_seed,
                    )

                    output = _run_open_closed_loop_batch(
                        atmosphere=atmosphere,
                        initial_phase=initial_phase,
                        realization_seed=batch_seed,
                        scintillation=atmosphere_cfg.scintillation,
                        WFS=WFS,
                        NN=NN,
                        coef_loss=coef_loss,
                        zComposeMat=zComposeMat,
                        telescope_pupil=telescope_pupil,
                        train_cfg=train_cfg,
                        noise_pipe=noise_pipe,
                        device=device,
                    )

                    total_loss = output["total_loss"]
                    step_logs = output["step_logs"]

                    (
                        open_phi,
                        open_amplitude,
                        open_effective_pupil,
                        open_I,
                        open_pred,
                        open_boxes,
                    ) = output["open_debug"]

                    (
                        last_phi,
                        last_amplitude,
                        last_effective_pupil,
                        last_I,
                        last_pred,
                        last_boxes,
                    ) = output["last_debug"]

                    running_loss += float(
                        total_loss.detach().cpu()
                    )
                    running_open_std += step_logs[0][
                        "std_spatial"
                    ]
                    running_last_std += step_logs[-1][
                        "std_spatial"
                    ]

                    completed = batch_idx + 1
                    dr0_batch = (
                        telescope_cfg.diameter / atmosphere.r0_batch
                    )
                    std0 = (
                        running_open_std
                        / completed
                    )

                    std_last = (
                        running_last_std
                        / completed
                    )

                    progress.set_postfix(
                        loss=running_loss / completed,

                        std0=std0,
                        std_last=std_last,

                        wfe0_nm=(
                            std0
                            * phase_rad_to_wfe_nm
                        ),

                        wfe_last_nm=(
                            std_last
                            * phase_rad_to_wfe_nm
                        ),

                        dr0_min=float(
                            dr0_batch.min().item()
                        ),

                        dr0_max=float(
                            dr0_batch.max().item()
                        ),
                    )

                    last_val_pack = {
                        "open_phi": open_phi[-1:].cpu(),
                        "open_amplitude": open_amplitude[-1:].cpu(),

                        "open_effective_pupil": (
                            open_effective_pupil[-1:].cpu()
                        ),

                        "open_I": open_I[-1:].cpu(),
                        "open_pred": open_pred[-1:].cpu(),

                        # NUEVO
                        "open_boxes": open_boxes[-1],

                        "last_phi": last_phi[-1:].cpu(),
                        "last_amplitude": last_amplitude[-1:].cpu(),

                        "last_effective_pupil": (
                            last_effective_pupil[-1:].cpu()
                        ),

                        "last_I": last_I[-1:].cpu(),
                        "last_pred": last_pred[-1:].cpu(),

                        # NUEVO
                        "last_boxes": last_boxes[-1],

                        "profile": asdict(profile),
                    }

            val_loss_epoch = running_loss / val_batches
            val_open_std_epoch = (
                running_open_std / val_batches
            )
            val_last_std_epoch = (
                running_last_std / val_batches
            )

            val_loss_history.append(val_loss_epoch)
            val_open_std_history.append(
                val_open_std_epoch
            )
            val_last_std_history.append(
                val_last_std_epoch
            )

            # ====================================================
            # SCHEDULER
            # ====================================================
            if scheduler is not None:
                scheduler.step()

            # ====================================================
            # CURVES + HISTORY
            # ====================================================
            _save_curve(
                stage_figures_path,
                train_open=train_open_std_history,
                val_open=val_open_std_history,
                train_last=train_last_std_history,
                val_last=val_last_std_history,
                title=(
                    f"Stage {stage_idx} | "
                    "online phi-only STD"
                ),
            )

            torch.save(
                {
                    "train_loss": train_loss_history,
                    "val_loss": val_loss_history,
                    "train_open_std": (
                        train_open_std_history
                    ),
                    "val_open_std": val_open_std_history,
                    "train_last_std": (
                        train_last_std_history
                    ),
                    "val_last_std": val_last_std_history,
                    "learning_rate_initial": learning_rate,
                    "learning_rate_current": (
                        optimizer.param_groups[0]["lr"]
                    ),
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "cl_iter": train_cfg.cl_iter,
                    "cl_gain_range": train_cfg.cl_gain_range,
                    "train_batches_per_epoch": train_batches,
                    "val_batches_per_epoch": val_batches,
                    "actual_train_samples_per_epoch": (
                        actual_train_samples
                    ),
                    "actual_val_samples_per_epoch": (
                        actual_val_samples
                    ),
                    "online_generation": True,
                    "validation_is_deterministic": True,
                },
                stage_figures_path / "history.pt",
            )

            # ====================================================
            # SAVE BEST MODEL
            # ====================================================
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                NN.save_full(
                    str(stage_model_path / "model_full.pt")
                )
                torch.save(WFS, wfs_path / "WFS.pt")

            # ====================================================
            # VALIDATION DEBUG PLOT
            # ====================================================
            if last_val_pack is not None:
                open_phi = last_val_pack["open_phi"].to(
                    device=device,
                    dtype=precision.real,
                )
                last_phi = last_val_pack["last_phi"].to(
                    device=device,
                    dtype=precision.real,
                )
                open_effective_pupil = last_val_pack[
                    "open_effective_pupil"
                ].to(
                    device=device,
                    dtype=precision.real,
                )
                last_effective_pupil = last_val_pack[
                    "last_effective_pupil"
                ].to(
                    device=device,
                    dtype=precision.real,
                )

                open_I_full = WFS.propagate(
                    pupil=open_effective_pupil,
                    phi=open_phi,
                    no_crop=True,
                )
                last_I_full = WFS.propagate(
                    pupil=last_effective_pupil,
                    phi=last_phi,
                    no_crop=True,
                )

                _save_debug_val_open_last_plot(
                    stage_figures_path,

                    open_phi=open_phi.cpu(),

                    open_amplitude=last_val_pack[
                        "open_amplitude"
                    ],

                    open_I=open_I_full.detach().cpu(),

                    open_boxes=last_val_pack[
                        "open_boxes"
                    ],

                    open_pred=last_val_pack[
                        "open_pred"
                    ],

                    last_phi=last_phi.cpu(),

                    last_amplitude=last_val_pack[
                        "last_amplitude"
                    ],

                    last_I=last_I_full.detach().cpu(),

                    last_boxes=last_val_pack[
                        "last_boxes"
                    ],

                    last_pred=last_val_pack[
                        "last_pred"
                    ],

                    network_resolution=model_cfg.resolution,

                    stage_idx=stage_idx,
                )

        # --------------------------------------------------------
        # Stage log
        # --------------------------------------------------------
        _save_text(
            stage_path / "stage_info.txt",
            "\n".join(
                [
                    f"experiment_name: {experiment_name}",
                    f"pid: {pid}",
                    f"stage_idx: {stage_idx}",
                    "dataset_saved: False",
                    "generation_mode: online",
                    f"total_configured_samples_per_epoch: {atmosphere_cfg.n_samples}",
                    f"actual_train_samples_per_epoch: {actual_train_samples}",
                    f"actual_val_samples_per_epoch: {actual_val_samples}",
                    f"train_batches_per_epoch: {train_batches}",
                    f"val_batches_per_epoch: {val_batches}",
                    f"batch_size: {train_cfg.batch_size}",
                    f"n_layers: {atmosphere_cfg.n_layers}",
                    f"fractional_r0: {atmosphere_cfg.fractional_r0}",
                    f"dr0_range: {atmosphere_cfg.dr0_range}",
                    f"wind_speed_range: {atmosphere_cfg.wind_speed_range}",
                    f"wind_direction_range: {atmosphere_cfg.wind_direction_range}",
                    f"altitude_range: {atmosphere_cfg.altitude_range}",
                    f"frame_rate_hz: {atmosphere_cfg.frame_rate}",
                    f"scintillation: {atmosphere_cfg.scintillation}",
                    f"propagation_mode: {atmosphere_cfg.propagation_mode}",
                    f"delta_mode: {atmosphere_cfg.delta_mode}",
                    f"frozen_flow_mode: {atmosphere_cfg.frozen_flow_mode}",
                    f"learning_rate: {learning_rate}",
                    f"weight_decay: {weight_decay}",
                    f"epochs: {epochs}",
                    f"cl_iter: {train_cfg.cl_iter}",
                    f"cl_gain_range: {train_cfg.cl_gain_range}",
                    f"metric: {loss_cfg.metric}",
                    f"loss_eps: {loss_cfg.eps}",
                    "validation_seed_policy: fixed_across_epochs",
                ]
            ),
        )


if __name__ == "__main__":
    main()
