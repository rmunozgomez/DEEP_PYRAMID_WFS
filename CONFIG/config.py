"""
Public configuration API used by 1-main.py.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import json
import os

from . import config_parameters as P
from .config_schema import (
    AtmosphereStageCfg,
    CameraNoiseCfg,
    CameraNoiseDomainCfg,
    CoefLossCfg,
    ExperimentCfg,
    ModelCfg,
    RangeCfg,
    SourceCfg,
    StageCfg,
    TelescopeCfg,
    TrainStageCfg,
    WfsCfg,
)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def _broadcast(value: Any, n_stages: int, name: str) -> List[Any]:
    values = _as_list(value)
    if len(values) == 1:
        return values * n_stages
    if len(values) != n_stages:
        raise ValueError(
            f"{name} has length {len(values)}; expected 1 or {n_stages}."
        )
    return values


def _pair(value: Sequence[Any], name: str) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly [min, max].")
    low, high = float(value[0]), float(value[1])
    if high < low:
        raise ValueError(f"{name} has max < min: {value}.")
    return low, high

def _range_cfg(
    value: Sequence[Any],
    name: str,
) -> RangeCfg:

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain (low, high, log)."
        )

    low = float(value[0])
    high = float(value[1])
    log = bool(value[2])

    if high < low:
        raise ValueError(
            f"{name} has max < min: {value}."
        )

    return RangeCfg(
        low=low,
        high=high,
        log=log,
    )

def _infer_num_stages() -> int:
    # The number of stages remains defined by EPOCHS.
    return len(_as_list(P.EPOCHS))


def _build_stages() -> List[StageCfg]:
    n = _infer_num_stages()

    atmosphere_values = {
        "dr0": _broadcast(P.ATMOSPHERE_DR0_RANGE, n, "ATMOSPHERE_DR0_RANGE"),
        "samples": _broadcast(P.ATMOSPHERE_SAMPLES, n, "ATMOSPHERE_SAMPLES"),
        "l0": _broadcast(P.ATMOSPHERE_l0, n, "ATMOSPHERE_l0"),
        "L0": _broadcast(P.ATMOSPHERE_L0, n, "ATMOSPHERE_L0"),
        "fraction": _broadcast(
            P.ATMOSPHERE_FRACTIONAL_R0, n, "ATMOSPHERE_FRACTIONAL_R0"
        ),
        "wind_speed": _broadcast(
            P.ATMOSPHERE_WIND_SPEED_RANGE,
            n,
            "ATMOSPHERE_WIND_SPEED_RANGE",
        ),
        "wind_direction": _broadcast(
            P.ATMOSPHERE_WIND_DIRECTION_RANGE,
            n,
            "ATMOSPHERE_WIND_DIRECTION_RANGE",
        ),
        "altitude": _broadcast(
            P.ATMOSPHERE_ALTITUDE_RANGE, n, "ATMOSPHERE_ALTITUDE_RANGE"
        ),
        "frame_rate": _broadcast(
            P.ATMOSPHERE_FRAME_RATE, n, "ATMOSPHERE_FRAME_RATE"
        ),
        "seed": _broadcast(P.ATMOSPHERE_SEED, n, "ATMOSPHERE_SEED"),
        "val_seed_offset": _broadcast(
            P.ATMOSPHERE_VALIDATION_SEED_OFFSET,
            n,
            "ATMOSPHERE_VALIDATION_SEED_OFFSET",
        ),
        "n_sh": _broadcast(
            P.ATMOSPHERE_N_SUBHARMONIC_LEVELS,
            n,
            "ATMOSPHERE_N_SUBHARMONIC_LEVELS",
        ),
        "sh_mode": _broadcast(
            P.ATMOSPHERE_SUBHARMONIC_MODE,
            n,
            "ATMOSPHERE_SUBHARMONIC_MODE",
        ),
        "direction_convention": _broadcast(
            P.ATMOSPHERE_DIRECTION_CONVENTION,
            n,
            "ATMOSPHERE_DIRECTION_CONVENTION",
        ),
        "normalize_fraction": _broadcast(
            P.ATMOSPHERE_NORMALIZE_FRACTIONAL_R0,
            n,
            "ATMOSPHERE_NORMALIZE_FRACTIONAL_R0",
        ),
        "remove_piston": _broadcast(
            P.ATMOSPHERE_REMOVE_PISTON,
            n,
            "ATMOSPHERE_REMOVE_PISTON",
        ),
        "store_components": _broadcast(
            P.ATMOSPHERE_STORE_COMPONENTS,
            n,
            "ATMOSPHERE_STORE_COMPONENTS",
        ),
        "r0_reference_wavelength": _broadcast(
            P.ATMOSPHERE_R0_REFERENCE_WAVELENGTH,
            n,
            "ATMOSPHERE_R0_REFERENCE_WAVELENGTH",
        ),
        "scintillation": _broadcast(
            P.ATMOSPHERE_SCINTILLATION,
            n,
            "ATMOSPHERE_SCINTILLATION",
        ),
        "frozen_flow": _broadcast(
            P.ATMOSPHERE_FROZEN_FLOW_MODE,
            n,
            "ATMOSPHERE_FROZEN_FLOW_MODE",
        ),
        "propagation": _broadcast(
            P.ATMOSPHERE_PROPAGATION_MODE,
            n,
            "ATMOSPHERE_PROPAGATION_MODE",
        ),
        "delta": _broadcast(
            P.ATMOSPHERE_DELTA_MODE, n, "ATMOSPHERE_DELTA_MODE"
        ),
        "asm_extra": _broadcast(
            P.ATMOSPHERE_ASM_EXTRA_PIXELS,
            n,
            "ATMOSPHERE_ASM_EXTRA_PIXELS",
        ),
        "asm_margin": _broadcast(
            P.ATMOSPHERE_ASM_MIN_PHYSICAL_MARGIN,
            n,
            "ATMOSPHERE_ASM_MIN_PHYSICAL_MARGIN",
        ),
        "asm_padding": _broadcast(
            P.ATMOSPHERE_ASM_PADDING_FACTOR,
            n,
            "ATMOSPHERE_ASM_PADDING_FACTOR",
        ),
        "delta_warning": _broadcast(
            P.ATMOSPHERE_DELTA_WRAP_WARNING_THRESHOLD,
            n,
            "ATMOSPHERE_DELTA_WRAP_WARNING_THRESHOLD",
        ),
        "reanchor": _broadcast(
            P.ATMOSPHERE_TEMPORAL_REANCHOR_INTERVAL,
            n,
            "ATMOSPHERE_TEMPORAL_REANCHOR_INTERVAL",
        ),
        "r0_sampling": _broadcast(
            P.ATMOSPHERE_R0_SAMPLING,
            n,
            "ATMOSPHERE_R0_SAMPLING",
        ),
        "layer_sampling": _broadcast(
            P.ATMOSPHERE_LAYER_SAMPLING,
            n,
            "ATMOSPHERE_LAYER_SAMPLING",
),
    }

    training_values = {
        "precision": _broadcast(P.PRECISION, n, "PRECISION"),
        "norm": _broadcast(P.NORM_TYPE, n, "NORM_TYPE"),
        "loss_metric": _broadcast(P.LOSS_TYPE, n, "LOSS_TYPE"),
        "loss_eps": _broadcast(P.LOSS_EPS, n, "LOSS_EPS"),
        "train_frac": _broadcast(P.TRAIN_FRAC, n, "TRAIN_FRAC"),
        "batch_size": _broadcast(P.BATCH_SIZE, n, "BATCH_SIZE"),
        "epochs": _broadcast(P.EPOCHS, n, "EPOCHS"),
        "lr": _broadcast(P.LR, n, "LR"),
        "weight_decay": _broadcast(
            P.WEIGHT_DECAY, n, "WEIGHT_DECAY"
        ),
        "lr_gamma": _broadcast(P.LR_GAMMA, n, "LR_GAMMA"),
        "cl_iter": _broadcast(P.CL_ITER, n, "CL_ITER"),
        "cl_gain": _broadcast(P.CL_GAIN_RANGE, n, "CL_GAIN_RANGE"),
        "noise": _broadcast(P.NOISE, n, "NOISE"),
    }

    stages: List[StageCfg] = []

    for index in range(n):
        atmosphere = AtmosphereStageCfg(
            dr0_range=_pair(
                atmosphere_values["dr0"][index],
                "ATMOSPHERE_DR0_RANGE",
            ),
            n_samples=int(atmosphere_values["samples"][index]),
            l0=float(atmosphere_values["l0"][index]),
            n_modes=int(P.ATMOSPHERE_MODES),
            L0=float(atmosphere_values["L0"][index]),

            dm_basis=bool(P.DM_BASIS),
            dm_basis_type=str(P.DM_BASIS_TYPE),
            dm_name=str(P.DM_NAME),

            fractional_r0=tuple(
                float(value)
                for value in atmosphere_values["fraction"][index]
            ),
            wind_speed_range=_pair(
                atmosphere_values["wind_speed"][index],
                "ATMOSPHERE_WIND_SPEED_RANGE",
            ),
            wind_direction_range=_pair(
                atmosphere_values["wind_direction"][index],
                "ATMOSPHERE_WIND_DIRECTION_RANGE",
            ),
            altitude_range=_pair(
                atmosphere_values["altitude"][index],
                "ATMOSPHERE_ALTITUDE_RANGE",
            ),

            frame_rate=float(atmosphere_values["frame_rate"][index]),
            seed=int(atmosphere_values["seed"][index]),
            validation_seed_offset=int(
                atmosphere_values["val_seed_offset"][index]
            ),

            n_subharmonic_levels=int(
                atmosphere_values["n_sh"][index]
            ),
            subharmonic_mode=str(
                atmosphere_values["sh_mode"][index]
            ),
            direction_convention=str(
                atmosphere_values["direction_convention"][index]
            ),
            normalize_fractional_r0=bool(
                atmosphere_values["normalize_fraction"][index]
            ),
            remove_piston=bool(
                atmosphere_values["remove_piston"][index]
            ),
            store_components=bool(
                atmosphere_values["store_components"][index]
            ),
            r0_reference_wavelength=float(
                atmosphere_values["r0_reference_wavelength"][index]
            ),
            scintillation=bool(
                atmosphere_values["scintillation"][index]
            ),

            frozen_flow_mode=str(
                atmosphere_values["frozen_flow"][index]
            ),
            propagation_mode=str(
                atmosphere_values["propagation"][index]
            ),
            delta_mode=str(atmosphere_values["delta"][index]),
            asm_extra_pixels=atmosphere_values["asm_extra"][index],
            asm_min_physical_margin=float(
                atmosphere_values["asm_margin"][index]
            ),
            asm_padding_factor=float(
                atmosphere_values["asm_padding"][index]
            ),
            delta_wrap_warning_threshold=float(
                atmosphere_values["delta_warning"][index]
            ),
            temporal_reanchor_interval=int(
                atmosphere_values["reanchor"][index]
            ),
            r0_sampling=str(
                atmosphere_values[
                "r0_sampling"
                ][index]
            ),
            layer_sampling=str(
                atmosphere_values[
                    "layer_sampling"
                ][index]
            ),
        )

        loss = CoefLossCfg(
            metric=str(training_values["loss_metric"][index]),
            eps=float(training_values["loss_eps"][index]),
        )

        train = TrainStageCfg(
            precision=str(training_values["precision"][index]),
            norm_type=str(training_values["norm"][index]),
            coef_loss=loss,
            train_frac=float(training_values["train_frac"][index]),
            batch_size=int(training_values["batch_size"][index]),
            epochs=int(training_values["epochs"][index]),
            lr=(
                None
                if training_values["lr"][index] is None
                else float(training_values["lr"][index])
            ),
            weight_decay=float(
                training_values["weight_decay"][index]
            ),
            lr_gamma=(
                None
                if training_values["lr_gamma"][index] is None
                else float(training_values["lr_gamma"][index])
            ),
            cl_iter=int(training_values["cl_iter"][index]),
            cl_gain_range=_pair(
                training_values["cl_gain"][index],
                "CL_GAIN_RANGE",
            ),
            noise=bool(training_values["noise"][index]),
        )

        stages.append(StageCfg(atmosphere=atmosphere, train=train))

    return stages

def _build_camera_cfg() -> CameraNoiseCfg:

    return CameraNoiseCfg(
        low=CameraNoiseDomainCfg(
            peak_e=_range_cfg(
                P.CAMERA_LOW_PEAK_E,
                "CAMERA_LOW_PEAK_E",
            ),
            bg_e=_range_cfg(
                P.CAMERA_LOW_BG_E,
                "CAMERA_LOW_BG_E",
            ),
            read_sigma_e=_range_cfg(
                P.CAMERA_LOW_READ_SIGMA_E,
                "CAMERA_LOW_READ_SIGMA_E",
            ),
            bias_dn=_range_cfg(
                P.CAMERA_LOW_BIAS_DN,
                "CAMERA_LOW_BIAS_DN",
            ),
        ),

        normal=CameraNoiseDomainCfg(
            peak_e=_range_cfg(
                P.CAMERA_NORMAL_PEAK_E,
                "CAMERA_NORMAL_PEAK_E",
            ),
            bg_e=_range_cfg(
                P.CAMERA_NORMAL_BG_E,
                "CAMERA_NORMAL_BG_E",
            ),
            read_sigma_e=_range_cfg(
                P.CAMERA_NORMAL_READ_SIGMA_E,
                "CAMERA_NORMAL_READ_SIGMA_E",
            ),
            bias_dn=_range_cfg(
                P.CAMERA_NORMAL_BIAS_DN,
                "CAMERA_NORMAL_BIAS_DN",
            ),
        ),

        good=CameraNoiseDomainCfg(
            peak_e=_range_cfg(
                P.CAMERA_GOOD_PEAK_E,
                "CAMERA_GOOD_PEAK_E",
            ),
            bg_e=_range_cfg(
                P.CAMERA_GOOD_BG_E,
                "CAMERA_GOOD_BG_E",
            ),
            read_sigma_e=_range_cfg(
                P.CAMERA_GOOD_READ_SIGMA_E,
                "CAMERA_GOOD_READ_SIGMA_E",
            ),
            bias_dn=_range_cfg(
                P.CAMERA_GOOD_BIAS_DN,
                "CAMERA_GOOD_BIAS_DN",
            ),
        ),

        p_low=float(P.CAMERA_P_LOW),
        p_normal=float(P.CAMERA_P_NORMAL),
        p_good=float(P.CAMERA_P_GOOD),

        parameter_mode=str(
            P.CAMERA_PARAMETER_MODE
        ),

        shot_noise=str(
            P.CAMERA_SHOT_NOISE
        ),

        output_mode=str(
            P.CAMERA_OUTPUT_MODE
        ),

        mono16_align=str(
            P.CAMERA_MONO16_ALIGN
        ),

        use_ste_adc=bool(
            P.CAMERA_USE_STE_ADC
        ),

        auto_gain=bool(
            P.CAMERA_AUTO_GAIN
        ),

        adc_headroom=float(
            P.CAMERA_ADC_HEADROOM
        ),

        min_gain_e_per_dn=float(
            P.CAMERA_MIN_GAIN_E_PER_DN
        ),

        add_prnu=bool(
            P.CAMERA_ADD_PRNU
        ),

        prnu_sigma=float(
            P.CAMERA_PRNU_SIGMA
        ),

        add_dsnu=bool(
            P.CAMERA_ADD_DSNU
        ),

        dsnu_sigma_e=float(
            P.CAMERA_DSNU_SIGMA_E
        ),

        seed_offset=int(
            P.CAMERA_SEED_OFFSET
        ),
    )

def get_config() -> ExperimentCfg:
    return ExperimentCfg(
        source=SourceCfg(
            wavelength=float(P.SOURCE_WAVELENGTH),
            pixel_pitch=float(P.SOURCE_PIXEL_PITCH),
        ),
        telescope=TelescopeCfg(
            diameter=float(
                P.TELESCOPE_DIAMETER
            ),

            resolution=int(
                P.TELESCOPE_RESOLUTION
            ),

            spiders=int(
                P.TELESCOPE_SPIDERS
            ),

            spiders_px=float(
                P.TELESCOPE_SPIDERS_PX
            ),

            spider_angles_deg=(
                None
                if P.TELESCOPE_SPIDER_ANGLES_DEG is None
                else tuple(
                    float(angle)
                    for angle
                    in P.TELESCOPE_SPIDER_ANGLES_DEG
                )
            ),

            central_obstruction_px=float(
                P.TELESCOPE_CENTRAL_OBSTRUCTION_PX
            ),

            central_obstruction_offset_px=tuple(
                float(value)
                for value
                in P.TELESCOPE_CENTRAL_OBSTRUCTION_OFFSET_PX
            ),

            samp=int(
                P.TELESCOPE_SAMP
            ),
        ),
        wfs=WfsCfg(
            return_type=str(P.WFS_RETURN_TYPE),
            heads=int(P.WFS_HEADS),
            alpha=float(P.WFS_ALPHA),
            filter_ratio=float(P.WFS_FILTER_RATIO),
            crop_pos_noise=int(P.WFS_CROP_POS_NOISE),
            crop_size_noise=int(P.WFS_CROP_SIZE_NOISE),
            offset=int(P.WFS_OFFSET),
        ),
        model=ModelCfg(
            name=str(P.MODEL),
            weights=P.WTS,
            resolution=int(P.NN_RESOLUTION),
        ),

        camera=_build_camera_cfg(),

        stages=_build_stages(),
    )


def save_run_info(
    cfg: ExperimentCfg,
    out_dir: str = "runs",
    run_name: Optional[str] = None,
    save_json: bool = True,
) -> str:
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{timestamp}_pid{pid}"

    run_path = Path(out_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "pid": pid,
        "cwd": str(Path.cwd()),
        "python": os.sys.executable,
    }

    config_dict = asdict(cfg)

    txt_path = run_path / "run_info.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("=== RUN INFO ===\n")
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")

        handle.write("\n=== CONFIG (BASE) ===\n")
        handle.write(f"source: {cfg.source}\n")
        handle.write(f"telescope: {cfg.telescope}\n")
        handle.write(f"wfs: {cfg.wfs}\n")
        handle.write(f"model: {cfg.model}\n")
        handle.write(f"camera: {cfg.camera}\n")

        handle.write("\n=== CONFIG (STAGES) ===\n")
        for index, stage in enumerate(cfg.stages, start=1):
            handle.write(f"\n--- Stage {index} ---\n")
            handle.write(f"atmosphere: {stage.atmosphere}\n")
            handle.write(f"train: {stage.train}\n")
            handle.write(f"coef_loss: {stage.train.coef_loss}\n")

    if save_json:
        json_path = run_path / "config.json"
        json_path.write_text(
            json.dumps(
                {"meta": metadata, "config": config_dict},
                indent=2,
            ),
            encoding="utf-8",
        )

    return str(txt_path)
