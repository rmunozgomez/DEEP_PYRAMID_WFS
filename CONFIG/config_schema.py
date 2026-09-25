"""
Dataclasses used by the experiment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union
import math


WFSReturnType = Literal["pupils", "full_frame"]
PrecisionType = Literal["single", "double"]
NormType = Literal[
    "zscore",
    "zscore_global",
    "max",
    "max_global",
    "minmax",
    "flux",
    "none",
]
LossType = Literal["std","std_grad_local"]

SubharmonicMode = Literal["oopao", "full"]
DirectionConvention = Literal["cartesian", "oopao"]
PropagationMode = Literal["geometric", "asm_delta"]
DeltaMode = Literal["final", "per_step"]
FrozenFlowMode = Literal["analytic", "periodic_screen"]
AsmExtraPixels = Union[int, Literal["auto"], None]


@dataclass(frozen=True)
class SourceCfg:
    wavelength: float = 635e-9
    pixel_pitch: float = 3.74e-6


@dataclass(frozen=True)
class TelescopeCfg:
    diameter: float = 3.0
    resolution: int = 128

    spiders: int = 0
    spiders_px: float = 0.0
    spider_angles_deg: Optional[Tuple[float, ...]] = None

    central_obstruction_px: float = 0.0
    central_obstruction_offset_px: Tuple[float, float] = (
        0.0,
        0.0,
    )

    samp: int = 2

    def __post_init__(self) -> None:

        if self.diameter <= 0:
            raise ValueError(
                "diameter must be positive."
            )

        if self.resolution <= 0:
            raise ValueError(
                "resolution must be positive."
            )

        if self.spiders < 0:
            raise ValueError(
                "spiders must be >= 0."
            )

        if self.spiders_px < 0:
            raise ValueError(
                "spiders_px must be >= 0."
            )

        if self.central_obstruction_px < 0:
            raise ValueError(
                "central_obstruction_px must be >= 0."
            )

        if self.central_obstruction_px >= self.resolution:
            raise ValueError(
                "central_obstruction_px must be smaller "
                "than telescope resolution."
            )

        if self.samp <= 0:
            raise ValueError(
                "samp must be positive."
            )

        if len(self.central_obstruction_offset_px) != 2:
            raise ValueError(
                "central_obstruction_offset_px must be (dx, dy)."
            )

        if not all(
            math.isfinite(float(value))
            for value
            in self.central_obstruction_offset_px
        ):
            raise ValueError(
                "central_obstruction_offset_px must contain "
                "finite values."
            )

        if self.spider_angles_deg is not None:

            if len(self.spider_angles_deg) != self.spiders:
                raise ValueError(
                    "When spider_angles_deg is supplied, its "
                    "length must equal spiders."
                )

            if not all(
                math.isfinite(float(angle))
                for angle
                in self.spider_angles_deg
            ):
                raise ValueError(
                    "spider_angles_deg must contain finite values."
                )


@dataclass(frozen=True)
class WfsCfg:
    return_type: WFSReturnType = "pupils"
    heads: int = 4
    alpha: float = 2.4
    filter_ratio: float = 1.0
    crop_pos_noise: int = 0
    crop_size_noise: int = 0
    offset: int = 0


@dataclass(frozen=True)
class ModelCfg:
    name: str = "GcVit"
    weights: Optional[str] = None
    resolution: int = 128


@dataclass(frozen=False)
class AtmosphereStageCfg:
    dr0_range: Tuple[float, float] = (10.0, 100.0)
    n_samples: int = 10_000
    l0: float = 1e-10
    n_modes: int = 68
    L0: float = 25.0

    dm_basis: bool = False
    dm_basis_type: str = "ACTUATOR"
    dm_name: str = "./"

    fractional_r0: Tuple[float, ...] = (1.0,)
    wind_speed_range: Tuple[float, float] = (0.0, 15.0)
    wind_direction_range: Tuple[float, float] = (0.0, 360.0)
    altitude_range: Tuple[float, float] = (0.0, 10_000.0)

    frame_rate: float = 1_000.0
    seed: int = 1234
    validation_seed_offset: int = 10_000_000

    n_subharmonic_levels: int = 3
    subharmonic_mode: SubharmonicMode = "full"
    direction_convention: DirectionConvention = "oopao"
    normalize_fractional_r0: bool = True
    remove_piston: bool = False
    store_components: bool = False
    r0_reference_wavelength: float = 635e-9
    scintillation: bool = True

    frozen_flow_mode: FrozenFlowMode = "analytic"
    propagation_mode: PropagationMode = "geometric"
    delta_mode: DeltaMode = "final"
    asm_extra_pixels: AsmExtraPixels = "auto"
    asm_min_physical_margin: float = 0.75
    asm_padding_factor: float = 2.0
    delta_wrap_warning_threshold: float = 1.9 * 3.141592653589793
    temporal_reanchor_interval: int = 256

    @property
    def n_layers(self) -> int:
        return len(self.fractional_r0)

    def __post_init__(self) -> None:
        dr0_low, dr0_high = self.dr0_range
        if dr0_low <= 0 or dr0_high < dr0_low:
            raise ValueError(f"Invalid dr0_range: {self.dr0_range}")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.l0 <= 0 or self.L0 <= 0:
            raise ValueError("l0 and L0 must be positive.")
        if not self.fractional_r0:
            raise ValueError("fractional_r0 must define at least one layer.")
        if any(value < 0 for value in self.fractional_r0):
            raise ValueError("fractional_r0 cannot contain negative values.")
        if sum(self.fractional_r0) <= 0:
            raise ValueError("fractional_r0 must have a positive sum.")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive.")
        if self.n_subharmonic_levels <= 0:
            raise ValueError("n_subharmonic_levels must be positive.")
        if self.r0_reference_wavelength <= 0:
            raise ValueError("r0_reference_wavelength must be positive.")
        if self.asm_padding_factor < 1.0:
            raise ValueError("asm_padding_factor must be >= 1.")
        if self.asm_min_physical_margin < 0:
            raise ValueError("asm_min_physical_margin must be >= 0.")
        if self.temporal_reanchor_interval < 0:
            raise ValueError("temporal_reanchor_interval must be >= 0.")

        for name, interval in (
            ("wind_speed_range", self.wind_speed_range),
            ("wind_direction_range", self.wind_direction_range),
            ("altitude_range", self.altitude_range),
        ):
            if interval[1] < interval[0]:
                raise ValueError(f"{name} has max < min: {interval}")


@dataclass(frozen=True)
class CoefLossCfg:
    metric: LossType = "std"
    eps: float = 1e-8


@dataclass(frozen=True)
class TrainStageCfg:
    precision: PrecisionType = "single"
    norm_type: NormType = "zscore"
    coef_loss: CoefLossCfg = field(default_factory=CoefLossCfg)

    train_frac: float = 0.8
    batch_size: int = 10
    epochs: int = 100
    lr: Optional[float] = None
    weight_decay: float = 0.0
    lr_gamma: Optional[float] = 0.9

    cl_iter: int = 2
    cl_gain_range: Tuple[float, float] = (0.3, 1.0)
    noise: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must lie strictly between 0 and 1.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.cl_iter < 0:
            raise ValueError("cl_iter must be >= 0.")
        if self.cl_gain_range[0] < 0:
            raise ValueError("closed-loop gain cannot be negative.")
        if self.cl_gain_range[1] < self.cl_gain_range[0]:
            raise ValueError("cl_gain_range has max < min.")


@dataclass(frozen=True)
class StageCfg:
    atmosphere: AtmosphereStageCfg
    train: TrainStageCfg


@dataclass(frozen=True)
class ExperimentCfg:
    source: SourceCfg = field(default_factory=SourceCfg)
    telescope: TelescopeCfg = field(default_factory=TelescopeCfg)
    wfs: WfsCfg = field(default_factory=WfsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    stages: List[StageCfg] = field(default_factory=list)
