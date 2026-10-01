from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Tuple, Optional, Union, Literal

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
AlignMode = Literal["lsb", "msb"]  # lsb=right-aligned, msb=left-aligned


# ============================================================
# Helpers
# ============================================================
def ensure_4d(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Se esperaba (B,C,H,W). Llegó: {tuple(x.shape)}")
    return x


def _as_tensor(v: Union[float, Tensor], like: Tensor) -> Tensor:
    if torch.is_tensor(v):
        return v.to(device=like.device, dtype=like.dtype)
    return torch.tensor(v, device=like.device, dtype=like.dtype)


def _broadcast(p: Tensor, x: Tensor) -> Tensor:
    while p.ndim < x.ndim:
        p = p.unsqueeze(-1)
    return p


def clamp_nonneg(x: Tensor, eps: float = 0.0) -> Tensor:
    return x.clamp_min(eps)


def randn_like_compat(x: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Compatibilidad: algunas versiones de PyTorch NO soportan generator= en torch.randn_like.
    """
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


# ============================================================
# A) Efectos óptico-sensor (diferenciables)
# ============================================================
def gaussian_blur2d(x: Tensor, sigma_px: float, ksize: int = 7) -> Tensor:
    """
    Blur gaussiano (defocus leve / MTF / smoothing). Diferenciable.
    x: (B,C,H,W)
    """
    x = ensure_4d(x)
    if sigma_px <= 0:
        return x
    if ksize % 2 == 0:
        ksize += 1

    device, dtype = x.device, x.dtype
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    k = (k / k.sum()).view(1, 1, ksize, ksize)

    C = x.shape[1]
    k = k.repeat(C, 1, 1, 1)  # (C,1,K,K)
    return F.conv2d(x, k, padding=ksize // 2, groups=C)


def pixel_response_box(x: Tensor, ksize: int = 3) -> Tensor:
    """
    Integración espacial del píxel (pixel aperture) como box filter. Diferenciable.
    """
    x = ensure_4d(x)
    if ksize <= 1:
        return x
    device, dtype = x.device, x.dtype
    k = torch.ones((1, 1, ksize, ksize), device=device, dtype=dtype)
    k = (k / k.sum()).view(1, 1, ksize, ksize)

    C = x.shape[1]
    k = k.repeat(C, 1, 1, 1)
    return F.conv2d(x, k, padding=ksize // 2, groups=C)


# ============================================================
# B) Escala a electrones esperados (común en simulación)
# ============================================================
def to_expected_electrons(
    intensity: Tensor,
    signal_e: Union[float, Tensor],
    *,
    scaling: SignalScaling = "peak",
    scope: SignalScope = "sample",
    eps: float = 1e-12,
) -> Tensor:
    """
    Convert arbitrary non-negative optical intensity into
    expected photoelectrons.

    scaling
    -------
    peak:
        signal_e is the expected peak electron level.

    flux:
        signal_e is the expected total electron budget.

    linear:
        intensity is multiplied directly by signal_e.

    scope
    -----
    sample:
        All channels belong to one detector observation.

    channel:
        Each channel is normalized independently.
    """

    I = ensure_4d(intensity)
    I = clamp_nonneg(
        I,
        0.0,
    )

    level = _broadcast(
        _as_tensor(
            signal_e,
            I,
        ),
        I,
    )

    if scaling == "linear":
        return I * level

    if scope == "sample":
        reduce_dims = (
            -3,
            -2,
            -1,
        )

    elif scope == "channel":
        reduce_dims = (
            -2,
            -1,
        )

    else:
        raise ValueError(
            f"Unknown signal scope: {scope}"
        )

    if scaling == "peak":

        reference = I.amax(
            dim=reduce_dims,
            keepdim=True,
        ).clamp_min(
            eps
        )

        normalized = I / reference

    elif scaling == "flux":

        reference = I.sum(
            dim=reduce_dims,
            keepdim=True,
        ).clamp_min(
            eps
        )

        normalized = I / reference

    else:
        raise ValueError(
            f"Unknown signal scaling: {scaling}"
        )

    return normalized * level


# ============================================================
# C) Background/dark en electrones, y bias en DN
# ============================================================
def add_background_e(lam_e: Tensor, bg_e: Union[float, Tensor]) -> Tensor:
    """
    Suma background/dark (electrones esperados).
    """
    x = ensure_4d(lam_e)
    b = _broadcast(_as_tensor(bg_e, x), x)
    return x + b


def add_bias_dn(dn: Tensor, bias_dn: Union[float, Tensor]) -> Tensor:
    """
    Suma pedestal/black level en DN.
    """
    x = ensure_4d(dn)
    b = _broadcast(_as_tensor(bias_dn, x), x)
    return x + b


# ============================================================
# D) Fixed Pattern Noise (opcional pero común)
# ============================================================
def apply_prnu_multiplicative(x: Tensor, prnu_map: Tensor) -> Tensor:
    """
    PRNU: ganancia fija por píxel (multiplicativo).
    prnu_map: (1|B, C, H, W), valores cerca de 1.0
    """
    x = ensure_4d(x)
    if prnu_map.ndim != 4:
        raise ValueError("prnu_map debe ser (1|B, C, H, W)")
    return x * prnu_map.to(device=x.device, dtype=x.dtype)


def apply_dsnu_additive(x: Tensor, dsnu_map: Tensor) -> Tensor:
    """
    DSNU: offset fijo por píxel (aditivo).
    dsnu_map: (1|B, C, H, W)
    """
    x = ensure_4d(x)
    if dsnu_map.ndim != 4:
        raise ValueError("dsnu_map debe ser (1|B, C, H, W)")
    return x + dsnu_map.to(device=x.device, dtype=x.dtype)


# ============================================================
# E) Shot noise (REAL vs DIFF)
# ============================================================
def shot_noise_poisson(
    lam_e: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    REAL: Poisson(λ). No diferenciable.

    El generator permite controlar explícitamente
    la secuencia aleatoria del ruido de cámara.
    """
    x = ensure_4d(lam_e)
    x = clamp_nonneg(x, 0.0)

    return torch.poisson(
        x,
        generator=generator,
    )


def shot_noise_gaussian_approx(
    lam_e: Tensor,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    DIFF: aproximación común del Poisson:
      y = λ + sqrt(λ) * N(0,1)
    """
    x = ensure_4d(lam_e)
    x = clamp_nonneg(x, eps)
    n = randn_like_compat(x, generator=generator)
    return x + torch.sqrt(x) * n


# ============================================================
# F) Read noise (en electrones)
# ============================================================
def read_noise_e(
    e: Tensor,
    sigma_e: Union[float, Tensor],
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Ruido de lectura: e + N(0, sigma_e)
    """
    x = ensure_4d(e)
    s = _broadcast(_as_tensor(sigma_e, x), x)
    n = randn_like_compat(x, generator=generator)
    return x + s * n


# ============================================================
# G) Saturación (full well)
# ============================================================
def clip_full_well_e(e: Tensor, full_well_e: Union[float, Tensor]) -> Tensor:
    """
    Clipping físico en electrones.
    Compatible con torch antiguos: evita clamp(float, Tensor).
    """
    x = ensure_4d(e)
    fw = _broadcast(_as_tensor(full_well_e, x), x)

    # clamp a [0, fw] sin mezclar float con Tensor
    x = torch.clamp_min(x, 0.0)
    x = torch.minimum(x, fw)
    return x



# ============================================================
# H) Conversión e- -> DN
# ============================================================
def electrons_to_dn(e: Tensor, gain_e_per_dn: Union[float, Tensor], eps: float = 1e-12) -> Tensor:
    """
    DN = e / gain. gain_e_per_dn = electrones por DN.
    """
    x = ensure_4d(e)
    g = _broadcast(_as_tensor(gain_e_per_dn, x), x).clamp_min(eps)
    return x / g


# ============================================================
# I) Cuantización ADC (REAL vs STE)
# ============================================================
def adc_quantize_round(dn_analog: Tensor, bits: int, clip: bool = True) -> Tensor:
    """
    REAL: round + clamp a [0, 2^bits-1]
    """
    x = ensure_4d(dn_analog)
    max_dn = float(2**bits - 1)
    y = torch.round(x)
    if clip:
        y = y.clamp(0.0, max_dn)
    return y


def round_ste(x: Tensor) -> Tensor:
    """
    STE para round:
      forward: round(x)
      backward: ~ identidad
    """
    return x + (torch.round(x) - x).detach()


def adc_quantize_ste(dn_analog: Tensor, bits: int, clip: bool = True) -> Tensor:
    """
    DIFF: cuantización con STE.
    """
    x = ensure_4d(dn_analog)
    max_dn = float(2**bits - 1)
    y = round_ste(x)
    if clip:
        y = y.clamp(0.0, max_dn)
    return y


# ============================================================
# J) Modos Mono8 / Mono12 / Mono16(cont.)
# ============================================================
def quantize_by_mode(
    dn_analog: Tensor,
    mode: Literal["Mono8", "Mono12", "Mono16"],
    *,
    use_ste: bool = False,
    align: AlignMode = "lsb",
) -> Tensor:
    """
    Aplica cuantización según modo:
    - Mono8  -> 8 bits (0..255)
    - Mono12 -> 12 bits (0..4095)
    - Mono16 -> contenedor uint16: por defecto 12 bits efectivos y alineación.
    """
    x = ensure_4d(dn_analog)

    if mode == "Mono8":
        bits = 8
    elif mode == "Mono12":
        bits = 12
    elif mode == "Mono16":
        # IMX426 ADC máx 12-bit. Mono16 suele ser contenedor.
        bits = 12
    else:
        raise ValueError("mode debe ser 'Mono8', 'Mono12' o 'Mono16'")

    q = adc_quantize_ste(x, bits) if use_ste else adc_quantize_round(x, bits)

    if mode != "Mono16":
        return q

    # Representación en 16-bit (alineación):
    if align == "lsb":
        return q  # 0..4095 dentro de uint16
    elif align == "msb":
        shift = 16 - bits  # 4
        return q * (2.0 ** shift)  # left shift (<<4)
    else:
        raise ValueError("align debe ser 'lsb' o 'msb'")


# ============================================================
# K) Pipeline modular
# ============================================================
Step = Tuple[Callable[..., Tensor], Dict[str, Any]]

@dataclass
class NoisePipeline:
    steps: List[Step]

    def __call__(self, x: Tensor) -> Tensor:
        for fn, kwargs in self.steps:
            x = fn(x, **kwargs)
        return x

# ============================================================
# L) Camera Noise Augmenter general
#    Entrada: cualquier tensor (B,C,H,W)
#    Ejemplos:
#       (B,1,N,N)  -> imagen propagada completa
#       (B,4,N,N)  -> 4 pupilas crop
#
#    Sin blur, sin pixel response, sin full well.
#    Mantiene:
#       - peak_e variable
#       - background variable
#       - shot noise
#       - readout noise
#       - PRNU/DSNU opcional
#       - cuantización ADC siempre activa
#       - auto-gain para evitar saturación artificial
# ============================================================

SignalMode = Literal[
    "low",
    "normal",
    "good",
]

ParamMode = Literal[
    "per_batch",
    "per_sample",
    "per_channel",
]

ShotNoiseMode = Literal[
    "poisson",
    "gaussian",
]

OutputMode = Literal[
    "Mono8",
    "Mono12",
    "Mono16",
]

SignalScaling = Literal[
    "peak",
    "flux",
    "linear",
]

SignalScope = Literal[
    "sample",
    "channel",
]

@dataclass
class Range:
    """
    Rango uniforme o log-uniforme.
    """
    low: float
    high: float
    log: bool = False

    def sample(
        self,
        shape: Tuple[int, ...],
        like: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:

        u = torch.rand(
            shape,
            device=like.device,
            dtype=like.dtype,
            generator=generator,
        )

        if self.log:
            lo = torch.log(torch.tensor(self.low, device=like.device, dtype=like.dtype))
            hi = torch.log(torch.tensor(self.high, device=like.device, dtype=like.dtype))
            return torch.exp(lo + (hi - lo) * u)

        return self.low + (self.high - self.low) * u


@dataclass
class CameraSignalDomain:
    """
    Signal/background regime.
    """
    signal_e: Range
    bg_e: Range


@dataclass
class CameraElectronics:
    """
    Physical/electronic detector properties.
    """
    gain_e_per_dn: Range
    read_sigma_e: Range
    bias_dn: Range

    full_well_e: Optional[Range] = None

    output_mode: OutputMode = "Mono8"
    mono16_align: AlignMode = "lsb"
    use_ste_adc: bool = False

    add_prnu: bool = True
    prnu_sigma: float = 0.005

    add_dsnu: bool = True
    dsnu_sigma_e: float = 0.2


@dataclass
class CameraNoiseAugmentConfig:

    low: CameraSignalDomain
    normal: CameraSignalDomain
    good: CameraSignalDomain

    electronics: CameraElectronics

    p_low: float = 0.30
    p_normal: float = 0.55
    p_good: float = 0.15

    signal_scaling: SignalScaling = "peak"
    signal_scope: SignalScope = "sample"

    signal_parameter_mode: ParamMode = "per_sample"
    electronics_parameter_mode: ParamMode = "per_batch"

    shot_noise: ShotNoiseMode = "poisson"

    return_metadata: bool = False

def _adc_max_dn(mode: OutputMode) -> float:
    if mode == "Mono8":
        return 255.0
    elif mode == "Mono12":
        return 4095.0
    elif mode == "Mono16":
        # Asumiendo 12 bits efectivos dentro del contenedor Mono16.
        return 4095.0
    else:
        raise ValueError("output_mode debe ser 'Mono8', 'Mono12' o 'Mono16'")


def _param_shape(x: Tensor, mode: ParamMode) -> Tuple[int, int]:
    """
    Devuelve la forma de los parámetros antes de llevarlos a (B,C,1,1).
    """
    x = ensure_4d(x)
    B, C, _, _ = x.shape

    if mode == "per_batch":
        return (1, 1)

    if mode == "per_sample":
        return (B, 1)

    if mode == "per_channel":
        return (B, C)

    raise ValueError("parameter_mode debe ser 'per_batch', 'per_sample' o 'per_channel'")


def _broadcast_param(p: Tensor, x: Tensor) -> Tensor:
    """
    Convierte parámetros (1,1), (B,1) o (B,C) a (B,C,1,1).
    """
    if p.ndim != 2:
        raise ValueError(f"Se esperaba parámetro 2D. Llegó {tuple(p.shape)}")

    return p[:, :, None, None].to(device=x.device, dtype=x.dtype)


def _choose_signal_domain(
    cfg: CameraNoiseAugmentConfig,
    x: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[str, CameraSignalDomain]:
    """
    Elige low / normal / good para toda la pasada.
    """
    probs = torch.tensor(
        [cfg.p_low, cfg.p_normal, cfg.p_good],
        device=x.device,
        dtype=x.dtype,
    )
    probs = probs / probs.sum()

    idx = torch.multinomial(
        probs,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).item()

    if idx == 0:
        return "low", cfg.low
    elif idx == 1:
        return "normal", cfg.normal
    else:
        return "good", cfg.good


def _sample_signal_params(
    domain: CameraSignalDomain,
    x: Tensor,
    parameter_mode: ParamMode,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:

    shape = _param_shape(
        x,
        parameter_mode,
    )

    return {
        "signal_e": domain.signal_e.sample(
            shape,
            x,
            generator,
        ),

        "bg_e": domain.bg_e.sample(
            shape,
            x,
            generator,
        ),
    }


def _sample_electronics_params(
    electronics: CameraElectronics,
    x: Tensor,
    parameter_mode: ParamMode,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Optional[Tensor]]:

    shape = _param_shape(
        x,
        parameter_mode,
    )

    params = {
        "gain_e_per_dn":
            electronics.gain_e_per_dn.sample(
                shape,
                x,
                generator,
            ),

        "read_sigma_e":
            electronics.read_sigma_e.sample(
                shape,
                x,
                generator,
            ),

        "bias_dn":
            electronics.bias_dn.sample(
                shape,
                x,
                generator,
            ),
    }

    if electronics.full_well_e is None:
        params["full_well_e"] = None

    else:
        params["full_well_e"] = (
            electronics.full_well_e.sample(
                shape,
                x,
                generator,
            )
        )

    return params

def auto_electrons_to_dn_no_saturation(
    e: Tensor,
    bias_dn: Tensor,
    output_mode: OutputMode,
    *,
    headroom: float = 0.90,
    min_gain_e_per_dn: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Convierte electrones a DN ajustando automáticamente el gain para que
    la imagen quepa dentro del rango ADC.

    e:
        (B,C,H,W) electrones medidos.

    bias_dn:
        (B,C,1,1) bias digital.

    Retorna:
        dn_analog:
            Imagen en DN antes de cuantización.

        gain_e_per_dn:
            Gain efectivo usado, en e-/DN, forma (B,C,1,1).
    """
    x = ensure_4d(e)

    max_dn = _adc_max_dn(output_mode)
    usable_max_dn = max_dn * headroom

    # Máximo por muestra/canal.
    e_max = x.amax(
        dim=(-3, -2, -1),
        keepdim=True,
    ).clamp_min(
        min_gain_e_per_dn
    )
    # Rango disponible después del bias.
    available_dn = (usable_max_dn - bias_dn).clamp_min(1.0)

    gain_e_per_dn = (e_max / available_dn).clamp_min(min_gain_e_per_dn)

    dn_analog = x / gain_e_per_dn + bias_dn

    return dn_analog, gain_e_per_dn

def electrons_to_dn_fixed_gain(
    e: Tensor,
    bias_dn: Tensor,
    gain_e_per_dn: Tensor,
) -> Tensor:
    """
    Convert electrons to DN using a gain fixed for the
    entire camera sequence.

    Parameters
    ----------
    e:
        Measured electrons, shape (B,C,H,W).

    bias_dn:
        Digital bias, broadcastable to (B,C,H,W).

    gain_e_per_dn:
        Fixed conversion gain [e-/DN], broadcastable to
        (B,C,H,W).
    """
    x = ensure_4d(e)

    gain = gain_e_per_dn.to(
        device=x.device,
        dtype=x.dtype,
    ).clamp_min(1e-12)

    bias = bias_dn.to(
        device=x.device,
        dtype=x.dtype,
    )

    return x / gain + bias

@dataclass
class CameraNoiseAugmenter:
    cfg: CameraNoiseAugmentConfig
    generator: Optional[torch.Generator] = None

    _prnu_map: Optional[Tensor] = field(
        default=None,
        init=False,
        repr=False,
    )

    _dsnu_map: Optional[Tensor] = field(
        default=None,
        init=False,
        repr=False,
    )

    _signal_mode: Optional[str] = field(
        default=None,
        init=False,
        repr=False,
    )

    _sequence_params: Optional[Dict[str, Tensor]] = field(
        default=None,
        init=False,
        repr=False,
    )

    _sequence_shape: Optional[Tuple[int, int, int, int]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def start_sequence(
        self,
        reference: Tensor,
    ) -> None:
        """
        Generate the fixed-pattern maps for one camera sequence.

        PRNU and DSNU remain constant for all frames belonging
        to the same open/closed-loop realization.

        A new call generates a new virtual camera realization.
        """
        x = ensure_4d(reference)
        
        self._sequence_shape = tuple(
            int(value)
            for value in x.shape
        )

        _, channels, height, width = x.shape

        map_shape = (
            1,
            channels,
            height,
            width,
        )

        # --------------------------------------------------
        # PRNU: fixed multiplicative pixel response
        # --------------------------------------------------
        if (
            self.cfg.electronics.add_prnu
            and self.cfg.electronics.prnu_sigma > 0
        ):
            prnu_noise = torch.randn(
                map_shape,
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )

            self._prnu_map = (
                1.0
                + self.cfg.electronics.prnu_sigma * prnu_noise
            ).clamp_min(0.0)

        else:
            self._prnu_map = None

        # --------------------------------------------------
        # DSNU: fixed additive pixel offset [electrons]
        # --------------------------------------------------
        if (
            self.cfg.electronics.add_dsnu
            and self.cfg.electronics.dsnu_sigma_e > 0
        ):
            dsnu_noise = torch.randn(
                map_shape,
                device=x.device,
                dtype=x.dtype,
                generator=self.generator,
            )

            self._dsnu_map = (
                self.cfg.electronics.dsnu_sigma_e
                * dsnu_noise
            )

        else:
            self._dsnu_map = None
        # --------------------------------------------------
        # Camera operating regime
        # --------------------------------------------------
        signal_mode, domain = _choose_signal_domain(
            self.cfg,
            x,
            self.generator,
        )

        self._signal_mode = signal_mode

        # --------------------------------------------------
        # Camera parameters fixed during this sequence
        # --------------------------------------------------
        signal_params = _sample_signal_params(
            domain,
            x,
            parameter_mode=(
                self.cfg.signal_parameter_mode
            ),
            generator=self.generator,
        )

        electronics_params = _sample_electronics_params(
            self.cfg.electronics,
            x,
            parameter_mode=(
                self.cfg.electronics_parameter_mode
            ),
            generator=self.generator,
        )

        self._sequence_params = {
            **signal_params,
            **electronics_params,
        }

    def __call__(self, I_unit: Tensor):
        """
        I_unit:
            Tensor limpio de intensidad con forma (B,C,H,W).

        Puede ser, por ejemplo:
            (B,1,N,N)
            (B,4,N,N)
            (B,C,N,N)

        Retorna:
            Tensor cuantizado con ruido, misma forma (B,C,H,W).
        """
        x = ensure_4d(I_unit)
        x = clamp_nonneg(x, 0.0)

        # ----------------------------------------------------
        # 1) Reutilizar estado fijo de esta secuencia
        # ----------------------------------------------------
        if (
            self._signal_mode is None
            or self._sequence_params is None
            or self._sequence_shape is None
        ):
            raise RuntimeError(
                "CameraNoiseAugmenter.start_sequence(reference) "
                "must be called before applying camera noise."
            )

        current_shape = tuple(
            int(value)
            for value in x.shape
        )

        if current_shape != self._sequence_shape:
            raise RuntimeError(
                "Camera input shape changed inside the same "
                "camera sequence: "
                f"expected {self._sequence_shape}, "
                f"got {current_shape}."
            )

        signal_mode = self._signal_mode
        params = self._sequence_params

        signal_e = _broadcast_param(
            params["signal_e"],
            x,
        )

        bg_e = _broadcast_param(
            params["bg_e"],
            x,
        )

        read_sigma_e = _broadcast_param(
            params["read_sigma_e"],
            x,
        )

        bias_dn = _broadcast_param(
            params["bias_dn"],
            x,
        )

        gain_e_per_dn = _broadcast_param(
            params["gain_e_per_dn"],
            x,
        )

        if params["full_well_e"] is None:
            full_well_e = None
        else:
            full_well_e = _broadcast_param(
                params["full_well_e"],
                x,
            )
        # ----------------------------------------------------
        # 2) Optical intensity -> expected electrons
        # ----------------------------------------------------
        lam_e = to_expected_electrons(
            x,
            signal_e=signal_e,
            scaling=self.cfg.signal_scaling,
            scope=self.cfg.signal_scope,
        )
        # ----------------------------------------------------
        # 3) PRNU fijo durante la secuencia
        # ----------------------------------------------------
        if self._prnu_map is not None:
            lam_e = apply_prnu_multiplicative(
                lam_e,
                self._prnu_map,
            )

        # ----------------------------------------------------
        # 4) Background / dark
        # ----------------------------------------------------
        lam_e = add_background_e(lam_e, bg_e=bg_e)

        # ----------------------------------------------------
        # 5) Shot noise
        # ----------------------------------------------------
        if self.cfg.shot_noise == "poisson":
            e = shot_noise_poisson(
                lam_e,
                generator=self.generator,
            )

        elif self.cfg.shot_noise == "gaussian":
            e = shot_noise_gaussian_approx(
                lam_e,
                generator=self.generator,
            )

        else:
            raise ValueError(
                "shot_noise debe ser 'poisson' o 'gaussian'"
            )

        # ----------------------------------------------------
        # 6) Fixed DSNU
        # ----------------------------------------------------
        if self._dsnu_map is not None:
            e = apply_dsnu_additive(
                e,
                self._dsnu_map,
            )

        # ----------------------------------------------------
        # 7) Physical full-well saturation
        # ----------------------------------------------------
        if full_well_e is not None:
            e = clip_full_well_e(
                e,
                full_well_e=full_well_e,
            )
        else:
            e = e.clamp_min(0.0)

        # ----------------------------------------------------
        # 8) Temporal readout noise
        # ----------------------------------------------------
        e = read_noise_e(
            e,
            sigma_e=read_sigma_e,
            generator=self.generator,
        )

        # Read noise can produce negative values.
        e = e.clamp_min(0.0)
        # ----------------------------------------------------
        # 9) Electrons -> DN with configured camera gain
        # ----------------------------------------------------
        dn = electrons_to_dn(
            e,
            gain_e_per_dn=gain_e_per_dn,
        )

        dn = add_bias_dn(
            dn,
            bias_dn=bias_dn,
        )

        # ----------------------------------------------------
        # 10) ADC quantization
        # ----------------------------------------------------
        dn_q = quantize_by_mode(
            dn,
            mode=self.cfg.electronics.output_mode,
            use_ste=self.cfg.electronics.use_ste_adc,
            align=self.cfg.electronics.mono16_align,
        )

        if self.cfg.return_metadata:
            metadata = {
                "signal_mode":
                    signal_mode,

                "signal_scaling":
                    self.cfg.signal_scaling,

                "signal_scope":
                    self.cfg.signal_scope,

                "signal_e":
                    params["signal_e"].detach(),

                "bg_e":
                    params["bg_e"].detach(),

                "read_sigma_e":
                    params["read_sigma_e"].detach(),

                "bias_dn":
                    params["bias_dn"].detach(),

                "gain_e_per_dn":
                    params["gain_e_per_dn"].detach(),

                "full_well_e": (
                    None
                    if params["full_well_e"] is None
                    else params["full_well_e"].detach()
                ),

                "output_mode":
                    self.cfg.electronics.output_mode,

                "input_shape":
                    tuple(I_unit.shape),
            }
            return dn_q, metadata

        return dn_q