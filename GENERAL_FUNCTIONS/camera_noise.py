from __future__ import annotations
from dataclasses import dataclass
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
def to_expected_electrons_from_unit(
    I_unit: Tensor,
    peak_e: Union[float, Tensor],
    eps: float = 1e-12,
) -> Tensor:
    """
    Convierte intensidad "arbitraria" (I_unit >= 0) a electrones esperados λ_e.
    Método típico: normalizar por max por imagen y definir peak_e.
      λ_e = (I / max(I)) * peak_e
    """
    I = ensure_4d(I_unit)
    I = clamp_nonneg(I, 0.0)

    Imax = I.amax(
        dim=(-3, -2, -1),
        keepdim=True,
    ).clamp_min(eps)

    Irel = I / Imax

    pe = _broadcast(_as_tensor(peak_e, I), I)
    lam_e = Irel * pe
    return lam_e


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
def shot_noise_poisson(lam_e: Tensor) -> Tensor:
    """
    REAL: Poisson(λ). No diferenciable.
    """
    x = ensure_4d(lam_e)
    x = clamp_nonneg(x, 0.0)
    return torch.poisson(x)


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
# L) Augmenter de ruido realista para cámara WFS
#    Entrada: (B,4,N,N), una pupila por canal
# ============================================================
from dataclasses import dataclass
from typing import Tuple, Optional, Literal


SignalMode = Literal["low", "normal", "good", "mixed"]


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

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Dict
import torch

SignalMode = Literal["low", "normal", "good"]
ParamMode = Literal["per_batch", "per_sample", "per_channel"]
ShotNoiseMode = Literal["poisson", "gaussian"]
OutputMode = Literal["Mono8", "Mono12", "Mono16"]


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
class CameraNoiseDomain:
    """
    Dominio de ruido para un régimen de señal.

    peak_e:
        Nivel máximo de electrones esperados.
        Controla baja / normal / buena señal.

    bg_e:
        Background o dark actual en electrones.

    read_sigma_e:
        Ruido de lectura en electrones RMS.

    bias_dn:
        Offset digital antes de cuantizar.
    """
    peak_e: Range
    bg_e: Range
    read_sigma_e: Range
    bias_dn: Range


@dataclass
class CameraNoiseAugmentConfig:
    """
    Configuración general.

    parameter_mode:
        "per_batch"   -> mismos parámetros para todo el batch.
        "per_sample"  -> parámetros distintos por muestra.
        "per_channel" -> parámetros distintos por canal.
                         Para (B,4,N,N), cada pupila puede tener ruido distinto.
                         Para (B,1,N,N), funciona igual sin problema.

    output_mode:
        "Mono8"  -> cuantiza a 0..255.
        "Mono12" -> cuantiza a 0..4095.
        "Mono16" -> contenedor 16 bit con 12 bits efectivos.

    auto_gain:
        Si True, calcula gain_e_per_dn automáticamente para que la imagen
        entre dentro del rango ADC sin saturar.
    """
    low: CameraNoiseDomain
    normal: CameraNoiseDomain
    good: CameraNoiseDomain

    p_low: float = 0.25
    p_normal: float = 0.50
    p_good: float = 0.25

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
) -> Tuple[str, CameraNoiseDomain]:
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


def _sample_domain_params(
    domain: CameraNoiseDomain,
    x: Tensor,
    parameter_mode: ParamMode,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tensor]:

    shape = _param_shape(x, parameter_mode)

    return {
        "peak_e": domain.peak_e.sample(shape, x, generator),
        "bg_e": domain.bg_e.sample(shape, x, generator),
        "read_sigma_e": domain.read_sigma_e.sample(shape, x, generator),
        "bias_dn": domain.bias_dn.sample(shape, x, generator),
    }


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


@dataclass
class CameraNoiseAugmenter:
    cfg: CameraNoiseAugmentConfig
    generator: Optional[torch.Generator] = None

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
        # 1) Elegir régimen de señal y muestrear parámetros
        # ----------------------------------------------------
        signal_mode, domain = _choose_signal_domain(
            self.cfg,
            x,
            self.generator,
        )

        params = _sample_domain_params(
            domain,
            x,
            parameter_mode=self.cfg.parameter_mode,
            generator=self.generator,
        )

        peak_e = _broadcast_param(params["peak_e"], x)
        bg_e = _broadcast_param(params["bg_e"], x)
        read_sigma_e = _broadcast_param(params["read_sigma_e"], x)
        bias_dn = _broadcast_param(params["bias_dn"], x)

        # ----------------------------------------------------
        # 2) Intensidad arbitraria -> electrones esperados
        # ----------------------------------------------------
        lam_e = to_expected_electrons_from_unit(x, peak_e=peak_e)

        # ----------------------------------------------------
        # 3) PRNU opcional
        # ----------------------------------------------------
        if self.cfg.add_prnu and self.cfg.prnu_sigma > 0:
            prnu = 1.0 + self.cfg.prnu_sigma * randn_like_compat(
                lam_e,
                self.generator,
            )
            lam_e = lam_e * prnu.clamp_min(0.0)

        # ----------------------------------------------------
        # 4) Background / dark
        # ----------------------------------------------------
        lam_e = add_background_e(lam_e, bg_e=bg_e)

        # ----------------------------------------------------
        # 5) Shot noise
        # ----------------------------------------------------
        if self.cfg.shot_noise == "poisson":
            e = shot_noise_poisson(lam_e)
        elif self.cfg.shot_noise == "gaussian":
            e = shot_noise_gaussian_approx(
                lam_e,
                generator=self.generator,
            )
        else:
            raise ValueError("shot_noise debe ser 'poisson' o 'gaussian'")

        # ----------------------------------------------------
        # 6) DSNU opcional
        # ----------------------------------------------------
        if self.cfg.add_dsnu and self.cfg.dsnu_sigma_e > 0:
            dsnu = self.cfg.dsnu_sigma_e * randn_like_compat(
                e,
                self.generator,
            )
            e = e + dsnu

        # ----------------------------------------------------
        # 7) Readout noise
        # ----------------------------------------------------
        e = read_noise_e(
            e,
            sigma_e=read_sigma_e,
            generator=self.generator,
        )

        # No hay full well ni saturación física.
        e = e.clamp_min(0.0)

        # ----------------------------------------------------
        # 8) Electron -> DN con auto-gain para no saturar
        # ----------------------------------------------------
        if self.cfg.auto_gain:
            dn, gain_e_per_dn = auto_electrons_to_dn_no_saturation(
                e,
                bias_dn=bias_dn,
                output_mode=self.cfg.output_mode,
                headroom=self.cfg.adc_headroom,
                min_gain_e_per_dn=self.cfg.min_gain_e_per_dn,
            )
        else:
            raise NotImplementedError(
                "En esta versión se usa auto_gain=True para asegurar que "
                "la imagen siempre cabe dentro del rango ADC."
            )

        # ----------------------------------------------------
        # 9) Cuantización ADC, siempre activa
        # ----------------------------------------------------
        dn_q = quantize_by_mode(
            dn,
            mode=self.cfg.output_mode,
            use_ste=self.cfg.use_ste_adc,
            align=self.cfg.mono16_align,
        )

        if self.cfg.return_metadata:
            metadata = {
                "signal_mode": signal_mode,
                "peak_e": params["peak_e"].detach(),
                "bg_e": params["bg_e"].detach(),
                "read_sigma_e": params["read_sigma_e"].detach(),
                "bias_dn": params["bias_dn"].detach(),
                "gain_e_per_dn": gain_e_per_dn.detach(),
                "adc_headroom": self.cfg.adc_headroom,
                "output_mode": self.cfg.output_mode,
                "input_shape": tuple(I_unit.shape),
            }
            return dn_q, metadata

        return dn_q