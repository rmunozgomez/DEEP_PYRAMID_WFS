
from __future__ import annotations

import math
import warnings
from typing import Iterable, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DirectionConvention = Literal["cartesian", "oopao"]
PropagationMode = Literal["geometric", "asm_delta"]
DeltaMode = Literal["final", "per_step"]
SubharmonicMode = Literal["oopao", "full"]
FrozenFlowMode = Literal["analytic", "periodic_screen"]

# VERIFIED INIT ARGUMENT: Atmosphere.__init__ includes delta_mode.

__version__ = "2026.07.23-online-profile-field-v3"


class Atmosphere(nn.Module):
    """
    Batched multilayer frozen-flow atmosphere implemented entirely in PyTorch.

    This class is deliberately equivalent to the previous NumPy workflow:

    1. Every layer is generated with the same global r0 and L0.
    2. Every generated layer is scaled by sqrt(fractional_r0[layer]).
    3. The high-frequency component follows the centered FFT convention used
       by the original `ft_phase_screen`.
    4. By default, the subharmonic component reproduces AOtools
       ``ft_sh_phase_screen`` exactly: all 3x3 frequencies are evaluated at
       each level and the zero-frequency coefficient is set to zero, leaving
       eight effective complex modes per level.
    5. High frequencies and subharmonics are kept separate and evolved with
       the same wind vector according to frozen flow.

    Optional OOPAO-style sequential ASM propagation provides scintillation and delta-phase extraction.

    Parameters
    ----------
    batch_size
        Number B of independent atmospheres.

    resolution
        Spatial resolution N. The public phase, intensity and field have shape [1, B, N, N].

    telescope_diameter
        Physical size represented by the phase grid [m].

    frame_rate
        Temporal sampling frequency [Hz].

    r0
        Fried-parameter specification [m]. A positive scalar keeps one fixed
        r0 for every realization in the batch. A two-value sequence [min, max]
        samples one independent r0 ~ U(min, max) for each batch realization
        whenever gen() creates a new batch. The sampled values remain fixed
        during update() so each temporal sequence keeps its own r0.

    L0
        Outer scale [m].

    wind_speed
        Wind speed per layer [m/s].

    fractional_r0
        Relative layer weights. To reproduce the previous implementation,
        these values are used directly through sqrt(fractional_r0) and are
        not normalized unless `normalize_fractional_r0=True`.

    altitude
        Layer altitudes [m]. Stored for future propagation support.

    wind_direction
        Wind direction per layer [deg].

    direction_convention
        "oopao" (default):
            vX = speed*sin(direction), vY = speed*cos(direction).
            Therefore 0 deg -> +Y and 90 deg -> +X, as in OOPAO.

        "cartesian":
            0 deg -> +X, 90 deg -> +Y.

    n_subharmonic_levels
        Number of subharmonic frequency levels. The previous implementation
        used 3.

    normalize_fractional_r0
        If True, divide layer weights by their sum before using them.
        False reproduces the previous code literally.

    remove_piston
        If True, remove piston independently from the high-frequency and
        subharmonic components of every layer.

    store_components
        If True, expose phase_layers, phase_hi_layers and phase_sh_layers.
        Disable it to reduce peak memory when only `phase` is required.

    device
        CPU or CUDA device.

    dtype
        torch.float32 is recommended. The corresponding complex dtype is
        torch.complex64.

    seed
        Optional seed for reproducible Torch generation.

    temporal_reanchor_interval
        Number of one-frame recursive updates between absolute temporal
        re-anchoring operations. Use 0 to disable re-anchoring. A value such
        as 256 keeps long complex64 runs numerically stable while preserving
        the fast recursive update path.

    asm_min_physical_margin
        Minimum physical turbulent guard band per side [m] used when
        ``asm_extra_pixels="auto"``. It is converted to pixels through the
        telescope sampling and combined with the Fresnel and N/4 safeguards.
        Manual ``asm_extra_pixels`` values ignore this parameter.

    frozen_flow_mode
        ``"analytic"`` preserves the previous implementation: high frequencies
        are periodic on the internal FFT grid while subharmonics are evaluated
        as analytic modes with their naturally larger periods.

        ``"periodic_screen"`` preserves the same t=0 realization but stores the
        complete initial subharmonic map on the internal screen. High and low
        frequencies are then shifted by the same absolute Fourier operator.
        This mode is intended when the finite generated screen must move as one
        rigid periodic object and return after one complete screen crossing.

    Public state
    ------------
    phase : Tensor [1, B, N, N]
        Sum of all current layers.

    phase_layers : Tensor [B, L, N, N] or None
        Total phase of each layer when store_components=True.

    phase_hi_layers : Tensor [B, L, N, N] or None
        High-frequency contribution per layer.

    phase_sh_layers : Tensor [B, L, N, N] or None
        Subharmonic contribution per layer.

    frame_index : int
        Current temporal frame, with gen() corresponding to frame zero.

    time_s : float
        Current physical time [s].
    """

    def __init__(
        self,
        *,
        batch_size: int,
        resolution: int,
        telescope_diameter: float,
        frame_rate: float,
        r0: float | Sequence[float],
        L0: float,
        wind_speed: Iterable[float],
        fractional_r0: Iterable[float],
        altitude: Iterable[float],
        wind_direction: Iterable[float],
        l0: float = 1e-10,
        n_subharmonic_levels: int = 3,
        subharmonic_mode: SubharmonicMode = "full",
        direction_convention: DirectionConvention = "oopao",
        normalize_fractional_r0: bool = True,
        remove_piston: bool = False,
        store_components: bool = True,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        wavelength: float = 500e-9,
        r0_reference_wavelength: float = 500e-9,
        propagation_mode: PropagationMode = "geometric",
        asm_extra_pixels: int | Literal["auto"] | None = "auto",
        asm_min_physical_margin: float = 0.75,
        asm_padding_factor: float = 2.0,
        delta_mode: DeltaMode = "final",
        delta_wrap_warning_threshold: float = 1.9 * math.pi,
        temporal_reanchor_interval: int = 256,
        frozen_flow_mode: FrozenFlowMode = "analytic",
    ) -> None:
        super().__init__()

        self.batch_size = int(batch_size)
        self.resolution = int(resolution)
        self.telescope_diameter = float(telescope_diameter)
        self.frame_rate = float(frame_rate)

        # r0 has two public modes:
        #   scalar      -> one fixed r0 shared by all B realizations
        #   [min, max]  -> one independent r0 ~ U(min, max) per realization
        # self.r0 remains a scalar PSD reference so all spectral statistics can
        # still be cached once. Individual screens are scaled exactly in gen().
        (
            self.r0_mode,
            self.r0_min,
            self.r0_max,
            self.r0,
        ) = self._parse_r0_spec(r0)

        self.L0 = float(L0)
        self.l0 = float(l0)
        self.n_subharmonic_levels = int(n_subharmonic_levels)
        self.subharmonic_mode = subharmonic_mode
        self.direction_convention = direction_convention
        self.normalize_fractional_r0 = bool(normalize_fractional_r0)
        self.remove_piston = bool(remove_piston)
        self.store_components = bool(store_components)

        self.wavelength = float(wavelength)
        self.r0_reference_wavelength = float(r0_reference_wavelength)
        self.rad2arcsec = (180.0 / math.pi) * 3600.0
        self.propagation_mode = propagation_mode
        # ``asm_extra_pixels`` can be fixed manually or determined from the
        # physical ASM sampling. ``None`` is treated as ``"auto"``.
        self._asm_extra_pixels_request = (
            "auto" if asm_extra_pixels is None else asm_extra_pixels
        )
        self.asm_extra_pixels_mode = (
            "auto" if self._asm_extra_pixels_request == "auto" else "manual"
        )
        self.asm_extra_pixels = (
            0
            if self.asm_extra_pixels_mode == "auto"
            else int(self._asm_extra_pixels_request)
        )
        # Minimum physical turbulent guard band per side [m] used only when
        # asm_extra_pixels="auto". The default 0.75 m preserves the validated
        # behavior: 32 px for D=3 m, N=128 and approximately 128 px for
        # D=0.6 m, N=128 after dyadic rounding.
        self.asm_min_physical_margin = float(asm_min_physical_margin)
        self.asm_padding_factor = float(asm_padding_factor)
        self.delta_mode = delta_mode
        # OOPAO propagation: paraxial ASM, equal input/output pitch, no spectral mask.
        self.asm_model = "paraxial"
        self.asm_bandlimit = False
        self.delta_wrap_warning_threshold = float(
            delta_wrap_warning_threshold
        )
        self.temporal_reanchor_interval = int(temporal_reanchor_interval)
        self.frozen_flow_mode = frozen_flow_mode

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")

        self.real_dtype = dtype
        self.complex_dtype = (
            torch.complex64
            if dtype == torch.float32
            else torch.complex128
        )

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

        self._validate_scalar_parameters()
        self._validate_propagation_parameters()

        speed = torch.as_tensor(
            list(wind_speed),
            dtype=dtype,
            device=requested_device,
        )
        fraction = torch.as_tensor(
            list(fractional_r0),
            dtype=dtype,
            device=requested_device,
        )
        altitude_tensor = torch.as_tensor(
            list(altitude),
            dtype=dtype,
            device=requested_device,
        )
        direction = torch.as_tensor(
            list(wind_direction),
            dtype=dtype,
            device=requested_device,
        )

        self.n_layers = int(fraction.numel())
        if self.n_layers == 0:
            raise ValueError("At least one layer must be defined.")

        if not (
            speed.numel()
            == fraction.numel()
            == altitude_tensor.numel()
            == direction.numel()
        ):
            raise ValueError(
                "wind_speed, fractional_r0, altitude and wind_direction "
                "must have the same length."
            )

        if torch.any(fraction < 0):
            raise ValueError("fractional_r0 values cannot be negative.")

        fraction_sum = fraction.sum()
        if fraction_sum <= 0:
            raise ValueError("fractional_r0 must have a positive sum.")
        fraction_is_normalized = torch.isclose(
            fraction_sum,
            torch.ones((), dtype=dtype, device=requested_device),
            rtol=1e-5,
            atol=1e-7,
        )
        if self.normalize_fractional_r0:
            if not fraction_is_normalized:
                warnings.warn(
                    "fractional_r0 did not sum to 1 and was normalized automatically.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fraction = fraction / fraction_sum
        elif not fraction_is_normalized:
            warnings.warn(
                "fractional_r0 does not sum to 1 and is being used without "
                "normalization because normalize_fractional_r0=False.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Store all layer-dependent parameters from highest to lowest layer.
        # This prevents user input order from changing the propagation physics.
        layer_order = torch.argsort(altitude_tensor, descending=True)
        if not torch.equal(
            layer_order,
            torch.arange(layer_order.numel(), device=requested_device),
        ):
            warnings.warn(
                "Atmospheric layers were reordered from highest to lowest altitude.",
                RuntimeWarning,
                stacklevel=2,
            )
        speed = speed[layer_order]
        fraction = fraction[layer_order]
        altitude_tensor = altitude_tensor[layer_order]
        direction = direction[layer_order]

        # Telescope sampling remains D / N. In ASM mode the atmospheric screen
        # is generated on a physically larger support with the same pixel pitch.
        self.pixel_size = self.telescope_diameter / self.resolution
        self.dt = 1.0 / self.frame_rate

        # Register physical layer parameters as buffers so .to(device) works.
        self.register_buffer("wind_speed", speed)
        self.register_buffer("fractional_r0", fraction)
        self.register_buffer("altitude", altitude_tensor)
        self.register_buffer("wind_direction", direction)
        self.register_buffer(
            "layer_amplitude",
            torch.sqrt(fraction),
        )
        self.register_buffer(
            "r0_batch",
            torch.full(
                (self.batch_size,),
                self.r0_min,
                dtype=dtype,
                device=requested_device,
            ),
        )

        vx, vy = self._wind_components(speed, direction)
        self.register_buffer("vx", vx)
        self.register_buffer("vy", vy)

        # Resolve the optional automatic ASM support after altitude is available.
        self._update_asm_extra_pixels()
        self._screen_resolution = self._required_screen_resolution()

        # Static precomputations.
        self._precompute_high_frequency_statistics()
        self._precompute_subharmonic_statistics()
        self._precompute_temporal_factors()
        self._precompute_propagation_geometry()
        self._precompute_asm_kernels()

        # Device-aware RNG.
        self.generator = torch.Generator(device=requested_device)
        if seed is None:
            seed = int(torch.seed())
        self.seed = int(seed)
        self.generator.manual_seed(self.seed)

        # Dynamic spectral states.
        self._hi_spectrum_0: Optional[torch.Tensor] = None
        self._hi_spectrum_t: Optional[torch.Tensor] = None
        self._sh_coeff_0: Optional[torch.Tensor] = None
        self._sh_coeff_t: Optional[torch.Tensor] = None
        # Optional periodic representation of the complete initial SH map.
        # It is used only by frozen_flow_mode="periodic_screen" so the full
        # generated layer is advected as one finite periodic screen.
        self._sh_spectrum_0: Optional[torch.Tensor] = None
        self._sh_spectrum_t: Optional[torch.Tensor] = None
        # Constant piston removed from the initial subharmonic realization.
        # It is kept fixed in time so the SH map translates rigidly.
        self._sh_piston_0: Optional[torch.Tensor] = None

        # Public outputs.
        self.register_buffer(
            "_phase",
            torch.zeros(
                self.batch_size,
                self.resolution,
                self.resolution,
                dtype=self.real_dtype,
                device=requested_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_intensity",
            torch.ones(
                self.batch_size,
                self.resolution,
                self.resolution,
                dtype=self.real_dtype,
                device=requested_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_field",
            torch.ones(
                self.batch_size,
                self.resolution,
                self.resolution,
                dtype=self.complex_dtype,
                device=requested_device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_diffractive_delta",
            torch.zeros(
                self.batch_size,
                self.resolution,
                self.resolution,
                dtype=self.real_dtype,
                device=requested_device,
            ),
            persistent=False,
        )

        self.phase_layers: Optional[torch.Tensor] = None
        self.phase_hi_layers: Optional[torch.Tensor] = None
        self.phase_sh_layers: Optional[torch.Tensor] = None

        self.frame_index = 0
        self.time_s = 0.0

    @property
    def phase(self) -> torch.Tensor:
        """Current phase with network-ready shape [1, B, N, N]."""
        return self._phase.unsqueeze(0)

    @property
    def intensity(self) -> torch.Tensor:
        """Current intensity with network-ready shape [1, B, N, N]."""
        return self._intensity.unsqueeze(0)

    @property
    def field(self) -> torch.Tensor:
        """Current complex field with network-ready shape [1, B, N, N]."""
        return self._field.unsqueeze(0)

    @property
    def diffractive_delta(self) -> torch.Tensor:
        """Current diffractive residual with shape [1, B, N, N]."""
        return self._diffractive_delta.unsqueeze(0)

    @property
    def device(self) -> torch.device:
        return self._phase.device

    @staticmethod
    def _parse_r0_spec(
        value: float | Sequence[float] | torch.Tensor,
    ) -> tuple[str, float, float, float]:
        """Parse the two supported Fried-parameter modes.

        ``r0=0.15``
            Fixed mode. Every realization in the batch uses r0=0.15 m.

        ``r0=[0.05, 0.30]``
            Random-batch mode. Every call to :meth:`gen` draws one independent
            r0 per batch realization from U(0.05, 0.30) m. Those values remain
            fixed during subsequent :meth:`update` calls.

        The returned fourth value is a scalar reference r0 used to cache the
        von Karman PSD. In range mode the geometric mean minimizes the largest
        multiplicative rescaling required across the interval.
        """
        if torch.is_tensor(value):
            values = value.detach().flatten().cpu().tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            scalar = float(value)
            if scalar <= 0.0:
                raise ValueError("r0 must be positive.")
            return "fixed", scalar, scalar, scalar

        if len(values) != 2:
            raise ValueError(
                "r0 must be either one positive scalar or [min, max]."
            )

        low, high = float(values[0]), float(values[1])
        if low <= 0.0 or high <= 0.0:
            raise ValueError("r0 range values must be positive.")
        if high < low:
            raise ValueError("r0 range must satisfy min <= max.")
        if math.isclose(low, high, rel_tol=0.0, abs_tol=0.0):
            return "fixed", low, low, low

        reference = math.sqrt(low * high)
        return "range", low, high, reference

    def _sample_r0_batch(self) -> None:
        """Populate ``r0_batch`` for the next newly generated realization."""
        if self.r0_mode == "fixed":
            self.r0_batch.fill_(self.r0_min)
            return

        self.r0_batch.uniform_(
            self.r0_min,
            self.r0_max,
            generator=self.generator,
        )

    def _validate_scalar_parameters(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive.")
        if self.telescope_diameter <= 0:
            raise ValueError("telescope_diameter must be positive.")
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive.")
        if self.r0 <= 0:
            raise ValueError("r0 must be positive.")
        if self.L0 <= 0:
            raise ValueError("L0 must be positive.")
        if self.l0 <= 0:
            raise ValueError("l0 must be positive.")
        if self.n_subharmonic_levels <= 0:
            raise ValueError(
                "n_subharmonic_levels must be positive."
            )
        if self.subharmonic_mode not in ("oopao", "full"):
            raise ValueError(
                "subharmonic_mode must be 'oopao' or 'full'."
            )
        if self.direction_convention not in ("cartesian", "oopao"):
            raise ValueError(
                "direction_convention must be 'cartesian' or 'oopao'."
            )
        if self.temporal_reanchor_interval < 0:
            raise ValueError("temporal_reanchor_interval must be >= 0.")
        if self.frozen_flow_mode not in ("analytic", "periodic_screen"):
            raise ValueError(
                "frozen_flow_mode must be 'analytic' or 'periodic_screen'."
            )

    def _validate_propagation_parameters(self) -> None:
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        if self.r0_reference_wavelength <= 0:
            raise ValueError("r0_reference_wavelength must be positive.")
        if self.propagation_mode not in ("geometric", "asm_delta"):
            raise ValueError(
                "propagation_mode must be 'geometric' or 'asm_delta'."
            )
        if self.asm_model not in ("exact", "paraxial"):
            raise ValueError("asm_model must be 'exact' or 'paraxial'.")
        if self._asm_extra_pixels_request != "auto":
            try:
                extra_value = int(self._asm_extra_pixels_request)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "asm_extra_pixels must be a non-negative integer, 'auto', or None."
                ) from exc
            if extra_value < 0:
                raise ValueError("asm_extra_pixels must be >= 0.")
        if self.asm_min_physical_margin < 0.0:
            raise ValueError("asm_min_physical_margin must be >= 0.")
        if self.asm_padding_factor < 1.0:
            raise ValueError("asm_padding_factor must be >= 1.0.")
        if self.delta_mode not in ("final", "per_step"):
            raise ValueError("delta_mode must be 'final' or 'per_step'.")

    @staticmethod
    def _nearest_power_of_two(value: float) -> int:
        """Return the power of two nearest to a positive value."""
        if value <= 1.0:
            return 1
        lower = 2 ** math.floor(math.log2(value))
        upper = lower * 2
        return int(lower if value - lower <= upper - value else upper)

    def _automatic_asm_extra_pixels(self) -> int:
        """Estimate the physical turbulent guard band for sequential ASM.

        Three independent lower bounds are evaluated in pixels per side:

        1. Physical guard band::

               margin_physical = asm_min_physical_margin / pixel_size

           This preserves approximately the same guard distance in metres when
           telescope diameter or sampling changes. With the default 0.75 m,
           N=128 gives 32 px for D=3 m and a target of 160 px for D=0.6 m,
           which rounds to 128 px using the nearest dyadic value.

        2. Fresnel/Nyquist reach from the highest layer::

               margin_fresnel = wavelength * z_max / (2 * pixel_size**2)

        3. A small sampling safeguard of N/4 pixels.

        The maximum of these terms is rounded to the nearest power of two.
        Manual ``asm_extra_pixels`` values are unaffected.
        """
        if self.propagation_mode != "asm_delta":
            return 0

        z_max = max(0.0, float(self.altitude.max().item()))

        physical_margin = (
            self.asm_min_physical_margin / self.pixel_size
            if self.asm_min_physical_margin > 0.0
            else 0.0
        )
        fresnel_margin = (
            self.wavelength * z_max / (2.0 * self.pixel_size * self.pixel_size)
        )
        minimum_pixel_margin = max(1.0, self.resolution / 4.0)

        target = max(
            physical_margin,
            fresnel_margin,
            minimum_pixel_margin,
        )
        return self._nearest_power_of_two(target)

    def _update_asm_extra_pixels(self) -> None:
        """Resolve the active ASM margin without changing manual values."""
        if self.asm_extra_pixels_mode == "auto":
            self.asm_extra_pixels = self._automatic_asm_extra_pixels()

    def _required_screen_resolution(self) -> int:
        """Return the internal atmospheric-grid size required by the active mode."""
        if self.propagation_mode == "asm_delta":
            return self.resolution + 2 * self.asm_extra_pixels
        return self.resolution

    def _crop_screen_center(self, value: torch.Tensor) -> torch.Tensor:
        """Crop the internal atmospheric support to the telescope resolution."""
        if self._screen_resolution == self.resolution:
            return value
        start = (self._screen_resolution - self.resolution) // 2
        stop = start + self.resolution
        return value[..., start:stop, start:stop]

    def _required_asm_resolution(self) -> int:
        """Return the temporary zero-padded FFT size used by ASM."""
        target = max(
            self._screen_resolution,
            int(math.ceil(self.asm_padding_factor * self._screen_resolution)),
        )
        if (target - self._screen_resolution) % 2 != 0:
            target += 1
        return target

    def _precompute_propagation_geometry(self) -> None:
        """Precompute the persistent full-domain ASM geometry.

        Atmospheric layers are generated on ``_screen_resolution`` =
        ``resolution + 2*asm_extra_pixels``. Before being applied, every layer
        is zero-phase padded to ``asm_resolution``. The complex field remains
        on that full ASM support throughout the complete layer cascade and is
        cropped only once, at the telescope plane.
        """
        # Physical atmospheric support on which every phase screen is generated.
        self.propagation_resolution = self._screen_resolution

        # Persistent ASM support. Unlike the previous implementation, this is
        # not cropped back after each propagation gap.
        self.asm_resolution = self._required_asm_resolution()
        self.asm_padding_pixels = (
            self.asm_resolution - self._screen_resolution
        ) // 2
        self._asm_screen_start = self.asm_padding_pixels
        self._asm_screen_stop = (
            self._asm_screen_start + self._screen_resolution
        )

        # Final telescope crop is taken directly from the persistent ASM domain.
        telescope_margin = (self.asm_resolution - self.resolution) // 2
        self._crop_start = telescope_margin
        self._crop_stop = telescope_margin + self.resolution

        # Subharmonics are evaluated directly on the enlarged generated support.
        coords_prop = (
            torch.arange(
                self.propagation_resolution,
                dtype=self.real_dtype,
                device=self.wind_speed.device,
            )
            - self.propagation_resolution / 2.0
        ) * self.pixel_size
        y_prop, x_prop = torch.meshgrid(coords_prop, coords_prop, indexing="ij")
        sh_phase_prop = 2.0 * math.pi * (
            self._fx_sh[:, None, None] * x_prop[None, :, :]
            + self._fy_sh[:, None, None] * y_prop[None, :, :]
        )
        basis_sh_prop = torch.polar(
            torch.ones_like(sh_phase_prop), sh_phase_prop
        ).to(self.complex_dtype)
        self.register_buffer("_basis_sh_prop", basis_sh_prop)

        # Layers are already stored top-to-bottom; calculate every gap and the
        # final propagation from the lowest layer to the telescope at 0 m.
        order = torch.argsort(self.altitude, descending=True)
        altitude_sorted = self.altitude[order]
        next_altitude = torch.cat((altitude_sorted[1:], torch.zeros(1, dtype=self.real_dtype, device=self.wind_speed.device)))
        self.register_buffer("_propagation_order", order)
        self.register_buffer("_propagation_distance", altitude_sorted - next_altitude)

    def _precompute_asm_kernels(self) -> None:
        """Precompute centered OOPAO-style ASM kernels for each layer gap."""
        n = self.asm_resolution
        dx = self.pixel_size
        wavelength = self.wavelength
        delta_f = 1.0 / (n * dx)
        vals = (
            torch.arange(n, dtype=self.real_dtype, device=self.wind_speed.device)
            - n / 2.0
        ) * delta_f
        fy, fx = torch.meshgrid(vals, vals, indexing="ij")
        f2 = fx.square() + fy.square()

        kernels = []
        k = 2.0 * math.pi / wavelength
        for z_tensor in self._propagation_distance:
            z = float(z_tensor.item())
            if z <= 0.0:
                kernel = torch.ones((n, n), dtype=self.complex_dtype, device=self.wind_speed.device)
            elif self.asm_model == "paraxial":
                # Same transfer function used by OOPAO for equal input/output pitch.
                angle = -math.pi * wavelength * z * f2
                kernel = torch.polar(torch.ones_like(angle), angle).to(self.complex_dtype)
            else:
                root_arg = 1.0 - wavelength**2 * f2
                propagating = root_arg >= 0.0
                angle = k * z * torch.sqrt(root_arg.clamp_min(0.0))
                kernel = torch.polar(torch.ones_like(angle), angle).to(self.complex_dtype)
                kernel.mul_(propagating.to(self.complex_dtype))

            if self.asm_bandlimit and z > 0.0:
                # Optional Matsushima-style rectangular support. Disabled by default
                # because a hard cutoff can create cross-hatched ringing.
                window = n * dx
                f_limit = 1.0 / (wavelength * math.sqrt(1.0 + (2.0 * z / window) ** 2))
                band = (fx.abs() <= f_limit) & (fy.abs() <= f_limit)
                kernel.mul_(band.to(self.complex_dtype))
            kernels.append(kernel)

        self.register_buffer("_asm_kernels", torch.stack(kernels, dim=0))

    def _pad_phase_for_propagation(self, phase: torch.Tensor) -> torch.Tensor:
        """Embed an enlarged atmospheric phase screen in the full ASM domain.

        Padding a phase screen with zero phase is physically equivalent to a
        uniform complex field outside the generated turbulent support. This
        avoids introducing a zero-amplitude rectangular aperture.
        """
        p = self.asm_padding_pixels
        if p == 0:
            return phase
        return F.pad(
            phase,
            (p, p, p, p),
            mode="constant",
            value=0.0,
        )

    def _crop_propagation_center(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        s0 = self._crop_start
        s1 = self._crop_stop
        return value[..., s0:s1, s0:s1]

    def _asm_propagate(
        self,
        field: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate one gap on the persistent full ASM support.

        ``field`` already has shape ``[B, asm_resolution, asm_resolution]``.
        No padding or cropping is performed here, so diffracted energy remains
        available when the next atmospheric layer is applied.
        """
        if field.shape[-2:] != (self.asm_resolution, self.asm_resolution):
            raise RuntimeError(
                "ASM field has an inconsistent spatial shape: expected "
                f"({self.asm_resolution}, {self.asm_resolution}), got "
                f"{tuple(field.shape[-2:])}."
            )

        field_freq = torch.fft.fft2(
            torch.fft.ifftshift(field, dim=(-2, -1)),
            dim=(-2, -1),
        )
        field_filtered = torch.fft.ifftshift(
            torch.fft.fftshift(field_freq, dim=(-2, -1))
            * kernel[None, :, :],
            dim=(-2, -1),
        )
        return torch.fft.fftshift(
            torch.fft.ifft2(field_filtered, dim=(-2, -1)),
            dim=(-2, -1),
        )

    @torch.no_grad()
    def _propagate_layers_asm_delta(self, total_layers: torch.Tensor) -> torch.Tensor:
        """Propagate all atmospheric layers without intermediate cropping.

        Each phase layer is generated on the enlarged physical atmospheric
        support, zero-phase padded to the common ASM support, and applied to a
        field that remains on that support for the entire cascade. The field,
        geometric phase and phase guide are cropped only once at the telescope.
        """
        batch = total_layers.shape[0]
        expected_layer_shape = (
            self._screen_resolution,
            self._screen_resolution,
        )
        if total_layers.shape[-2:] != expected_layer_shape:
            raise RuntimeError(
                "Atmospheric layer support is inconsistent: expected "
                f"{expected_layer_shape}, got {tuple(total_layers.shape[-2:])}."
            )

        n_asm = self.asm_resolution
        field = torch.ones(
            (batch, n_asm, n_asm),
            dtype=self.complex_dtype,
            device=self.device,
        )
        phi_geo = torch.zeros(
            (batch, n_asm, n_asm),
            dtype=self.real_dtype,
            device=self.device,
        )
        phase_guide = torch.zeros_like(phi_geo)
        max_span = 0.0

        for position, layer_index in enumerate(self._propagation_order):
            idx = int(layer_index.item())

            # All layers use exactly the same physical and ASM supports.
            phi_layer = self._pad_phase_for_propagation(total_layers[:, idx])

            field.mul_(
                torch.exp(1j * phi_layer).to(self.complex_dtype)
            )
            phi_geo.add_(phi_layer)
            if self.delta_mode == "per_step":
                phase_guide.add_(phi_layer)

            if float(self._propagation_distance[position].item()) > 1e-6:
                field = self._asm_propagate(
                    field,
                    self._asm_kernels[position],
                )

            if self.delta_mode == "per_step":
                step_delta = torch.angle(
                    field
                    * torch.exp(-1j * phase_guide).to(self.complex_dtype)
                )
                phase_guide.add_(step_delta)
                span = (
                    step_delta.amax(dim=(-2, -1))
                    - step_delta.amin(dim=(-2, -1))
                )
                max_span = max(max_span, float(span.max().item()))

        # The only spatial crop in the complete propagation cascade.
        field_tel = self._crop_propagation_center(field)
        phi_geo_tel = self._crop_propagation_center(phi_geo)
        intensity = field_tel.abs().square()

        if self.delta_mode == "final":
            delta = torch.angle(
                field_tel
                * torch.exp(-1j * phi_geo_tel).to(self.complex_dtype)
            )
            phase_final = phi_geo_tel + delta
            span = (
                delta.amax(dim=(-2, -1))
                - delta.amin(dim=(-2, -1))
            )
            max_span = float(span.max().item())
        else:
            phase_final = self._crop_propagation_center(phase_guide)
            delta = phase_final - phi_geo_tel

        if max_span > self.delta_wrap_warning_threshold:
            warnings.warn(
                f"Diffractive residual span={max_span:.3f} rad; "
                "the delta itself may wrap. Increase asm_extra_pixels or "
                "the telescope sampling resolution.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._field.copy_(field_tel)
        self._intensity.copy_(intensity)
        self._diffractive_delta.copy_(delta)
        return phase_final

    def _wind_components(
        self,
        speed: torch.Tensor,
        direction_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.deg2rad(direction_deg)

        if self.direction_convention == "cartesian":
            # Previous notebook convention when oopao_convention=False.
            vx = speed * torch.cos(theta)
            vy = speed * torch.sin(theta)
        else:
            # Previous notebook convention when oopao_convention=True.
            vx = speed * torch.sin(theta)
            vy = speed * torch.cos(theta)

        return vx, vy

    def _von_karman_psd(
        self,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        fm = 5.92 / (self.l0 * 2.0 * math.pi)
        f0 = 1.0 / self.L0

        return (
            0.023
            * self.r0 ** (-5.0 / 3.0)
            * torch.exp(-(frequency / fm).square())
            / (frequency.square() + f0 * f0).pow(11.0 / 6.0)
        )

    def _precompute_high_frequency_statistics(self) -> None:
        """
        Precompute both frequency arrangements required by the algorithm.

        Centered frequencies:
            Used to reproduce the previous initial phase generation.

        Native torch.fft frequencies:
            Used to evolve the real phase via the Fourier-shift theorem.
        """
        n = self._screen_resolution
        delta_f = 1.0 / (n * self.pixel_size)

        centered_frequency = (
            torch.arange(
                n,
                dtype=self.real_dtype,
                device=self._phase.device
                if hasattr(self, "phase")
                else self.wind_speed.device,
            )
            - n / 2.0
        ) * delta_f

        fy_centered, fx_centered = torch.meshgrid(
            centered_frequency,
            centered_frequency,
            indexing="ij",
        )

        radial_frequency = torch.sqrt(
            fx_centered.square() + fy_centered.square()
        )

        psd = self._von_karman_psd(radial_frequency)
        psd[n // 2, n // 2] = 0.0

        self.register_buffer(
            "_sqrt_psd_hi_centered",
            torch.sqrt(psd) * delta_f,
        )

        fft_frequency = torch.fft.fftfreq(
            n,
            d=self.pixel_size,
            dtype=self.real_dtype,
            device=self.wind_speed.device,
        )
        fy_fft, fx_fft = torch.meshgrid(
            fft_frequency,
            fft_frequency,
            indexing="ij",
        )

        self.register_buffer("_fx_hi", fx_fft)
        self.register_buffer("_fy_hi", fy_fft)

    def _precompute_subharmonic_statistics(self) -> None:
        """
        Precompute Schmidt-style subharmonic modes.

        ``subharmonic_mode="full"`` reproduces AOtools
        ``ft_sh_phase_screen`` used by the benchmark: all nine positions of
        the 3x3 grid are traversed and the zero-frequency coefficient is zero,
        leaving eight effective complex modes per level.

        ``subharmonic_mode="oopao"`` is retained only as a compatibility mode
        for the older truncated 2x2 loop (three effective modes per level).
        """
        n = self._screen_resolution
        domain_size = n * self.pixel_size

        coordinates = (
            torch.arange(
                n,
                dtype=self.real_dtype,
                device=self.wind_speed.device,
            )
            - n / 2.0
        ) * self.pixel_size

        y, x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )

        fx_modes: list[torch.Tensor] = []
        fy_modes: list[torch.Tensor] = []
        delta_f_modes: list[torch.Tensor] = []
        spatial_bases: list[torch.Tensor] = []

        for level in range(1, self.n_subharmonic_levels + 1):
            delta_f = 1.0 / ((3**level) * domain_size)

            values = torch.tensor(
                [-delta_f, 0.0, delta_f],
                dtype=self.real_dtype,
                device=self.wind_speed.device,
            )
            fy_grid, fx_grid = torch.meshgrid(
                values,
                values,
                indexing="ij",
            )

            if self.subharmonic_mode == "oopao":
                # Exact equivalent of:
                #   for i in range(0, 2):
                #       for j in range(0, 2):
                # with PSD_phi[1, 1] = 0.
                index_y = torch.tensor(
                    [0, 0, 1],
                    dtype=torch.long,
                    device=self.wind_speed.device,
                )
                index_x = torch.tensor(
                    [0, 1, 0],
                    dtype=torch.long,
                    device=self.wind_speed.device,
                )
            else:
                # Exact AOtools 3x3 loop: all frequencies except the
                # zero-frequency center, whose PSD coefficient is zero.
                index_y, index_x = torch.meshgrid(
                    torch.arange(3, device=self.wind_speed.device),
                    torch.arange(3, device=self.wind_speed.device),
                    indexing="ij",
                )
                keep = ~((index_y == 1) & (index_x == 1))
                index_y = index_y[keep]
                index_x = index_x[keep]

            fx_level = fx_grid[index_y, index_x]
            fy_level = fy_grid[index_y, index_x]

            spatial_phase = 2.0 * math.pi * (
                fx_level[:, None, None] * x[None, :, :]
                + fy_level[:, None, None] * y[None, :, :]
            )

            spatial_basis = torch.polar(
                torch.ones_like(spatial_phase),
                spatial_phase,
            ).to(self.complex_dtype)

            fx_modes.append(fx_level)
            fy_modes.append(fy_level)
            delta_f_modes.append(
                torch.full_like(fx_level, delta_f)
            )
            spatial_bases.append(spatial_basis)

        fx_sh = torch.cat(fx_modes, dim=0)
        fy_sh = torch.cat(fy_modes, dim=0)
        delta_f_sh = torch.cat(delta_f_modes, dim=0)
        basis_sh = torch.cat(spatial_bases, dim=0)

        radial_frequency = torch.sqrt(
            fx_sh.square() + fy_sh.square()
        )
        psd_sh = self._von_karman_psd(radial_frequency)

        self.n_subharmonics = int(fx_sh.numel())

        self.register_buffer("_fx_sh", fx_sh)
        self.register_buffer("_fy_sh", fy_sh)
        self.register_buffer("_basis_sh", basis_sh)
        # Mean of each analytic basis over the finite screen. AOtools removes
        # the mean of the initial low-frequency map; using this precomputed
        # vector reproduces that operation without building an extra map.
        self.register_buffer(
            "_basis_sh_mean",
            basis_sh.mean(dim=(-2, -1)),
        )
        self.register_buffer(
            "_sqrt_psd_sh",
            torch.sqrt(psd_sh) * delta_f_sh,
        )

    def _precompute_temporal_factors(self) -> None:
        """
        Precompute one-frame frozen-flow rotations for each layer.

        Both high-frequency and subharmonic states use the same vector:
            (vx[layer], vy[layer]).
        """
        hi_dot_velocity = (
            self._fx_hi[None, :, :] * self.vx[:, None, None]
            + self._fy_hi[None, :, :] * self.vy[:, None, None]
        )
        hi_angle_per_step = (
            -2.0 * math.pi * hi_dot_velocity * self.dt
        )
        hi_step = torch.polar(
            torch.ones_like(hi_angle_per_step),
            hi_angle_per_step,
        ).to(self.complex_dtype)

        sh_dot_velocity = (
            self._fx_sh[None, :] * self.vx[:, None]
            + self._fy_sh[None, :] * self.vy[:, None]
        )
        sh_angle_per_step = (
            -2.0 * math.pi * sh_dot_velocity * self.dt
        )
        sh_step = torch.polar(
            torch.ones_like(sh_angle_per_step),
            sh_angle_per_step,
        ).to(self.complex_dtype)

        self.register_buffer("_hi_step", hi_step)
        self.register_buffer("_sh_step", sh_step)

    def _complex_randn(
        self,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        real = torch.randn(
            shape,
            dtype=self.real_dtype,
            device=self.wind_speed.device,
            generator=self.generator,
        )
        imag = torch.randn(
            shape,
            dtype=self.real_dtype,
            device=self.wind_speed.device,
            generator=self.generator,
        )
        return torch.complex(real, imag)

    def _require_generated(self) -> None:
        if (
            self._hi_spectrum_0 is None
            or self._hi_spectrum_t is None
            or self._sh_coeff_0 is None
            or self._sh_coeff_t is None
            or self._sh_piston_0 is None
            or (
                self.frozen_flow_mode == "periodic_screen"
                and (self._sh_spectrum_0 is None or self._sh_spectrum_t is None)
            )
        ):
            raise RuntimeError(
                "No atmosphere has been generated. Call gen() first."
            )

    @torch.no_grad()
    def gen(
        self,
        *,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate B independent multilayer atmospheric realizations at t=0.

        Returns
        -------
        Tensor [B, N, N]
            Sum of all atmospheric layers.
        """
        if seed is not None:
            self.seed = int(seed)
            self.generator.manual_seed(self.seed)

        # A random r0 range is sampled once per newly generated batch. These
        # per-realization values stay unchanged throughout update()/at()/rst().
        self._sample_r0_batch()

        B = self.batch_size
        L = self.n_layers
        N = self._screen_resolution
        K = self.n_subharmonics

        # -------------------------------------------------------------
        # 1. High-frequency component
        # -------------------------------------------------------------
        # Equivalent to:
        # cn = (normal + 1j*normal) * sqrt(PSD_phi) * delta_f
        centered_coefficients = (
            self._complex_randn((B, L, N, N))
            * self._sqrt_psd_hi_centered[None, None, :, :]
        )

        # Exact AOtools convention:
        # ift2(cn, 1) = ifftshift(ifft2(fftshift(cn))) * N**2
        # torch.fft.ifft2 uses the 1/N**2 normalization by default, so the
        # explicit multiplication restores the AOtools amplitude.
        phase_hi_initial = torch.fft.ifftshift(
            torch.fft.ifft2(
                torch.fft.fftshift(
                    centered_coefficients,
                    dim=(-2, -1),
                ),
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        ).real * float(N * N)

        # Layer weighting plus one r0 per batch realization. Since
        # PSD_phi ∝ r0^(-5/3), phase amplitude scales as r0^(-5/6).
        r0_scale = (self.r0 / self.r0_batch).pow(5.0 / 6.0)
        phase_hi_initial = (
            phase_hi_initial
            * self.layer_amplitude[None, :, None, None]
            * r0_scale[:, None, None, None]
        )

        # Save the real-screen FFT in native unshifted torch ordering.
        self._hi_spectrum_0 = torch.fft.fft2(
            phase_hi_initial,
            dim=(-2, -1),
        )
        self._hi_spectrum_t = self._hi_spectrum_0.clone()

        # -------------------------------------------------------------
        # 2. Subharmonic component
        # -------------------------------------------------------------
        sh_coefficients = (
            self._complex_randn((B, L, K))
            * self._sqrt_psd_sh[None, None, :]
        )

        sh_coefficients = (
            sh_coefficients
            * self.layer_amplitude[None, :, None]
            * r0_scale[:, None, None]
        )

        self._sh_coeff_0 = sh_coefficients
        self._sh_coeff_t = sh_coefficients.clone()

        # AOtools performs:
        #   phs_lo = phs_lo.real - phs_lo.real.mean()
        # at generation time. Store that initial mean as a fixed piston.
        # Subtracting the same constant at every frame preserves an exact
        # frozen-flow translation, unlike re-centering each frame.
        self._sh_piston_0 = torch.einsum(
            "blk,k->bl",
            self._sh_coeff_0,
            self._basis_sh_mean,
        ).real

        if self.frozen_flow_mode == "periodic_screen":
            # Preserve the AOtools-equivalent t=0 realization exactly, then
            # periodize that complete low-frequency map on the same internal
            # grid as the high-frequency component. Both components can now
            # be translated with one identical Fourier-shift operator.
            phase_sh_initial = torch.einsum(
                "blk,knm->blnm",
                self._sh_coeff_0,
                self._basis_sh,
            ).real
            phase_sh_initial = (
                phase_sh_initial - self._sh_piston_0[:, :, None, None]
            )
            self._sh_spectrum_0 = torch.fft.fft2(
                phase_sh_initial,
                dim=(-2, -1),
            )
            self._sh_spectrum_t = self._sh_spectrum_0.clone()
        else:
            self._sh_spectrum_0 = None
            self._sh_spectrum_t = None

        self.frame_index = 0
        self.time_s = 0.0

        return self._reconstruct_phase()

    @torch.no_grad()
    def _reconstruct_phase(self) -> torch.Tensor:
        """Reconstruct the current frozen-flow state.

        Important invariants
        --------------------
        - The subharmonic map is never re-centered frame by frame.
          Re-centering would add a time-dependent piston and make a rigid
          translation look like a deformation.
        - Optional piston removal is applied only to the final telescope-plane
          output, never to the internal layer states.
        """
        self._require_generated()

        # Periodic high-frequency component for all B x L layers.
        phase_hi = torch.fft.ifft2(
            self._hi_spectrum_t,
            dim=(-2, -1),
        ).real

        if self.frozen_flow_mode == "periodic_screen":
            # The complete initial SH map is shifted on the same periodic grid
            # as phase_hi. This guarantees a single rigid finite-screen motion,
            # constant full-screen statistics and an exact screen-period return.
            phase_sh = torch.fft.ifft2(
                self._sh_spectrum_t,
                dim=(-2, -1),
            ).real
        else:
            # Analytic low-frequency component. Each coefficient has already
            # been rotated according to the same frozen-flow wind vector.
            phase_sh = torch.einsum(
                "blk,knm->blnm",
                self._sh_coeff_t,
                self._basis_sh,
            ).real

            # AOtools subtracts the low-frequency mean only when the realization
            # is generated. Keep that piston fixed to preserve analytic advection.
            phase_sh = phase_sh - self._sh_piston_0[:, :, None, None]

        total_layers_reference = phase_hi + phase_sh

        # The generated phase corresponds to r0 at the reference wavelength.
        phase_scale = self.r0_reference_wavelength / self.wavelength
        total_layers = total_layers_reference * phase_scale

        geometric_phase_full = total_layers.sum(dim=1)
        geometric_phase = self._crop_screen_center(geometric_phase_full)

        if self.propagation_mode == "asm_delta":
            output_phase = self._propagate_layers_asm_delta(total_layers)
        else:
            output_phase = geometric_phase
            self._intensity.fill_(1.0)
            self._diffractive_delta.zero_()
            self._field.copy_(
                torch.polar(
                    torch.ones_like(geometric_phase),
                    geometric_phase,
                ).to(self.complex_dtype)
            )

        # Piston is physically irrelevant to the WFS/PSF. If requested, remove
        # it once, from the final output only. Internal layer states remain rigid.
        if self.remove_piston:
            output_phase = output_phase - output_phase.mean(
                dim=(-2, -1),
                keepdim=True,
            )

        self._phase.copy_(output_phase)

        if self.store_components:
            self.phase_hi_layers = self._crop_screen_center(
                phase_hi * phase_scale
            )
            self.phase_sh_layers = self._crop_screen_center(
                phase_sh * phase_scale
            )
            self.phase_layers = self._crop_screen_center(total_layers)
        else:
            self.phase_hi_layers = None
            self.phase_sh_layers = None
            self.phase_layers = None

        return self.phase

    def _high_frequency_absolute_factor(self, frame_index: int) -> torch.Tensor:
        """Return the exact finite-screen Fourier-shift factor at one frame.

        The displacement is expressed in pixels and reduced modulo the internal
        screen size. Values numerically indistinguishable from a complete turn
        are snapped to zero, making a full loop return exactly to the t=0 state
        instead of retaining a small complex64 phase residue.
        """
        n = self._screen_resolution
        work_dtype = torch.float64
        device = self.device

        shift_x = (
            self.vx.to(work_dtype)
            * (float(frame_index) * self.dt / self.pixel_size)
        )
        shift_y = (
            self.vy.to(work_dtype)
            * (float(frame_index) * self.dt / self.pixel_size)
        )
        shift_x = torch.remainder(shift_x, float(n))
        shift_y = torch.remainder(shift_y, float(n))

        tol = 128.0 * torch.finfo(work_dtype).eps * max(1.0, float(n), abs(float(frame_index)))
        shift_x = torch.where(
            (shift_x.abs() <= tol) | ((shift_x - float(n)).abs() <= tol),
            torch.zeros_like(shift_x),
            shift_x,
        )
        shift_y = torch.where(
            (shift_y.abs() <= tol) | ((shift_y - float(n)).abs() <= tol),
            torch.zeros_like(shift_y),
            shift_y,
        )

        freq = torch.fft.fftfreq(n, d=1.0, dtype=work_dtype, device=device)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        angle = -2.0 * math.pi * (
            fx[None, :, :] * shift_x[:, None, None]
            + fy[None, :, :] * shift_y[:, None, None]
        )
        return torch.polar(torch.ones_like(angle), angle).to(self.complex_dtype)

    def _analytic_sh_absolute_factor(self, frame_index: int) -> torch.Tensor:
        """Return an absolute unit-modulus factor for analytic subharmonics.

        Computing the phase from physical time in float64 avoids the slow
        modulus drift caused by repeatedly multiplying complex64 coefficients.
        The phase is reduced modulo one cycle before evaluating the exponential.
        """
        work_dtype = torch.float64
        time_s = float(frame_index) * self.dt
        cycles = (
            self._fx_sh.to(work_dtype)[None, :]
            * self.vx.to(work_dtype)[:, None]
            + self._fy_sh.to(work_dtype)[None, :]
            * self.vy.to(work_dtype)[:, None]
        ) * time_s
        cycles = cycles - torch.round(cycles)

        # Frequencies were originally stored in real_dtype. Snap only tiny
        # float32 representation residues at theoretically complete cycles.
        snap_tol = 2.0e-6 if self.real_dtype == torch.float32 else 1.0e-12
        cycles = torch.where(
            cycles.abs() <= snap_tol,
            torch.zeros_like(cycles),
            cycles,
        )
        angle = -2.0 * math.pi * cycles
        return torch.polar(torch.ones_like(angle), angle).to(self.complex_dtype)

    def _is_high_frequency_screen_return(self, frame_index: int) -> bool:
        """Whether every layer has completed an integer finite-screen turn."""
        factor = self._high_frequency_absolute_factor(frame_index)
        ones = torch.ones((), dtype=self.complex_dtype, device=self.device)
        tol = 5.0e-6 if self.real_dtype == torch.float32 else 1.0e-12
        return bool(torch.allclose(factor, ones, rtol=0.0, atol=tol))

    def _set_absolute_temporal_state(self, frame_index: int) -> None:
        """Set all temporal states directly from the t=0 realization."""
        hi_factor = self._high_frequency_absolute_factor(frame_index)
        self._hi_spectrum_t = self._hi_spectrum_0 * hi_factor[None, :, :, :]

        sh_factor = self._analytic_sh_absolute_factor(frame_index)
        self._sh_coeff_t = self._sh_coeff_0 * sh_factor[None, :, :]

        if self.frozen_flow_mode == "periodic_screen":
            self._sh_spectrum_t = self._sh_spectrum_0 * hi_factor[None, :, :, :]

    @torch.no_grad()
    def update(
        self,
        n: int = 1,
    ) -> torch.Tensor:
        """Advance the complete batched atmosphere by ``n`` frames.

        For one-frame updates the state is advanced recursively for speed.
        Every ``temporal_reanchor_interval`` frames it is recomputed directly
        from the initial realization, preventing long-run complex64 drift.
        Multi-frame jumps are evaluated directly at the target frame.
        """
        self._require_generated()

        n_steps = int(n)
        if n_steps <= 0:
            raise ValueError("n must be positive.")

        target_frame = self.frame_index + n_steps

        # In strict finite-screen mode, evaluate every frame absolutely. This
        # prevents recursive complex64 drift and makes the motion independent
        # of whether the caller uses update(1), update(n) or at(n).
        if self.frozen_flow_mode == "periodic_screen":
            self._set_absolute_temporal_state(target_frame)
            self.frame_index = target_frame
            self.time_s = target_frame * self.dt
            return self._reconstruct_phase()

        reanchor = (
            n_steps != 1
            or self._is_high_frequency_screen_return(target_frame)
            or (
                self.temporal_reanchor_interval > 0
                and target_frame % self.temporal_reanchor_interval == 0
            )
        )

        if reanchor:
            self._set_absolute_temporal_state(target_frame)
        else:
            self._hi_spectrum_t.mul_(
                self._hi_step[None, :, :, :]
            )
            # Subharmonics are inexpensive (typically 24 modes), so evaluate
            # them absolutely every frame. Their coefficient magnitudes then
            # remain identical to t=0 instead of drifting in complex64.
            sh_factor = self._analytic_sh_absolute_factor(target_frame)
            self._sh_coeff_t = self._sh_coeff_0 * sh_factor[None, :, :]

        self.frame_index = target_frame
        self.time_s = target_frame * self.dt
        return self._reconstruct_phase()

    @torch.no_grad()
    def step(self, n: int = 1) -> torch.Tensor:
        """Deprecated compatibility alias for :meth:`update`."""
        warnings.warn(
            "Atmosphere.step() was renamed to Atmosphere.update(); "
            "step() remains temporarily available for compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.update(n)

    @torch.no_grad()
    def at(
        self,
        frame: int,
    ) -> torch.Tensor:
        """Evaluate an absolute frame directly from the initial state."""
        self._require_generated()

        frame_index = int(frame)
        if frame_index < 0:
            raise ValueError("frame cannot be negative.")

        self._set_absolute_temporal_state(frame_index)
        self.frame_index = frame_index
        self.time_s = frame_index * self.dt
        return self._reconstruct_phase()

    @torch.no_grad()
    def rst(self) -> torch.Tensor:
        """
        Return the current generated batch to frame zero.
        """
        self._require_generated()

        self._hi_spectrum_t = self._hi_spectrum_0.clone()
        self._sh_coeff_t = self._sh_coeff_0.clone()
        if self._sh_spectrum_0 is not None:
            self._sh_spectrum_t = self._sh_spectrum_0.clone()

        self.frame_index = 0
        self.time_s = 0.0

        return self._reconstruct_phase()

    @torch.no_grad()
    def new(
        self,
        *,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a new independent realization at frame zero.
        """
        return self.gen(seed=seed)

    @property
    def r0_at_wavelength(self) -> float:
        """Fried parameter at the current optical wavelength [m]."""
        return self.r0 * (
            self.wavelength / self.r0_reference_wavelength
        ) ** (6.0 / 5.0)

    def r0_at(self, wavelength: float) -> float:
        """Fried parameter at an arbitrary wavelength [m]."""
        wavelength = float(wavelength)
        if wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        return self.r0 * (
            wavelength / self.r0_reference_wavelength
        ) ** (6.0 / 5.0)

    @property
    def seeing_arcsec(self) -> float:
        """OOPAO-compatible seeing lambda/r0 at current wavelength."""
        return self.rad2arcsec * (
            self.wavelength / self.r0_at_wavelength
        )

    @property
    def seeing_arcsec_reference(self) -> float:
        """Seeing at the r0 reference wavelength [arcsec]."""
        return self.rad2arcsec * (
            self.r0_reference_wavelength / self.r0
        )

    @property
    def cn2_integral(self) -> float:
        """Integrated Cn² dh implied by r0 [m^(1/3)]."""
        k_ref = 2.0 * math.pi / self.r0_reference_wavelength
        return self.r0 ** (-5.0 / 3.0) / (0.423 * k_ref**2)

    @property
    def cn2_layer_integrals(self) -> torch.Tensor:
        """Per-layer integrated Cn² dh using the normalized profile."""
        weights = self.fractional_r0 / self.fractional_r0.sum().clamp_min(
            torch.finfo(self.real_dtype).eps
        )
        return weights * self.cn2_integral

    @property
    def cn2_mean(self) -> float:
        """Mean Cn² over max layer altitude, matching OOPAO's summary."""
        path = max(1.0, float(self.altitude.max().item()))
        return self.cn2_integral / path

    @property
    def equivalent_wind_speed(self) -> float:
        """Roddier equivalent wind speed V0 [m/s]."""
        weights = self.fractional_r0 / self.fractional_r0.sum().clamp_min(
            torch.finfo(self.real_dtype).eps
        )
        value = torch.sum(weights * self.wind_speed.pow(5.0 / 3.0))
        return float(value.pow(3.0 / 5.0).item())

    @property
    def coherence_time(self) -> float:
        """Atmospheric coherence time tau0 at reference wavelength [s]."""
        v0 = self.equivalent_wind_speed
        return math.inf if v0 == 0.0 else 0.31 * self.r0 / v0

    @property
    def effective_turbulence_altitude(self) -> float:
        """Cn²-weighted effective altitude h0 [m]."""
        weights = self.fractional_r0 / self.fractional_r0.sum().clamp_min(
            torch.finfo(self.real_dtype).eps
        )
        value = torch.sum(weights * self.altitude.pow(5.0 / 3.0))
        return float(value.pow(3.0 / 5.0).item())

    @property
    def isoplanatic_angle_arcsec(self) -> float:
        """Approximate isoplanatic angle at current wavelength [arcsec]."""
        h0 = self.effective_turbulence_altitude
        if h0 == 0.0:
            return math.inf
        theta_rad = 0.314 * self.r0_at_wavelength / h0
        return theta_rad * self.rad2arcsec

    @property
    def rytov_variance(self) -> float:
        """Plane-wave Rytov variance from the discrete Cn² profile."""
        weights = self.fractional_r0 / self.fractional_r0.sum().clamp_min(
            torch.finfo(self.real_dtype).eps
        )
        k = 2.0 * math.pi / self.wavelength
        cn2_total_current = self.r0_at_wavelength ** (-5.0 / 3.0) / (
            0.423 * k**2
        )
        weighted_height = torch.sum(
            weights * self.altitude.pow(5.0 / 6.0)
        )
        return float(2.2524 * k ** (7.0 / 6.0) * cn2_total_current * weighted_height.item())

    def atmospheric_parameters(self) -> dict[str, object]:
        """Return the main atmospheric and propagation diagnostics."""
        return {
            "r0_reference_m": self.r0,
            "r0_mode": self.r0_mode,
            "r0_range_m": (self.r0_min, self.r0_max),
            "r0_batch_m": self.r0_batch.detach().cpu(),
            "r0_reference_wavelength_m": self.r0_reference_wavelength,
            "wavelength_m": self.wavelength,
            "r0_at_wavelength_m": self.r0_at_wavelength,
            "seeing_arcsec": self.seeing_arcsec,
            "seeing_reference_arcsec": self.seeing_arcsec_reference,
            "L0_m": self.L0,
            "l0_m": self.l0,
            "cn2_integral_m_1_3": self.cn2_integral,
            "cn2_mean_m_minus_2_3": self.cn2_mean,
            "cn2_layer_integrals_m_1_3": self.cn2_layer_integrals.detach().cpu(),
            "equivalent_wind_speed_mps": self.equivalent_wind_speed,
            "tau0_s_reference": self.coherence_time,
            "effective_altitude_m": self.effective_turbulence_altitude,
            "isoplanatic_angle_arcsec": self.isoplanatic_angle_arcsec,
            "rytov_variance": self.rytov_variance,
            "direction_convention": self.direction_convention,
            "wind_speed_mps": self.wind_speed.detach().cpu(),
            "wind_direction_deg": self.wind_direction.detach().cpu(),
            "vx_mps": self.vx.detach().cpu(),
            "vy_mps": self.vy.detach().cpu(),
            "altitude_m": self.altitude.detach().cpu(),
            "fractional_r0": self.fractional_r0.detach().cpu(),
            "subharmonic_mode": self.subharmonic_mode,
            "subharmonic_levels": self.n_subharmonic_levels,
            "subharmonic_modes_per_level": (
                3 if self.subharmonic_mode == "oopao" else 8
            ),
            "frozen_flow_mode": self.frozen_flow_mode,
        }


    # ------------------------------------------------------------------
    # Compact public API and dependency-aware reconfiguration
    # ------------------------------------------------------------------
    def _drop_buffers(self, *names: str) -> None:
        """Remove cached buffers before recomputing them."""
        for name in names:
            if name in self._buffers:
                delattr(self, name)

    def _clear_state(self) -> None:
        """Invalidate the current random realization after a grid/PSD change."""
        self._hi_spectrum_0 = None
        self._hi_spectrum_t = None
        self._sh_coeff_0 = None
        self._sh_coeff_t = None
        self._sh_spectrum_0 = None
        self._sh_spectrum_t = None
        self._sh_piston_0 = None
        self.phase_layers = None
        self.phase_hi_layers = None
        self.phase_sh_layers = None
        self.frame_index = 0
        self.time_s = 0.0

    def _resize_outputs(self) -> None:
        """Resize public outputs after changing telescope resolution."""
        shape = (self.batch_size, self.resolution, self.resolution)
        self._phase = torch.zeros(shape, dtype=self.real_dtype, device=self.device)
        self._intensity = torch.ones(shape, dtype=self.real_dtype, device=self.device)
        self._field = torch.ones(shape, dtype=self.complex_dtype, device=self.device)
        self._diffractive_delta = torch.zeros(shape, dtype=self.real_dtype, device=self.device)

    def _rebuild_space(self) -> None:
        """Recompute spatial PSD and subharmonic constants only."""
        self._drop_buffers(
            "_sqrt_psd_hi_centered", "_fx_hi", "_fy_hi",
            "_fx_sh", "_fy_sh", "_basis_sh", "_basis_sh_mean",
            "_basis_sh_prop", "_sqrt_psd_sh",
        )
        self._precompute_high_frequency_statistics()
        self._precompute_subharmonic_statistics()

    def _rebuild_time(self) -> None:
        """Recompute frozen-flow factors after wind or frame-rate changes."""
        self._drop_buffers("_hi_step", "_sh_step")
        self._precompute_temporal_factors()

    def _rebuild_prop(self) -> None:
        """Recompute layer distances, padding geometry and ASM kernels."""
        self._drop_buffers("_propagation_order", "_propagation_distance", "_asm_kernels", "_basis_sh_prop")
        self._precompute_propagation_geometry()
        self._precompute_asm_kernels()

    def _regen_if_ready(self) -> torch.Tensor:
        """Regenerate only when a realization already exists."""
        if self._hi_spectrum_t is None:
            return self.phase
        return self.gen()

    def _refresh_psd_amplitudes(self) -> None:
        """Refresh only the r0/L0/l0-dependent PSD amplitudes.

        The frequency grids, spatial bases, ASM kernels and temporal factors
        are unchanged. This is substantially cheaper than rebuilding the whole
        atmosphere whenever only ``r0`` changes between training batches.
        """
        n = self._screen_resolution
        delta_f = 1.0 / (n * self.pixel_size)

        centered_frequency = (
            torch.arange(
                n,
                dtype=self.real_dtype,
                device=self.device,
            )
            - n / 2.0
        ) * delta_f

        fy_centered, fx_centered = torch.meshgrid(
            centered_frequency,
            centered_frequency,
            indexing="ij",
        )
        radial_frequency = torch.sqrt(
            fx_centered.square() + fy_centered.square()
        )

        psd_hi = self._von_karman_psd(radial_frequency)
        psd_hi[n // 2, n // 2] = 0.0
        self._sqrt_psd_hi_centered.copy_(
            torch.sqrt(psd_hi) * delta_f
        )

        radial_frequency_sh = torch.sqrt(
            self._fx_sh.square() + self._fy_sh.square()
        )
        psd_sh = self._von_karman_psd(radial_frequency_sh)

        # Every subharmonic mode lies on a 3x3 grid. Therefore the grid
        # spacing of a mode is max(|fx|, |fy|).
        delta_f_sh = torch.maximum(
            self._fx_sh.abs(),
            self._fy_sh.abs(),
        )
        self._sqrt_psd_sh.copy_(
            torch.sqrt(psd_sh) * delta_f_sh
        )

    @torch.no_grad()
    def configure_profile(
        self,
        *,
        r0: float | Sequence[float],
        wind_speed: Iterable[float],
        wind_direction: Iterable[float],
        altitude: Iterable[float],
        fractional_r0: Optional[Iterable[float]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Configure one profile and generate its frame-zero batch exactly once.

        This method is intended for on-the-fly training. It updates all
        batch-shared physical parameters together, preserves their layer
        correspondence while sorting from highest to lowest altitude, rebuilds
        only the affected cached quantities, invalidates the previous
        realization, and finally calls :meth:`gen` once.

        The returned tensor is the newly generated frame-zero phase. The
        matching amplitude and complex field are available immediately through
        ``intensity`` and ``field``. The caller must use this returned frame
        directly and call only :meth:`update` for later temporal frames.

        The number of layers is fixed when the object is constructed.
        """
        (
            r0_mode,
            r0_min,
            r0_max,
            r0_reference,
        ) = self._parse_r0_spec(r0)

        speed = torch.as_tensor(
            list(wind_speed),
            dtype=self.real_dtype,
            device=self.device,
        )
        direction = torch.as_tensor(
            list(wind_direction),
            dtype=self.real_dtype,
            device=self.device,
        )
        alt = torch.as_tensor(
            list(altitude),
            dtype=self.real_dtype,
            device=self.device,
        )

        if fractional_r0 is None:
            fraction = self.fractional_r0.clone()
        else:
            fraction = torch.as_tensor(
                list(fractional_r0),
                dtype=self.real_dtype,
                device=self.device,
            )

        if not (
            speed.numel()
            == direction.numel()
            == alt.numel()
            == fraction.numel()
            == self.n_layers
        ):
            raise ValueError(
                "r0 profile arrays must contain exactly one value per layer."
            )
        if torch.any(speed < 0):
            raise ValueError("wind_speed cannot contain negative values.")
        if torch.any(alt < 0):
            raise ValueError("altitude cannot contain negative values.")
        if torch.any(fraction < 0) or fraction.sum() <= 0:
            raise ValueError(
                "fractional_r0 must be non-negative with a positive sum."
            )

        if self.normalize_fractional_r0:
            fraction = fraction / fraction.sum()

        direction = torch.remainder(direction, 360.0)

        # Keep each fraction, wind vector and altitude associated with the same
        # physical layer while enforcing the class high-to-low invariant.
        order = torch.argsort(alt, descending=True)
        speed = speed[order]
        direction = direction[order]
        alt = alt[order]
        fraction = fraction[order]

        r0_changed = not math.isclose(
            self.r0,
            r0_reference,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        wind_changed = not (
            torch.allclose(speed, self.wind_speed)
            and torch.allclose(direction, self.wind_direction)
        )
        altitude_changed = not torch.allclose(alt, self.altitude)
        fraction_changed = not torch.allclose(
            fraction,
            self.fractional_r0,
        )

        self.r0_mode = r0_mode
        self.r0_min = r0_min
        self.r0_max = r0_max
        self.r0 = r0_reference
        self.wind_speed.copy_(speed)
        self.wind_direction.copy_(direction)
        self.altitude.copy_(alt)
        self.fractional_r0.copy_(fraction)
        self.layer_amplitude.copy_(torch.sqrt(fraction))

        vx, vy = self._wind_components(
            self.wind_speed,
            self.wind_direction,
        )
        self.vx.copy_(vx)
        self.vy.copy_(vy)

        old_screen_resolution = self._screen_resolution
        self._update_asm_extra_pixels()
        self._screen_resolution = self._required_screen_resolution()
        screen_changed = (
            self._screen_resolution != old_screen_resolution
        )

        if screen_changed:
            self._rebuild_space()
            self._rebuild_time()
            self._rebuild_prop()
        else:
            if r0_changed:
                self._refresh_psd_amplitudes()
            if wind_changed:
                self._rebuild_time()
            if altitude_changed:
                self._rebuild_prop()

        # Every online batch must be an independent realization. Clear any
        # previous temporal state and generate frame zero exactly once here.
        # The main training loop must use this return value directly instead of
        # calling gen() again.
        self._clear_state()
        return self.gen(seed=seed)

    @torch.no_grad()
    def set_r0(
        self,
        value: float | Sequence[float],
    ) -> torch.Tensor:
        """Set fixed-r0 or random-r0-range mode and generate a new batch."""
        mode, low, high, reference = self._parse_r0_spec(value)
        reference_changed = not math.isclose(
            self.r0, reference, rel_tol=1e-12, abs_tol=0.0
        )

        self.r0_mode = mode
        self.r0_min = low
        self.r0_max = high
        self.r0 = reference

        if reference_changed:
            self._refresh_psd_amplitudes()

        self._clear_state()
        return self.gen()

    @torch.no_grad()
    def set_L0(self, value: float) -> torch.Tensor:
        """Set outer scale, rebuild the PSD and regenerate the atmosphere."""
        value = float(value)
        if value <= 0:
            raise ValueError("L0 must be positive.")
        self.L0 = value
        self._rebuild_space()
        self._clear_state()
        return self.gen()

    @torch.no_grad()
    def set_l0(self, value: float) -> torch.Tensor:
        """Set inner scale, rebuild the PSD and regenerate the atmosphere."""
        value = float(value)
        if value <= 0:
            raise ValueError("l0 must be positive.")
        self.l0 = value
        self._rebuild_space()
        self._clear_state()
        return self.gen()

    @torch.no_grad()
    def set_wvl(self, value: float) -> torch.Tensor:
        """Set optical wavelength and rebuild only wavelength-dependent ASM kernels."""
        value = float(value)
        if value <= 0:
            raise ValueError("wavelength must be positive.")
        self.wavelength = value
        old_screen = self._screen_resolution
        self._update_asm_extra_pixels()
        self._screen_resolution = self._required_screen_resolution()
        if self._screen_resolution != old_screen:
            self._rebuild_space()
            self._rebuild_time()
            self._rebuild_prop()
            self._clear_state()
            return self.gen()
        self._rebuild_prop()
        if self._hi_spectrum_t is not None:
            return self._reconstruct_phase()
        return self.phase

    @torch.no_grad()
    def set_fps(self, value: float) -> torch.Tensor:
        """Set temporal sampling and keep the current frame consistent.

        The frame index is preserved. After rebuilding the temporal factors,
        the spectral state is recomputed absolutely at that frame under the
        new sampling interval.
        """
        value = float(value)
        if value <= 0:
            raise ValueError("frame_rate must be positive.")

        self.frame_rate = value
        self.dt = 1.0 / value
        self._rebuild_time()

        if self._hi_spectrum_0 is not None:
            self._set_absolute_temporal_state(self.frame_index)
            self.time_s = self.frame_index * self.dt
            return self._reconstruct_phase()

        return self.phase

    @torch.no_grad()
    def set_wind(
        self,
        speed: Optional[Iterable[float]] = None,
        direction: Optional[Iterable[float]] = None,
    ) -> torch.Tensor:
        """Update layer wind speed/direction and keep time consistent.

        After rebuilding the frozen-flow factors, the current frame is
        evaluated absolutely from the original random realization. This avoids
        mixing the old wind trajectory with the new one.
        """
        new_speed = self.wind_speed if speed is None else torch.as_tensor(
            list(speed), dtype=self.real_dtype, device=self.device
        )
        new_direction = self.wind_direction if direction is None else torch.as_tensor(
            list(direction), dtype=self.real_dtype, device=self.device
        )

        if (
            new_speed.numel() != self.n_layers
            or new_direction.numel() != self.n_layers
        ):
            raise ValueError(
                "speed and direction must contain one value per layer."
            )
        if torch.any(new_speed < 0):
            raise ValueError("wind speed cannot be negative.")

        self.wind_speed.copy_(new_speed)
        self.wind_direction.copy_(torch.remainder(new_direction, 360.0))

        vx, vy = self._wind_components(
            self.wind_speed,
            self.wind_direction,
        )
        self.vx.copy_(vx)
        self.vy.copy_(vy)
        self._rebuild_time()

        if self._hi_spectrum_0 is not None:
            self._set_absolute_temporal_state(self.frame_index)
            self.time_s = self.frame_index * self.dt
            return self._reconstruct_phase()

        return self.phase

    @torch.no_grad()
    def set_temporal_reanchor_interval(self, value: int) -> torch.Tensor:
        """Set the numerical re-anchoring interval used by ``update``.

        ``0`` disables re-anchoring. Positive values re-evaluate the temporal
        state exactly from frame zero whenever the current frame is a multiple
        of the interval.
        """
        value = int(value)
        if value < 0:
            raise ValueError(
                "temporal_reanchor_interval must be >= 0."
            )
        self.temporal_reanchor_interval = value
        return self.phase

    @torch.no_grad()
    def set_alt(self, values: Iterable[float]) -> torch.Tensor:
        """Set altitudes, reorder complete layers top-to-bottom and rebuild ASM."""
        alt = torch.as_tensor(list(values), dtype=self.real_dtype, device=self.device)
        if alt.numel() != self.n_layers:
            raise ValueError("altitude must contain one value per layer.")
        if torch.any(alt < 0):
            raise ValueError("altitude cannot be negative.")
        order = torch.argsort(alt, descending=True)
        self.altitude.copy_(alt[order])
        for name in ("wind_speed", "wind_direction", "fractional_r0", "layer_amplitude", "vx", "vy"):
            value = getattr(self, name).clone()[order]
            getattr(self, name).copy_(value)
        for name in (
            "_hi_spectrum_0", "_hi_spectrum_t",
            "_sh_coeff_0", "_sh_coeff_t",
            "_sh_spectrum_0", "_sh_spectrum_t", "_sh_piston_0",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[:, order].clone())
        self._rebuild_time()
        old_screen = self._screen_resolution
        self._update_asm_extra_pixels()
        self._screen_resolution = self._required_screen_resolution()
        if self._screen_resolution != old_screen:
            self._rebuild_space()
            self._rebuild_time()
            self._rebuild_prop()
            self._clear_state()
            return self.gen()
        self._rebuild_prop()
        if self._hi_spectrum_t is not None:
            return self._reconstruct_phase()
        return self.phase

    @torch.no_grad()
    def set_frac(self, values: Iterable[float]) -> torch.Tensor:
        """Set the Cn² fractions, normalize their sum to one and regenerate."""
        frac = torch.as_tensor(list(values), dtype=self.real_dtype, device=self.device)
        if frac.numel() != self.n_layers:
            raise ValueError("fractional_r0 must contain one value per layer.")
        if torch.any(frac < 0) or frac.sum() <= 0:
            raise ValueError("fractional_r0 must be non-negative with positive sum.")
        if not torch.isclose(frac.sum(), torch.ones((), device=self.device, dtype=self.real_dtype), rtol=1e-5, atol=1e-7):
            warnings.warn("fractional_r0 was normalized automatically.", RuntimeWarning, stacklevel=2)
        frac = frac / frac.sum()
        self.fractional_r0.copy_(frac)
        self.layer_amplitude.copy_(torch.sqrt(frac))
        self._clear_state()
        return self.gen()

    @torch.no_grad()
    def set_tel(
        self,
        *,
        res: Optional[int] = None,
        D: Optional[float] = None,
        wvl: Optional[float] = None,
    ) -> torch.Tensor:
        """Update telescope resolution, diameter and/or wavelength efficiently."""
        grid_changed = False
        if res is not None:
            res = int(res)
            if res <= 0:
                raise ValueError("resolution must be positive.")
            grid_changed |= res != self.resolution
            self.resolution = res
        if D is not None:
            D = float(D)
            if D <= 0:
                raise ValueError("telescope diameter must be positive.")
            grid_changed |= D != self.telescope_diameter
            self.telescope_diameter = D
        if wvl is not None:
            wvl = float(wvl)
            if wvl <= 0:
                raise ValueError("wavelength must be positive.")
            self.wavelength = wvl
        if grid_changed:
            self.pixel_size = self.telescope_diameter / self.resolution
            self._update_asm_extra_pixels()
            self._screen_resolution = self._required_screen_resolution()
            self._rebuild_space()
            self._rebuild_time()
            self._rebuild_prop()
            self._resize_outputs()
            self._clear_state()
            return self.gen()
        if wvl is not None:
            old_screen = self._screen_resolution
            self._update_asm_extra_pixels()
            self._screen_resolution = self._required_screen_resolution()
            if self._screen_resolution != old_screen:
                self._rebuild_space()
                self._rebuild_time()
                self._rebuild_prop()
                self._clear_state()
                return self.gen()
            self._rebuild_prop()
            if self._hi_spectrum_t is not None:
                return self._reconstruct_phase()
        return self.phase

    @torch.no_grad()
    def set_prop(
        self,
        mode: Optional[PropagationMode] = None,
        *,
        delta: Optional[DeltaMode] = None,
        extra: int | Literal["auto"] | None = None,
        padding: Optional[float] = None,
    ) -> torch.Tensor:
        """Configure propagation and rebuild only its dependent constants.

        ``delta='final'`` is the OOPAO method: extract one residual at the
        telescope. ``delta='per_step'`` extracts a residual after every gap.
        ``extra`` is the number of genuinely generated atmospheric pixels on
        each side when ASM is active; the propagated result is center-cropped.
        ``padding`` is the temporary zero-padding factor used by every ASM FFT.
        """
        old_screen = self._screen_resolution
        if mode is not None:
            if mode not in ("geometric", "asm_delta"):
                raise ValueError("mode must be 'geometric' or 'asm_delta'.")
            self.propagation_mode = mode
        if delta is not None:
            if delta not in ("final", "per_step"):
                raise ValueError("delta must be 'final' or 'per_step'.")
            self.delta_mode = delta
        if extra is not None:
            if extra == "auto":
                self._asm_extra_pixels_request = "auto"
                self.asm_extra_pixels_mode = "auto"
            else:
                extra = int(extra)
                if extra < 0:
                    raise ValueError("extra must be >= 0 or 'auto'.")
                self._asm_extra_pixels_request = extra
                self.asm_extra_pixels_mode = "manual"
                self.asm_extra_pixels = extra
        if padding is not None:
            padding = float(padding)
            if padding < 1.0:
                raise ValueError("padding must be >= 1.0.")
            self.asm_padding_factor = padding

        self._update_asm_extra_pixels()
        self._screen_resolution = self._required_screen_resolution()
        screen_changed = self._screen_resolution != old_screen
        if screen_changed:
            self._rebuild_space()
            self._rebuild_time()
        self._rebuild_prop()
        if screen_changed:
            self._clear_state()
            return self.gen()
        if self._hi_spectrum_t is not None:
            return self._reconstruct_phase()
        return self.phase

    def info(self) -> dict[str, object]:
        """Return atmospheric, telescope and propagation diagnostics."""
        values = self.atmospheric_parameters()
        values.update({
            "resolution": self.resolution,
            "telescope_diameter_m": self.telescope_diameter,
            "pixel_size_m": self.pixel_size,
            "frame_rate_hz": self.frame_rate,
            "propagation_mode": self.propagation_mode,
            "internal_screen_resolution": self._screen_resolution,
            "asm_extra_pixels_mode": self.asm_extra_pixels_mode,
            "asm_extra_pixels_per_side": self.asm_extra_pixels,
            "asm_min_physical_margin_m": self.asm_min_physical_margin,
            "asm_physical_margin_per_side_m": self.asm_extra_pixels * self.pixel_size,
            "public_output_shape": (1, self.batch_size, self.resolution, self.resolution),
            "asm_padding_factor": self.asm_padding_factor,
            "asm_padding_pixels_per_side": self.asm_padding_pixels,
            "asm_resolution": self.asm_resolution,
            "delta_mode": self.delta_mode,
            "temporal_reanchor_interval": self.temporal_reanchor_interval,
            "propagation_resolution": self.propagation_resolution,
            "propagation_distances_m": self._propagation_distance.detach().cpu(),
        })
        return values

    def extra_repr(self) -> str:
        return (
            f"B={self.batch_size}, L={self.n_layers}, "
            f"N={self.resolution}, D={self.telescope_diameter:g} m, "
            f"fps={self.frame_rate:g}, r0_mode={self.r0_mode}, "
            f"r0_range=[{self.r0_min:g},{self.r0_max:g}] m, "
            f"L0={self.L0:g} m, wavelength={self.wavelength:g} m, "
            f"seeing={self.seeing_arcsec:.3f} arcsec, dir={self.direction_convention}, "
            f"subharmonics={self.subharmonic_mode}, "
            f"frozen_flow={self.frozen_flow_mode}, "
            f"propagation={self.propagation_mode}, "
            f"asm_extra={self.asm_extra_pixels}px, "
            f"Nprop={self.propagation_resolution}, "
            f"reanchor={self.temporal_reanchor_interval}, "
            f"device={self.device}, dtype={self.real_dtype}"
        )

