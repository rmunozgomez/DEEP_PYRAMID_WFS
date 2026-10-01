import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PYRAMID.functions_pyr_torch import *

def pad2size(
    x: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Zero-padding centrado (o recorte centrado) para tensores (B, M, H, W).

    Parámetros
    ----------
    x : torch.Tensor
        Tensor de entrada con shape (B, M, H, W), real o complejo.
    out_hw : (H_out, W_out)
        Tamaño espacial objetivo.

    Retorna
    -------
    y : torch.Tensor
        Tensor con shape (B, M, H_out, W_out).
    """
    if x.ndim != 4:
        raise ValueError("El tensor debe tener shape (B, M, H, W).")

    B, M, H, W = x.shape
    H_out, W_out = out_hw

    if H_out <= 0 or W_out <= 0:
        raise ValueError("out_hw debe ser positivo.")

    # ---------- RECORTE (si out < in) ----------
    y = x
    if H_out < H:
        dh = (H - H_out) // 2
        y = y[:, :, dh:dh + H_out, :]
        H = H_out

    if W_out < W:
        dw = (W - W_out) // 2
        y = y[:, :, :, dw:dw + W_out]
        W = W_out

    # ---------- PADDING (si out > in) ----------
    pad_h = H_out - H
    pad_w = W_out - W

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad espera (left, right, top, bottom)
        y = F.pad(
            y,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0
        )

    return y

def circular_pupil(
    n: int,
    *,
    device=None,
    dtype=torch.float32,
    soft_edge_px: float = 0.0,
) -> torch.Tensor:
    """
    Generate a centered circular pupil.

    Parameters
    ----------
    n:
        Spatial resolution.

    device:
        Torch device.

    dtype:
        Output floating-point dtype.

    soft_edge_px:
        Width in pixels of the cosine edge transition.

        0:
            hard binary pupil.

        > 0:
            cosine transition outside the nominal radius.

    Returns
    -------
    pupil:
        Tensor with shape [1, 1, n, n].

        The shape is always identical regardless of whether
        a hard or soft edge is used.
    """

    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    if soft_edge_px < 0:
        raise ValueError(
            "soft_edge_px cannot be negative."
        )

    # ============================================================
    # Working precision
    # ============================================================

    if dtype == torch.float64:
        work_dtype = torch.float64
    else:
        work_dtype = torch.float32

    # ============================================================
    # Geometric center
    # ============================================================

    center = (
        n - 1
    ) / 2.0

    radius_px = center

    coordinates = torch.arange(
        n,
        device=device,
        dtype=work_dtype,
    )

    Y, X = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    radius = torch.sqrt(
        (X - center) ** 2
        + (Y - center) ** 2
    )

    # ============================================================
    # Hard edge
    # ============================================================

    if soft_edge_px == 0:

        pupil = (
            radius <= radius_px
        ).to(
            dtype=dtype
        )

    # ============================================================
    # Soft edge
    # ============================================================

    else:

        width = torch.as_tensor(
            float(soft_edge_px),
            device=device,
            dtype=work_dtype,
        )

        # Normalized transition:
        #
        # r <= radius       -> 0
        # r >= radius + w   -> 1
        transition = (
            (radius - radius_px)
            / width
        ).clamp(
            0.0,
            1.0,
        )

        pupil = (
            0.5
            * (
                1.0
                + torch.cos(
                    torch.pi
                    * transition
                )
            )
        )

        # The cosine expression gives 1 before the transition
        # and 0 after it because transition is clamped [0,1].
        pupil = pupil.to(
            dtype=dtype
        )

    # ============================================================
    # Canonical WFS pupil shape
    # ============================================================

    return pupil[
        None,
        None,
        :,
        :,
    ].contiguous()


class Pyramid:
    def __init__(
            self,
            telescope_resolution = 128,
            telescope_diameter = 3.0,
            telescope_samp = 2,
            output_resolution = 128,
            filter_ratio = 1.0,
            nHeads = 4,
            alpha = 2.4,
            crop_mode = "pupils",
            crop_pos_noise = 0,
            crop_size_noise = 0,
            offset = 0,
            wavelength = 635e-9,
            pixel_pitch = 3.74e-6,
            precision = None,
            device = "cpu",
            telescope_pupil = None
            ):
        
        self.precision = precision
        self.device = device

        self.crop_mode = crop_mode
        self.crop_pos_noise = crop_pos_noise
        self.crop_size_noise = crop_size_noise
        self.offset = offset

        self.telescope_resolution = telescope_resolution
        self.telescope_diameter = telescope_diameter
        self.telescope_samp = telescope_samp
        self.output_resolution = output_resolution

        self.filter_resolution_total = 2* telescope_samp * telescope_resolution
        self.filter_ratio = filter_ratio
        self.filter_resolution = int(self.filter_ratio * self.filter_resolution_total)


        self.nHeads = nHeads
        self.alpha = alpha
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch
        if telescope_pupil is None:
            self.telescope_pupil = circular_pupil(
                telescope_resolution,
                device=self.device,
                dtype=precision.real,
            )
        else:
            expected_shape = (
                1,
                1,
                telescope_resolution,
                telescope_resolution,
            )
            if tuple(telescope_pupil.shape) != expected_shape:
                raise ValueError(
                    "telescope_pupil debe tener shape "
                    f"{expected_shape}, recibido "
                    f"{tuple(telescope_pupil.shape)}"
                )
            self.telescope_pupil = telescope_pupil.to(
                device=self.device,
                dtype=precision.real,
            )

        self.pyramid_mask = fourier_geometry(alpha=self.alpha, 
                                             nhead=self.nHeads, 
                                             nPx = self.filter_resolution,
                                             wvl = self.wavelength * 1e9,
                                             ps = self.pixel_pitch * 1e9,
                                             precision = self.precision,
                                             device = self.device
                                             )
        self.pyramid_mask = torch.conj(self.pyramid_mask)
        
        self.mask_phasor = pad2size(self.pyramid_mask,(self.filter_resolution_total,self.filter_resolution_total))
        self.mask = torch.angle(self.mask_phasor)

        self.coords, self.crop_sizes = predict_pupil_centers_m(
            alpha_deg=alpha,
            fovPx=self.filter_resolution_total,
            ps_m=self.pixel_pitch,
            wvl_m=self.wavelength,
            nhead=self.nHeads,
            pupil_diam_px=self.telescope_resolution,
            offset_px=self.offset,
            device=self.device,
            return_crop_sizes=True,
        )
        # ============================================================
        # Precomputed crop geometry
        # ============================================================

        crop_sizes_tensor = torch.as_tensor(
            self.crop_sizes,
            dtype=torch.int64,
            device="cpu",
        ).reshape(-1)

        n_crop_centers = len(
            self.coords
        )

        if crop_sizes_tensor.numel() != n_crop_centers:
            raise ValueError(
                "crop_sizes and coords must have "
                "the same number of pupils."
            )

        base_crop_size = int(
            crop_sizes_tensor.max().item()
        )

        # Keep crop size even, matching the previous implementation.
        if base_crop_size % 2 != 0:
            base_crop_size += 1

        self._base_crop_size = base_crop_size

        minimum_crop_size = (
            self._base_crop_size
            - 2 * int(self.crop_size_noise)
        )

        if minimum_crop_size <= 0:
            raise ValueError(
                "crop_size_noise produces a non-positive crop size: "
                f"{minimum_crop_size}"
            )

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """
        Expected WFS output shape for one realization.

        This is derived directly from the WFS configuration and
        does not require executing a propagation.
        """

        if self.crop_mode == "pupils":
            return (
                1,
                int(self.nHeads),
                int(self.output_resolution),
                int(self.output_resolution),
            )

        if self.crop_mode == "full_frame":
            return (
                1,
                1,
                int(self.filter_resolution_total),
                int(self.filter_resolution_total),
            )

        raise ValueError(
            "Unsupported crop_mode: "
            f"{self.crop_mode!r}. "
            "Expected 'pupils' or 'full_frame'."
        )

    def propagate(
        self,
        phi,
        pupil,
        no_crop=False,
        return_both=False,
        return_boxes=False,
    ):
        """
        phi:   [B, 1, N, N]
        pupil: [1, 1, N, N] o [B, 1, N, N]

        Opciones
        --------
        no_crop=True
            -> I_full

        return_both=True
            -> I_full, I_resized

        return_boxes=True
            -> además devuelve los bounding boxes EXACTOS
            usados por crop_pyr.

        Los boxes tienen formato:
            (x0, y0, x1, y1)

        y están expresados en coordenadas del I_full original.
        """

        # ============================================================
        # Campo complejo
        # ============================================================

        field = pupil * torch.exp(
            1j * phi
        )

        field_pad = pad2size(
            field,
            (
                self.filter_resolution_total,
                self.filter_resolution_total,
            ),
        )

        # ============================================================
        # Fourier plane
        # ============================================================

        psf = torch.fft.fftshift(
            torch.fft.fft2(
                field_pad,
                dim=(-2, -1),
            ),
            dim=(-2, -1),
        )

        # ============================================================
        # Pyramid
        # ============================================================

        wfs_psf = psf * self.mask_phasor

        prop = torch.fft.fft2(
            wfs_psf,
            dim=(-2, -1),
        )

        I_full = torch.abs(prop) ** 2

        I_full = torch.flip(
            I_full,
            dims=[3],
        )

        I_full = torch.flip(
            I_full,
            dims=[2],
        )

        # ============================================================
        # Full frame
        # ============================================================

        if no_crop:
            return I_full

        # ============================================================
        # Crops
        # ============================================================

        if self.crop_mode == "pupils":

            if return_boxes:

                I_resized, boxes = self.crop_pyr(
                    I_full,
                    return_boxes=True,
                )

            else:

                I_resized = self.crop_pyr(
                    I_full,
                    return_boxes=False,
                )

                boxes = None

            if return_both and return_boxes:
                return (
                    I_full,
                    I_resized,
                    boxes,
                )

            if return_both:
                return (
                    I_full,
                    I_resized,
                )

            if return_boxes:
                return (
                    I_resized,
                    boxes,
                )

            return I_resized

        return I_full        

    
    def __call__(self, phi, pupil):
        return self.propagate(phi, pupil)
    
    def resize_tensor(self,x: torch.Tensor, M: int, mode: str = "bilinear") -> torch.Tensor:
        return F.interpolate(x, size=(M, M), mode=mode, align_corners=False if mode in ("bilinear", "bicubic", "trilinear", "linear") else None)
    
    def crop_pyr(
        self,
        intensity: torch.Tensor,
        return_boxes: bool = False,
    ):
        """
        Vectorized pyramid-pupil extraction.

        Parameters
        ----------
        intensity:
            Full WFS intensity with shape [B, 1, H, W].

        return_boxes:
            If True, also return the exact integer bounding boxes
            used to define each crop.

        Returns
        -------
        crops:
            Tensor [B, N, output_resolution, output_resolution].

        boxes:
            Optional NumPy array [B, N, 4] with
            (x0, y0, x1, y1).

        Notes
        -----
        Crop-size jitter is sampled once per batch element and is
        shared between all pupils of that realization.

        Position jitter is sampled independently for every pupil.

        The previous:
            integer crop -> F.interpolate(..., bilinear)

        operation is reproduced with a single batched grid_sample.
        """

        if intensity.ndim != 4:
            raise ValueError(
                "intensity must have shape [B,C,H,W]. "
                f"Received {tuple(intensity.shape)}."
            )

        B, C, H, W = intensity.shape

        if C != 1:
            raise ValueError(
                "Pyramid crop currently expects one full-frame "
                f"intensity channel. Received C={C}."
            )

        device = intensity.device
        dtype = intensity.dtype

        M = int(
            self.output_resolution
        )

        if M <= 0:
            raise ValueError(
                "output_resolution must be positive."
            )

        # ============================================================
        # Base pupil centers
        # ============================================================

        centers_base = torch.as_tensor(
            self.coords,
            device=device,
        )

        if (
            centers_base.ndim != 2
            or centers_base.shape[-1] != 2
        ):
            raise ValueError(
                "self.coords must have shape [N,2]. "
                f"Received {tuple(centers_base.shape)}."
            )

        # Previous implementation used int(self.coords[n, ...]).
        # int64 conversion preserves that integer-center behaviour.
        centers_base = centers_base.to(
            dtype=torch.int64
        )

        N = int(
            centers_base.shape[0]
        )

        # ============================================================
        # Crop-size jitter
        # One size per realization, shared by all its pupils.
        # ============================================================

        crop_size_noise = int(
            self.crop_size_noise
        )

        if crop_size_noise > 0:

            size_step = torch.randint(
                low=-crop_size_noise,
                high=crop_size_noise + 1,
                size=(B,),
                device=device,
                dtype=torch.int64,
            )

        else:

            size_step = torch.zeros(
                B,
                device=device,
                dtype=torch.int64,
            )

        size_n = (
            int(self._base_crop_size)
            + 2 * size_step
        )

        half_n = (
            size_n // 2
        )

        # ============================================================
        # Position jitter
        # Independent dx/dy for every pupil.
        # ============================================================

        crop_pos_noise = int(
            self.crop_pos_noise
        )

        if crop_pos_noise > 0:

            jitter = torch.randint(
                low=-crop_pos_noise,
                high=crop_pos_noise + 1,
                size=(
                    B,
                    N,
                    2,
                ),
                device=device,
                dtype=torch.int64,
            )

        else:

            jitter = torch.zeros(
                (
                    B,
                    N,
                    2,
                ),
                device=device,
                dtype=torch.int64,
            )

        centers = (
            centers_base.unsqueeze(0)
            + jitter
        )

        x = centers[..., 0]
        y = centers[..., 1]

        # ============================================================
        # Integer crop boxes
        # ============================================================

        half = half_n[:, None]

        x0 = x - half
        x1 = x + half

        y0 = y - half
        y1 = y + half

        # ============================================================
        # Sampling coordinates
        # ============================================================
        #
        # Previous implementation:
        #
        #   crop = image[y0:y1, x0:x1]
        #   resize(crop, M, align_corners=False)
        #
        # For align_corners=False the source coordinate for output
        # pixel j is:
        #
        #   src = x0 + (j + 0.5) * size / M - 0.5
        #
        # Clamping to [x0, x1-1] reproduces interpolate's border
        # behaviour inside the extracted crop.
        # ============================================================

        output_coordinate = (
            torch.arange(
                M,
                device=device,
                dtype=dtype,
            )
            + 0.5
        )

        size_float = (
            size_n
            .to(dtype=dtype)
            .view(B, 1, 1)
        )

        x0_float = (
            x0
            .to(dtype=dtype)
            .unsqueeze(-1)
        )

        y0_float = (
            y0
            .to(dtype=dtype)
            .unsqueeze(-1)
        )

        x_last_float = (
            (x1 - 1)
            .to(dtype=dtype)
            .unsqueeze(-1)
        )

        y_last_float = (
            (y1 - 1)
            .to(dtype=dtype)
            .unsqueeze(-1)
        )

        source_x = (
            x0_float
            + output_coordinate.view(1, 1, M)
            * size_float
            / float(M)
            - 0.5
        )

        source_y = (
            y0_float
            + output_coordinate.view(1, 1, M)
            * size_float
            / float(M)
            - 0.5
        )

        # Reproduce F.interpolate border behaviour on each
        # individually extracted crop.
        source_x = torch.maximum(
            source_x,
            x0_float,
        )

        source_x = torch.minimum(
            source_x,
            x_last_float,
        )

        source_y = torch.maximum(
            source_y,
            y0_float,
        )

        source_y = torch.minimum(
            source_y,
            y_last_float,
        )

        # ============================================================
        # Pixel coordinates -> grid_sample normalized coordinates
        # align_corners=False
        # ============================================================

        grid_x = (
            2.0
            * (source_x + 0.5)
            / float(W)
            - 1.0
        )

        grid_y = (
            2.0
            * (source_y + 0.5)
            / float(H)
            - 1.0
        )

        # source_x describes output columns.
        # source_y describes output rows.
        grid_x = (
            grid_x
            .unsqueeze(-2)
            .expand(
                B,
                N,
                M,
                M,
            )
        )

        grid_y = (
            grid_y
            .unsqueeze(-1)
            .expand(
                B,
                N,
                M,
                M,
            )
        )

        grid = torch.stack(
            (
                grid_x,
                grid_y,
            ),
            dim=-1,
        )

        # Instead of repeating the input N times, concatenate the
        # N pupil grids along the output-height dimension.
        grid = grid.reshape(
            B,
            N * M,
            M,
            2,
        )

        # ============================================================
        # ONE vectorized interpolation operation
        # ============================================================

        sampled = F.grid_sample(
            intensity,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # sampled:
        # [B, 1, N*M, M]
        #
        # -> [B, N, M, M]
        crops = (
            sampled
            .reshape(
                B,
                C,
                N,
                M,
                M,
            )
            [:, 0]
            .contiguous()
        )

        # ============================================================
        # Optional debug boxes
        # ============================================================

        if return_boxes:

            boxes = torch.stack(
                (
                    x0,
                    y0,
                    x1,
                    y1,
                ),
                dim=-1,
            )

            boxes = (
                boxes
                .detach()
                .cpu()
                .numpy()
            )

            return crops, boxes

        return crops