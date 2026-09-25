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

def circular_pupil(n: int, *, device=None, dtype=torch.float32, soft_edge_px: float = 0.0):
    """
    Genera una pupila circular centrada en una matriz n×n, con el radio máximo que cabe.

    - Si soft_edge_px == 0: retorna máscara float 0/1 (borde duro).
    - Si soft_edge_px  > 0: retorna máscara suave en [0,1] con transición coseno.

    Retorna: (pupil, radius_px)
      pupil: (n, n)
      radius_px: float
    """
    if n <= 0:
        raise ValueError("n debe ser > 0")

    # Centro geométrico
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0

    # Radio máximo que cabe (hasta el borde más cercano)
    radius_px = min(cx, cy, (n - 1 - cx), (n - 1 - cy))

    y = torch.arange(n, device=device, dtype=torch.float32)
    x = torch.arange(n, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    R = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    if soft_edge_px <= 0.0:
        pupil = (R <= radius_px).to(dtype)
        return pupil.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    # Borde suave: 1 dentro, transición coseno en [r, r+w], 0 fuera
    w = float(soft_edge_px)
    pupil = torch.ones((n, n), device=device, dtype=torch.float32)
    pupil = torch.where(R >= (radius_px + w), torch.zeros_like(pupil), pupil)

    trans = (R > radius_px) & (R < (radius_px + w))
    t = (R[trans] - radius_px) / w  # 0..1
    pupil[trans] = 0.5 * (1.0 + torch.cos(torch.pi * t))  # 1 -> 0 suave

    return pupil.to(dtype)


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
        
        self.telescope_pupil = circular_pupil(telescope_resolution,device=self.device,dtype=precision.real)
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
        self.piston_wfs = self.propagate(pupil=self.telescope_pupil, phi = self.telescope_pupil)

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
            1j * phi * pupil
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

        B, C, H, W = intensity.shape
        N = self.coords.shape[0]

        crop_sizes = np.asarray(
            self.crop_sizes,
            dtype=np.int64,
        ).reshape(-1)

        if crop_sizes.shape[0] != N:
            raise ValueError(
                "self.crop_sizes debe tener el mismo largo que self.coords"
            )

        base_size = int(
            np.max(crop_sizes)
        )

        if base_size % 2 != 0:
            base_size += 1

        crops = torch.empty(
            (
                B,
                N,
                self.output_resolution,
                self.output_resolution,
            ),
            device=intensity.device,
            dtype=intensity.dtype,
        )

        boxes = np.zeros(
            (
                B,
                N,
                4,
            ),
            dtype=np.int64,
        )

        max_crop_size = (
            base_size
            + 2 * int(self.crop_size_noise)
        )

        max_half = max_crop_size // 2

        pad = (
            max_half
            + int(self.crop_pos_noise)
            + 1
        )

        intensity_pad = F.pad(
            intensity,
            (
                pad,
                pad,
                pad,
                pad,
            ),
            mode="constant",
            value=0.0,
        )

        for b in range(B):

            # ========================================================
            # Crop size jitter
            # Uno distinto por imagen del batch,
            # pero compartido entre todas sus pupilas.
            # ========================================================

            if self.crop_size_noise > 0:

                size_step = int(
                    torch.randint(
                        low=-self.crop_size_noise,
                        high=self.crop_size_noise + 1,
                        size=(1,),
                        device=intensity.device,
                    ).item()
                )

            else:

                size_step = 0

            size_n = (
                base_size
                + 2 * size_step
            )

            if size_n <= 0:
                raise ValueError(
                    f"Crop size inválido: {size_n}"
                )

            half_n = size_n // 2

            # ========================================================
            # Pupilas
            # ========================================================

            for n in range(N):
                # ====================================================
                # Position jitter
                # ====================================================

                if self.crop_pos_noise > 0:

                    dx = int(
                        torch.randint(
                            low=-self.crop_pos_noise,
                            high=self.crop_pos_noise + 1,
                            size=(1,),
                            device=intensity.device,
                        ).item()
                    )

                    dy = int(
                        torch.randint(
                            low=-self.crop_pos_noise,
                            high=self.crop_pos_noise + 1,
                            size=(1,),
                            device=intensity.device,
                        ).item()
                    )

                else:

                    dx = 0
                    dy = 0

                # ====================================================
                # Centro
                # ====================================================

                x = int(
                    self.coords[n, 0]
                ) + dx

                y = int(
                    self.coords[n, 1]
                ) + dy

                # ====================================================
                # Bounding box
                # ====================================================

                x0 = x - half_n
                x1 = x + half_n

                y0 = y - half_n
                y1 = y + half_n

                boxes[
                    b,
                    n,
                    :,
                ] = (
                    x0,
                    y0,
                    x1,
                    y1,
                )

                # ====================================================
                # Crop
                # ====================================================

                xp = x + pad
                yp = y + pad

                crop = intensity_pad[
                    b:b + 1,
                    :,
                    yp - half_n:yp + half_n,
                    xp - half_n:xp + half_n,
                ]

                # ====================================================
                # Resize individual a NN resolution
                # ====================================================

                crop_resized = F.interpolate(
                    crop,
                    size=(
                        self.output_resolution,
                        self.output_resolution,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                crops[
                    b,
                    n,
                    :,
                    :,
                ] = crop_resized[
                    0,
                    0,
                ]

        if return_boxes:
            return crops, boxes

        return crops
