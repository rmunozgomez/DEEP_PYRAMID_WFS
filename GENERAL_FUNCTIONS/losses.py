# losses.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from MODAL_BASIS.Zernike import get_zernike, zernike_compose_torch


@dataclass(frozen=True)
class CoefLossCfg:
    # Available metrics:
    #   "std"            -> original loss
    #   "std_grad_local" -> STD + gradient RMS + local RMS
    metric: str = "std"

    eps: float = 1e-8

    # Only used by metric="std_grad_local"
    grad_weight: float = 0.1
    local_weight: float = 0.1

    # Spatial window sizes used to detect local error accumulations.
    local_windows: Tuple[int, ...] = (4, 8, 16)

    # Fraction of highest-local-error positions used.
    # 0.10 -> worst 10 %
    # 1.00 -> entire pupil
    local_top_fraction: float = 0.10


class CoefLoss(nn.Module):
    """Phi-only spatial loss.

    The network predicts coefficients in the selected DM/Zernike basis.
    They are composed into a phase map and ALL loss terms are evaluated
    only in phase space.

        phi_est = compose(pred, zComposeMat)
        residual = phi_est - phi_target

    Available metrics
    -----------------
    "std":
        Original loss:

            L = mean_batch(STD_pupil(residual))

    "std_grad_local":
        Combined phase-domain loss:

            L = L_std
                + grad_weight  * L_gradient
                + local_weight * L_local

        where:

        L_std:
            Global residual phase STD inside the pupil.

        L_gradient:
            RMS of horizontal/vertical finite differences, considering
            only neighbour pairs fully inside the telescope pupil.
            This gives additional sensitivity to fine spatial structure
            and high spatial frequencies.

        L_local:
            Multi-scale local RMS of the piston-removed residual.
            By default, only the worst 10 % of local regions are averaged,
            making this term sensitive to spatial accumulations of error.

    No loss term is evaluated directly on modal coefficients. Therefore,
    the objective remains independent of whether the representation is
    Zernike, DM influence functions, a non-orthogonal basis, etc.
    """

    def __init__(
        self,
        cfg: Any,
        n_modes: Optional[int] = None,
        tel_diameter: Optional[float] = None,
        tel_resolution: Optional[int] = None,
        tel_pupil: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        dm_basis: bool = False,
        dm_basis_path: Optional[str] = None,
        zComposeMat: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.metric = str(getattr(cfg, "metric", "std")).lower()
        self.eps = float(getattr(cfg, "eps", 1e-8))

        self.grad_weight = float(
            getattr(cfg, "grad_weight", 0.1)
        )
        self.local_weight = float(
            getattr(cfg, "local_weight", 0.1)
        )

        self.local_windows = tuple(
            int(v)
            for v in getattr(
                cfg,
                "local_windows",
                (4, 8, 16),
            )
        )

        self.local_top_fraction = float(
            getattr(
                cfg,
                "local_top_fraction",
                0.10,
            )
        )

        valid_metrics = {
            "std",
            "std_grad_local",
        }

        if self.metric not in valid_metrics:
            raise ValueError(
                f"Unknown metric='{self.metric}'. "
                f"Valid options are {sorted(valid_metrics)}."
            )

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        if self.grad_weight < 0:
            raise ValueError("grad_weight must be >= 0.")

        if self.local_weight < 0:
            raise ValueError("local_weight must be >= 0.")

        if len(self.local_windows) == 0:
            raise ValueError(
                "local_windows must contain at least one window size."
            )

        if any(k <= 0 for k in self.local_windows):
            raise ValueError(
                "All local window sizes must be positive."
            )

        if not (0.0 < self.local_top_fraction <= 1.0):
            raise ValueError(
                "local_top_fraction must be in (0, 1]."
            )

        if tel_pupil is None:
            raise ValueError("tel_pupil cannot be None.")

        pupil = tel_pupil

        if device is not None:
            pupil = pupil.to(device)

        if dtype is not None:
            pupil = pupil.to(dtype)

        if pupil.ndim == 2:
            pupil = pupil.unsqueeze(0)

        elif pupil.ndim == 3:
            if pupil.shape[0] != 1:
                raise ValueError(
                    "A 3-D tel_pupil must have shape [1,H,W]. "
                    f"Received {tuple(pupil.shape)}."
                )

        elif pupil.ndim == 4:
            if pupil.shape[:2] != (1, 1):
                raise ValueError(
                    "A 4-D tel_pupil must have shape [1,1,H,W]. "
                    f"Received {tuple(pupil.shape)}."
                )

            pupil = pupil[:, 0]

        else:
            raise ValueError(
                "tel_pupil must be [H,W], [1,H,W] or [1,1,H,W]. "
                f"Received {tuple(pupil.shape)}."
            )

        self.register_buffer(
            "tel_pupil",
            pupil.contiguous(),
        )

        # Number of spatial positions whose centre lies inside the pupil.
        self._n_pupil_pixels = int(
            (pupil > 0).sum().item()
        )

        if self._n_pupil_pixels <= 0:
            raise ValueError(
                "tel_pupil does not contain any valid pupil pixels."
            )

        self._local_top_k = max(
            1,
            math.ceil(
                self.local_top_fraction
                * self._n_pupil_pixels
            ),
        )

        # --------------------------------------------------------------
        # Phase composition matrix
        # --------------------------------------------------------------

        if zComposeMat is not None:
            compose_matrix = zComposeMat

        elif dm_basis:
            if dm_basis_path is None:
                raise ValueError(
                    "dm_basis=True requires "
                    "dm_basis_path or zComposeMat."
                )

            dm_data = torch.load(
                dm_basis_path,
                map_location="cpu",
            )

            compose_matrix = dm_data["zComposeMat"]

        else:
            if n_modes is None or tel_diameter is None:
                raise ValueError(
                    "n_modes and tel_diameter are required "
                    "when no zComposeMat is supplied."
                )

            _, compose_matrix = get_zernike(
                pupil.squeeze().detach().cpu(),
                diameter=tel_diameter,
                nModes=n_modes,
                type="torch",
            )

        if device is not None:
            compose_matrix = compose_matrix.to(device)

        if dtype is not None:
            compose_matrix = compose_matrix.to(dtype)

        self.register_buffer(
            "zComposeMat",
            compose_matrix.contiguous(),
        )

    # ==================================================================
    # Utilities
    # ==================================================================

    def _prepare_mask(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return mask as [B,H,W] with same dtype/device as value."""

        if value.ndim != 3:
            raise ValueError(
                f"value must be [B,H,W]. "
                f"Received {tuple(value.shape)}."
            )

        if mask.ndim != 3:
            raise ValueError(
                f"mask must be [1,H,W] or [B,H,W]. "
                f"Received {tuple(mask.shape)}."
            )

        if mask.shape[0] == 1:
            mask = mask.expand(
                value.shape[0],
                -1,
                -1,
            )

        elif mask.shape[0] != value.shape[0]:
            raise ValueError(
                "mask batch dimension must be 1 or match value. "
                f"Received value={tuple(value.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        return mask.to(
            dtype=value.dtype,
            device=value.device,
        )

    def _remove_piston(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the spatial mean inside the pupil."""

        mask = self._prepare_mask(
            value,
            mask,
        )

        n_valid = (
            mask.sum(dim=(-2, -1))
            .clamp_min(1.0)
        )

        mean = (
            (value * mask)
            .sum(dim=(-2, -1))
            / n_valid
        )

        centered = (
            value
            - mean[:, None, None]
        )

        return centered * mask

    # ==================================================================
    # 1. Global phase STD
    # ==================================================================

    def _masked_spatial_std(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Spatial phase STD inside the pupil."""

        mask = self._prepare_mask(
            value,
            mask,
        )

        n_valid = (
            mask.sum(dim=(-2, -1))
            .clamp_min(1.0)
        )

        mean = (
            (value * mask)
            .sum(dim=(-2, -1))
            / n_valid
        )

        centered = (
            value
            - mean[:, None, None]
        )

        variance = (
            centered.square()
            * mask
        ).sum(
            dim=(-2, -1)
        ) / n_valid

        return torch.sqrt(
            variance + self.eps
        )

    # ==================================================================
    # 2. Gradient RMS
    # ==================================================================

    def _masked_gradient_rms(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """RMS of nearest-neighbour phase differences.

        Only neighbour pairs whose two pixels are inside the pupil
        contribute to the metric.

        This prevents the artificial pupil-edge discontinuity
        (phase -> zero outside the aperture) from contributing.
        """

        mask = self._prepare_mask(
            value,
            mask,
        )

        # Horizontal finite differences
        dx = (
            value[:, :, 1:]
            - value[:, :, :-1]
        )

        mask_x = (
            mask[:, :, 1:]
            * mask[:, :, :-1]
        )

        # Vertical finite differences
        dy = (
            value[:, 1:, :]
            - value[:, :-1, :]
        )

        mask_y = (
            mask[:, 1:, :]
            * mask[:, :-1, :]
        )

        energy_x = (
            dx.square()
            * mask_x
        ).sum(
            dim=(-2, -1)
        )

        energy_y = (
            dy.square()
            * mask_y
        ).sum(
            dim=(-2, -1)
        )

        n_pairs = (
            mask_x.sum(dim=(-2, -1))
            + mask_y.sum(dim=(-2, -1))
        ).clamp_min(1.0)

        gradient_variance = (
            energy_x + energy_y
        ) / n_pairs

        return torch.sqrt(
            gradient_variance
            + self.eps
        )

    # ==================================================================
    # 3. Local multi-scale RMS
    # ==================================================================

    @staticmethod
    def _same_padding_2d(
        kernel_size: int,
    ) -> Tuple[int, int, int, int]:
        """Asymmetric padding required to preserve H,W for any kernel."""

        pad_before = (
            kernel_size - 1
        ) // 2

        pad_after = (
            kernel_size
            // 2
        )

        # F.pad ordering:
        # left, right, top, bottom
        return (
            pad_before,
            pad_after,
            pad_before,
            pad_after,
        )

    def _masked_local_rms(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale local RMS of the piston-removed residual.

        The local RMS is evaluated with windows specified by
        self.local_windows.

        For local_top_fraction < 1, only the highest-error fraction
        of pupil-centred windows contributes. This explicitly targets
        spatial concentrations of residual error.

        Returns
        -------
        Tensor [B]
            One local-error value per sample.
        """

        mask = self._prepare_mask(
            value,
            mask,
        )

        value_sq = (
            value.square()
            * mask
        ).unsqueeze(1)

        mask_4d = mask.unsqueeze(1)

        local_scores = []

        for kernel_size in self.local_windows:

            padding = self._same_padding_2d(
                kernel_size
            )

            padded_sq = F.pad(
                value_sq,
                padding,
                mode="constant",
                value=0.0,
            )

            padded_mask = F.pad(
                mask_4d,
                padding,
                mode="constant",
                value=0.0,
            )

            # Because both quantities use the same average-pooling
            # normalization, their ratio is equivalent to:
            #
            #   sum(error^2) / sum(mask)
            #
            local_sq_sum = F.avg_pool2d(
                padded_sq,
                kernel_size=kernel_size,
                stride=1,
            )

            local_mask_sum = F.avg_pool2d(
                padded_mask,
                kernel_size=kernel_size,
                stride=1,
            )

            local_mse = (
                local_sq_sum
                / local_mask_sum.clamp_min(
                    self.eps
                )
            )

            local_rms = torch.sqrt(
                local_mse
                + self.eps
            )[:, 0]

            # ----------------------------------------------------------
            # Aggregate only centres inside the pupil
            # ----------------------------------------------------------

            if self.local_top_fraction >= 1.0:

                n_valid = (
                    mask.sum(
                        dim=(-2, -1)
                    )
                    .clamp_min(1.0)
                )

                score = (
                    local_rms
                    * mask
                ).sum(
                    dim=(-2, -1)
                ) / n_valid

            else:

                valid_centres = (
                    mask > 0
                )

                flat_local = (
                    local_rms.flatten(
                        start_dim=1
                    )
                )

                flat_valid = (
                    valid_centres.flatten(
                        start_dim=1
                    )
                )

                # Invalid points cannot appear in top-k.
                flat_local = (
                    flat_local.masked_fill(
                        ~flat_valid,
                        float("-inf"),
                    )
                )

                worst_values = torch.topk(
                    flat_local,
                    k=self._local_top_k,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).values

                score = (
                    worst_values.mean(dim=1)
                )

            local_scores.append(
                score
            )

        # Equal contribution from all spatial scales.
        return torch.stack(
            local_scores,
            dim=0,
        ).mean(
            dim=0
        )

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        phi_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        del target

        if phi_target is None:
            raise ValueError(
                "phi_target must be supplied."
            )

        # --------------------------------------------------------------
        # Compose predicted coefficients -> physical phase
        # --------------------------------------------------------------

        phi_est = zernike_compose_torch(
            zernike_phi_vector=pred,
            zComposeMat=self.zComposeMat,
        )

        if (
            phi_est.ndim == 4
            and phi_est.shape[1] == 1
        ):
            phi_est = phi_est[:, 0]

        elif phi_est.ndim != 3:
            raise ValueError(
                "phi_est must be [B,H,W] or [B,1,H,W]. "
                f"Received {tuple(phi_est.shape)}."
            )

        if (
            phi_target.ndim == 4
            and phi_target.shape[1] == 1
        ):
            phi_target = phi_target[:, 0]

        elif phi_target.ndim != 3:
            raise ValueError(
                "phi_target must be [B,H,W] or [B,1,H,W]. "
                f"Received {tuple(phi_target.shape)}."
            )

        # --------------------------------------------------------------
        # Physical residual phase
        # --------------------------------------------------------------

        phase_error = (
            phi_est
            - phi_target
        ) * self.tel_pupil

        # ==============================================================
        # Original metric
        # ==============================================================

        std_per_sample = (
            self._masked_spatial_std(
                phase_error,
                self.tel_pupil,
            )
        )

        std_loss = (
            std_per_sample.mean()
        )

        if self.metric == "std":

            # EXACT original behaviour.
            return std_loss, {
                "std_spatial": float(
                    std_loss.detach().cpu()
                ),
            }

        # ==============================================================
        # Combined metric:
        #
        # STD + gradient + local
        # ==============================================================

        # Remove global piston before evaluating the auxiliary
        # spatial terms.
        #
        # Gradient is naturally piston-insensitive, but doing this
        # explicitly also guarantees that LocalRMS does not start
        # penalising an unobservable global piston.
        centered_error = (
            self._remove_piston(
                phase_error,
                self.tel_pupil,
            )
        )

        # --------------------------------------------------------------
        # High-spatial-frequency sensitivity
        # --------------------------------------------------------------

        grad_per_sample = (
            self._masked_gradient_rms(
                centered_error,
                self.tel_pupil,
            )
        )

        grad_loss = (
            grad_per_sample.mean()
        )

        # --------------------------------------------------------------
        # Local spatial accumulation sensitivity
        # --------------------------------------------------------------

        local_per_sample = (
            self._masked_local_rms(
                centered_error,
                self.tel_pupil,
            )
        )

        local_loss = (
            local_per_sample.mean()
        )

        # --------------------------------------------------------------
        # Final combined objective
        # --------------------------------------------------------------

        loss = (
            std_loss
            + self.grad_weight * grad_loss
            + self.local_weight * local_loss
        )

        return loss, {
            "loss_total": float(
                loss.detach().cpu()
            ),
            "std_spatial": float(
                std_loss.detach().cpu()
            ),
            "gradient_rms": float(
                grad_loss.detach().cpu()
            ),
            "local_rms": float(
                local_loss.detach().cpu()
            ),
        }