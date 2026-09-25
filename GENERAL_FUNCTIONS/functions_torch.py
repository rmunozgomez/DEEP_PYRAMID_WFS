import torch
import torch.nn.functional as F
import sys

class dC: pass    
def get_precision(type):
    precision = dC()
    if type=='hsingle':
        precision.type = 'hsingle'
        precision.int = torch.int16
        precision.real = torch.float16
        precision.complex = torch.complex64
    elif type=='single':
        precision.type = 'single'
        precision.int = torch.int32
        precision.real = torch.float32
        precision.complex = torch.complex64
    elif type=='double':
        precision.type = 'double'
        precision.int = torch.int64
        precision.real = torch.float64
        precision.complex = torch.complex128
    else:
        sys.exit('precision not recognized')
    return precision

def pad2size(
    x: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("El tensor debe tener shape (B, M, H, W).")
    B, M, H, W = x.shape
    H_out, W_out = out_hw
    if H_out <= 0 or W_out <= 0:
        raise ValueError("out_hw debe ser positivo.")
    y = x
    if H_out < H:
        dh = (H - H_out) // 2
        y = y[:, :, dh:dh + H_out, :]
        H = H_out
    if W_out < W:
        dw = (W - W_out) // 2
        y = y[:, :, :, dw:dw + W_out]
        W = W_out
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
    if n <= 0:
        raise ValueError("n debe ser > 0")

    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    radius_px = min(cx, cy, (n - 1 - cx), (n - 1 - cy))

    y = torch.arange(n, device=device, dtype=torch.float32)
    x = torch.arange(n, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    R = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    if soft_edge_px <= 0.0:
        pupil = (R <= radius_px).to(dtype)
        return pupil.to(dtype).unsqueeze(0).unsqueeze(0)

    # Borde suave: 1 dentro, transición coseno en [r, r+w], 0 fuera
    w = float(soft_edge_px)
    pupil = torch.ones((n, n), device=device, dtype=torch.float32)
    pupil = torch.where(R >= (radius_px + w), torch.zeros_like(pupil), pupil)

    trans = (R > radius_px) & (R < (radius_px + w))
    t = (R[trans] - radius_px) / w  # 0..1
    pupil[trans] = 0.5 * (1.0 + torch.cos(torch.pi * t))  # 1 -> 0 suave
    return pupil.to(dtype).unsqueeze(0).unsqueeze(0)

def circular_pupil_telescope(
    n: int,
    *,
    device=None,
    dtype=torch.float32,
    soft_edge_px: float = 0.0,

    # Compatibilidad con implementación anterior
    spiders: int = 0,
    spiders_px: float = 0.0,

    # NUEVO: ángulo exacto de cada spider
    spider_angles_deg=None,

    # Obstrucción central
    central_obstruction_diam_px: float = 0.0,

    # NUEVO: posición absoluta (x, y) de la obstrucción en píxeles.
    # None -> centro geométrico de la pupila
    central_obstruction_center_px=None,
):
    """
    Genera una pupila circular con:

    - borde externo opcionalmente suave
    - obstrucción central de diámetro configurable
    - posición (x,y) configurable de la obstrucción
    - spiders de ancho configurable
    - ángulo individual configurable para cada spider

    Convención angular de spider_angles_deg:

        0 deg   -> derecha
        90 deg  -> abajo
        180 deg -> izquierda
        270 deg -> arriba

    Esto sigue las coordenadas de una imagen/matriz:
        +x -> derecha
        +y -> abajo

    Cada ángulo representa UN brazo desde el centro hacia afuera.

    Output:
        (1, 1, n, n)
    """

    # ============================================================
    # Validaciones
    # ============================================================

    if n <= 0:
        raise ValueError("n debe ser > 0")

    if spiders < 0:
        raise ValueError("spiders debe ser >= 0")

    if spiders_px < 0:
        raise ValueError("spiders_px debe ser >= 0")

    if central_obstruction_diam_px < 0:
        raise ValueError(
            "central_obstruction_diam_px debe ser >= 0"
        )

    # ============================================================
    # Centro de la pupila principal
    # ============================================================

    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0

    radius_px = min(
        cx,
        cy,
        (n - 1 - cx),
        (n - 1 - cy),
    )

    # ============================================================
    # Grid
    # ============================================================

    y = torch.arange(
        n,
        device=device,
        dtype=torch.float32,
    )

    x = torch.arange(
        n,
        device=device,
        dtype=torch.float32,
    )

    X, Y = torch.meshgrid(
        x,
        y,
        indexing="xy",
    )

    Xc = X - cx
    Yc = Y - cy

    R = torch.sqrt(
        Xc**2 + Yc**2
    )

    # ============================================================
    # Pupila exterior
    # ============================================================

    if soft_edge_px <= 0.0:

        pupil = (
            R <= radius_px
        ).to(torch.float32)

    else:

        w = float(soft_edge_px)

        pupil = torch.ones(
            (n, n),
            device=device,
            dtype=torch.float32,
        )

        pupil = torch.where(
            R >= (radius_px + w),
            torch.zeros_like(pupil),
            pupil,
        )

        transition = (
            (R > radius_px)
            & (R < radius_px + w)
        )

        t = (
            R[transition] - radius_px
        ) / w

        pupil[transition] = (
            0.5
            * (
                1.0
                + torch.cos(torch.pi * t)
            )
        )

    # ============================================================
    # Obstrucción central
    # ============================================================

    if central_obstruction_diam_px > 0.0:

        if central_obstruction_center_px is None:

            obs_cx = cx
            obs_cy = cy

        else:

            if len(central_obstruction_center_px) != 2:
                raise ValueError(
                    "central_obstruction_center_px debe ser (x, y)"
                )

            obs_cx = float(
                central_obstruction_center_px[0]
            )

            obs_cy = float(
                central_obstruction_center_px[1]
            )

        r_obs = (
            float(central_obstruction_diam_px)
            / 2.0
        )

        R_obs = torch.sqrt(
            (X - obs_cx) ** 2
            + (Y - obs_cy) ** 2
        )

        central_mask = (
            R_obs >= r_obs
        )

        pupil = (
            pupil
            * central_mask.to(torch.float32)
        )

    # ============================================================
    # Spiders
    # ============================================================

    if spiders_px > 0.0:

        # --------------------------------------------------------
        # Ángulos explícitos
        # --------------------------------------------------------

        if spider_angles_deg is not None:

            angles_deg = [
                float(angle)
                for angle in spider_angles_deg
            ]

        # --------------------------------------------------------
        # Comportamiento anterior:
        # spiders equiespaciados + rotación aleatoria
        # --------------------------------------------------------

        elif spiders > 0:

            theta0 = (
                torch.rand(
                    (),
                    device=device,
                )
                * (
                    360.0
                    / float(spiders)
                )
            ).item()

            angles_deg = [
                theta0
                + k
                * (
                    360.0
                    / float(spiders)
                )
                for k in range(spiders)
            ]

        else:

            angles_deg = []

        # --------------------------------------------------------
        # Construcción de máscaras
        # --------------------------------------------------------

        spider_mask = torch.ones(
            (n, n),
            device=device,
            dtype=torch.float32,
        )

        half_width = (
            float(spiders_px)
            / 2.0
        )

        for angle_deg in angles_deg:

            theta = torch.tensor(
                angle_deg
                * torch.pi
                / 180.0,
                device=device,
                dtype=torch.float32,
            )

            # Dirección del brazo
            ux = torch.cos(theta)
            uy = torch.sin(theta)

            # Distancia perpendicular a la línea
            dist_to_line = torch.abs(
                -uy * Xc
                + ux * Yc
            )

            # Coordenada longitudinal del brazo
            radial_coord = (
                ux * Xc
                + uy * Yc
            )

            # Solamente hacia un lado del centro
            one_arm = (
                radial_coord >= 0.0
            )

            spider_region = (
                (dist_to_line <= half_width)
                & one_arm
                & (R <= radius_px)
            )

            spider_mask = torch.where(
                spider_region,
                torch.zeros_like(spider_mask),
                spider_mask,
            )

        pupil = pupil * spider_mask

    return (
        pupil
        .to(dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )

def lens2phase(
    n: int,
    f: float,
    pixel_pitch: float,
    wavelength: float,
    *,
    device=None,
    dtype=torch.float32,
    center: tuple[float, float] | None = None,
    wrap_2pi: bool = True,
) -> torch.Tensor:
    """
    Retorna la fase (rad) de un lente delgado en una matriz 2D n×n:
        phi(x,y) = -(pi/(lambda*f)) * (x^2 + y^2)

    Parámetros
    ----------
    n : int
        Tamaño de la matriz (n×n).
    f : float
        Distancia focal [m].
    pixel_pitch : float
        Tamaño de pixel [m/px].
    wavelength : float
        Longitud de onda [m].
    center : (cx, cy) en pixeles, opcional
        Centro del lente (permite subpixel). Default: centro geométrico.
    wrap_2pi : bool
        Si True, envuelve a [0, 2π). Si False, devuelve fase "unwrapped".

    Retorna
    -------
    phi : torch.Tensor
        Tensor (n, n) con fase en radianes.
    """
    if n <= 0:
        raise ValueError("n debe ser > 0")
    if f == 0:
        raise ValueError("f no puede ser 0")
    if pixel_pitch <= 0 or wavelength <= 0:
        raise ValueError("pixel_pitch y wavelength deben ser > 0")

    if center is None:
        cx = (n - 1) / 2.0
        cy = (n - 1) / 2.0
    else:
        cx, cy = center

    # coordenadas en metros
    idx = torch.arange(n, device=device, dtype=torch.float64)
    X, Y = torch.meshgrid((idx - cx) * pixel_pitch, (idx - cy) * pixel_pitch, indexing="xy")

    phi = -(torch.pi / (wavelength * f)) * (X**2 + Y**2)  # rad

    if wrap_2pi:
        phi = torch.remainder(phi, 2.0 * torch.pi)

    return phi.to(dtype)

def get_psf(pupil,phi,fovPx):
    phi = pupil * torch.exp(1j*phi)
    phi = pad2size(phi,(fovPx,fovPx))
    psf = torch.fft.fftshift(torch.fft.fft2(phi,dim=(-2,-1)), dim=(-2,-1))
    return torch.abs(psf)**2
    
def norm_I(I,norm=None):
        if norm==None:
            return I
        elif norm=='max':
            return I/torch.amax(I,dim=(-2,-1),keepdim=True)
        elif norm=='zscore':
            return (I-torch.mean(I,dim=(-2,-1),keepdim=True))/torch.std(I,dim=(-2,-1),keepdim=True)