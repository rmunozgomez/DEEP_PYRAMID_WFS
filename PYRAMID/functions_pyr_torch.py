import math
import torch
import numpy as np
from torch import unsqueeze as UNZ


def predict_pupil_centers_m(
    alpha_deg,
    nhead,
    fovPx,
    ps_m,
    wvl_m,
    pupil_diam_px,
    offset_px=0,
    y_down=True,
    device="cpu",
    return_crop_sizes=False,
):
    """
    Devuelve coords en el formato que usa crop_pyr:
        coords[:, 0] = x = col
        coords[:, 1] = y = row

    Comportamiento:
    - Si NO hay solape entre pupilas: devuelve una coordenada por cara.
    - Si SÍ hay solape: devuelve una sola coordenada global y un crop size global.

    Orden cuando no hay solape:
    1) El primer crop es el más arriba.
    2) Si hay empate en altura, el más a la izquierda.
    3) Luego sigue en sentido horario.
    """

    if nhead < 2:
        raise ValueError("nhead debe ser >= 2")

    alpha = torch.tensor(alpha_deg * math.pi / 180.0, device=device, dtype=torch.float64)

    N = float(fovPx)
    ps_m = float(ps_m)
    wvl_m = float(wvl_m)

    # magnitud del corrimiento en píxeles
    s = (N * ps_m * torch.tan(alpha) / wvl_m).item()

    # centro del frame
    c = (N - 1.0) / 2.0

    step = 2.0 * math.pi / nhead
    thetas = [(i + 0.5) * step for i in range(nhead)]

    # -------------------------------------------------
    # Calcular centros naturales como (row, col)
    # IMPORTANTE:
    # row <- cos(theta)
    # col <- sin(theta)
    # -------------------------------------------------
    centers_rc = []
    for th in thetas:
        drow = s * math.cos(th)
        dcol = s * math.sin(th)

        if not y_down:
            drow = -drow

        row = int(np.round(c + drow))
        col = int(np.round(c + dcol))
        centers_rc.append([row, col])

    centers_rc = np.asarray(centers_rc, dtype=np.int64)

    # -------------------------------------------------
    # ORDEN:
    # 1) primero el más arriba
    # 2) si hay empate, el más a la izquierda
    # 3) luego en sentido horario
    # -------------------------------------------------
    rc_center = centers_rc.mean(axis=0)
    r0, c0 = rc_center[0], rc_center[1]

    drow = centers_rc[:, 0] - r0
    dcol = centers_rc[:, 1] - c0

    # Ángulo medido desde "arriba" y creciendo en sentido horario
    # arriba -> 0
    # derecha -> pi/2
    # abajo -> pi
    # izquierda -> 3pi/2
    angles = np.mod(np.arctan2(dcol, -drow), 2 * np.pi)

    # Orden horario base partiendo desde arriba
    cw_order = np.argsort(angles)
    centers_sorted = centers_rc[cw_order]

    # Elegir como primer crop:
    # el más arriba; si hay empate, el más a la izquierda
    start_idx = np.lexsort((centers_sorted[:, 1], centers_sorted[:, 0]))[0]

    # Rotar para que ese quede primero
    centers_rc = np.roll(centers_sorted, -start_idx, axis=0)

    # -------------------------------------------------
    # Tamaño base de crop por pupila
    # -------------------------------------------------
    crop_size = int(pupil_diam_px + 2 * offset_px)
    if crop_size % 2 != 0:
        crop_size += 1
    half = crop_size / 2.0

    # Cajas por pupila: [r0, r1, c0, c1]
    boxes = []
    for row, col in centers_rc:
        r_ini = row - half
        r_fin = row + half
        c_ini = col - half
        c_fin = col + half
        boxes.append([r_ini, r_fin, c_ini, c_fin])
    boxes = np.asarray(boxes, dtype=np.float64)

    # -------------------------------------------------
    # Detectar si existe solape entre cualquier par
    # -------------------------------------------------
    overlap_exists = False
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            r_overlap = min(boxes[i, 1], boxes[j, 1]) > max(boxes[i, 0], boxes[j, 0])
            c_overlap = min(boxes[i, 3], boxes[j, 3]) > max(boxes[i, 2], boxes[j, 2])
            if r_overlap and c_overlap:
                overlap_exists = True
                break
        if overlap_exists:
            break

    # -------------------------------------------------
    # Caso 1: sin solape -> una cara por crop
    # -------------------------------------------------
    if not overlap_exists:
        coords_xy = np.stack([centers_rc[:, 1], centers_rc[:, 0]], axis=1)  # (x, y)

        if return_crop_sizes:
            crop_sizes = np.full((coords_xy.shape[0],), crop_size, dtype=np.int64)
            return coords_xy, crop_sizes

        return coords_xy

    # -------------------------------------------------
    # Caso 2: con solape -> un crop global
    # -------------------------------------------------
    r_min = np.min(boxes[:, 0])
    r_max = np.max(boxes[:, 1])
    c_min = np.min(boxes[:, 2])
    c_max = np.max(boxes[:, 3])

    row_global = int(np.round((r_min + r_max) / 2.0))
    col_global = int(np.round((c_min + c_max) / 2.0))

    size_global = int(np.ceil(max(r_max - r_min, c_max - c_min)))
    if size_global % 2 != 0:
        size_global += 1

    coords_xy = np.asarray([[col_global, row_global]], dtype=np.int64)

    if return_crop_sizes:
        crop_sizes = np.asarray([size_global], dtype=np.int64)
        return coords_xy, crop_sizes

    return coords_xy



def fourier_geometry(alpha, nhead, **kwargs):
    """
    Fourier geometry function generated a general geometry given nhead and alpha.

    Same external structure as original code:
        fourier_geometry(alpha, nhead, **kwargs)

    Same output structure:
        return torch.fft.fftshift(fourierMask)

    Corrections:
        1. Uses FFT-consistent integer grid.
        2. Avoids torch.linspace for even nPx.
        3. Avoids Heaviside boolean ambiguity.
        4. Assigns each pixel to one angular sector only.
        5. Keeps original pyramid face angles.
           For nhead=4: 45, 135, 225, 315 degrees.
    """

    def angle_wrap(X, Y):
        """
        Robust angle in [0, 2pi).
        Replaces the quadrant-based angle_wrap.
        """
        return torch.remainder(torch.atan2(Y, X), 2 * torch.pi)

    precision = kwargs.get('precision', torch.float32)
    device = kwargs.get('device', 'cpu')

    nPx = torch.tensor(kwargs.get('nPx', 512), dtype=precision.int)
    nPx_int = int(nPx.item())

    wvl = torch.tensor(
        kwargs.get('wvl', 635),
        dtype=precision.real,
        device=device
    )

    alpha = torch.tensor(
        alpha * torch.pi / 180,
        dtype=precision.real,
        device=device
    )

    ps = torch.tensor(
        kwargs.get('ps', 3.74e3),
        dtype=precision.real,
        device=device
    )

    rooftop = (
        torch.tensor(
            kwargs.get('rooftop', 0),
            dtype=precision.real,
            device=device
        ) * ps
    )

    nhead = torch.tensor(nhead, dtype=precision.int)
    nhead_int = int(nhead.item())

    # ==========================================================
    # Main geometry
    # ==========================================================
    if nhead_int >= 2:

        # ------------------------------------------------------
        # FFT-consistent grid
        # ------------------------------------------------------
        # For nPx = 128:
        #   -64, -63, ..., -1, 0, 1, ..., 63
        #
        # This is the correct centered sampling for an even FFT grid.
        # Do NOT use linspace(-63,63,128), because the step becomes
        # 126/127 and can introduce an effective sub-pixel shift.
        x = (
            torch.arange(
                nPx_int,
                dtype=precision.real,
                device=device
            ) - nPx_int // 2
        ) * ps

        X, Y = torch.meshgrid(x, x, indexing='ij')

        step = torch.tensor(
            2 * torch.pi / nhead_int,
            dtype=precision.real,
            device=device
        )

        k = 2 * torch.pi / wvl

        O_wrap = angle_wrap(X, Y)

        pyr = torch.zeros(
            (nPx_int, nPx_int),
            dtype=precision.complex,
            device=device
        )

        beforeExp = torch.zeros_like(
            pyr,
            dtype=precision.real,
            device=device
        )

        # ------------------------------------------------------
        # Sector assignment without overlap
        # ------------------------------------------------------
        # Same geometry as original:
        #
        # Original:
        #   phase = (nTheta[i] + nTheta[i+1]) / 2
        #
        # Here:
        #   phase = (i + 0.5) * step
        #
        # For nhead=4:
        #   45, 135, 225, 315 degrees
        # ------------------------------------------------------
        sector_idx = torch.floor(O_wrap / step).to(torch.long)
        sector_idx = torch.clamp(sector_idx, 0, nhead_int - 1)

        phase = (sector_idx.to(precision.real) + 0.5) * step

        cor = torch.cos(phase) * X + torch.sin(phase) * Y

        # Same as:
        #   cor = (cor - rooftop) * Heaviside(cor - rooftop)
        # but without boundary ambiguity.
        cor = torch.clamp(cor - rooftop, min=0)

        beforeExp = k * torch.tan(alpha) * cor

    elif nhead_int == 0:

        beforeExp = torch.zeros(
            (nPx_int, nPx_int),
            dtype=precision.real,
            device=device
        )

    else:
        raise KeyError('Incorrect number of heads')

    # ==========================================================
    # Same output structure as your original code
    # ==========================================================
    afterExp = torch.exp(1j * beforeExp)

    fourierMask = UNZ(
        UNZ(
            torch.fft.fftshift(
                afterExp / torch.sum(torch.abs(afterExp.flatten()))
            ),
            0
        ),
        0
    )

    return torch.fft.fftshift(fourierMask)


# def fourier_geometry(alpha,nhead,**kwargs):
#     """
#     Fourier geometry function generated a general geometry given nhead and alpha
#     """
#     def Heaviside(x):
#         out = torch.zeros_like(x)
#         out[x==0] = 0.5
#         out[x>0] = 1 
#         return out.to(torch.bool)
#     def angle_wrap(X,Y):# consider that X is Y and Y is X
#         O = torch.zeros_like(X)
#         mask1 = Heaviside(X)*Heaviside(Y)#(X>=0) & (Y>=0)
#         O[mask1] = torch.atan2(torch.abs(Y[mask1]),torch.abs(X[mask1]))
#         mask2 = Heaviside(-X)*Heaviside(Y)#(X<0) & (Y>=0)
#         O[mask2] = torch.atan2(Y[mask2],X[mask2])
#         mask3 = Heaviside(-X)*Heaviside(-Y)#(X<0) & (Y<0)
#         O[mask3] = torch.atan2(torch.abs(Y[mask3]),torch.abs(X[mask3])) + torch.pi
#         mask4 = Heaviside(X)*Heaviside(-Y)#(X>=0) & (Y<0)
#         O[mask4] = 2*torch.pi-torch.atan2(torch.abs(Y[mask4]),torch.abs(X[mask4]))
#         return O
#     precision = kwargs.get('precision',torch.float32)#get_precision(type='double'))
#     device = kwargs.get('device', 'cpu')
#     nPx = torch.tensor(kwargs.get('nPx',512), dtype=precision.int)
#     wvl = torch.tensor(kwargs.get('wvl',635), dtype=precision.real)
#     alpha = torch.tensor(alpha*torch.pi/180, dtype=precision.real,device=device)
#     ps = torch.tensor(kwargs.get('ps',3.74e3), dtype=precision.real,device=device)   
#     rooftop = (torch.tensor(kwargs.get('rooftop',0),dtype=precision.real,device=device)*ps)
#     nhead = torch.tensor(nhead, dtype=precision.int)# con float funca mal
#     grid
#     if nhead >= 2:
#         frequency grid
#         x = torch.linspace( -(nPx-1)//2,(nPx-1)//2,nPx , dtype=precision.real,device=device)*ps
#         X,Y = torch.meshgrid(x,x, indexing='ij')
        
#         step = torch.tensor(2*np.pi/nhead, dtype=precision.real)
#         nTheta = torch.arange(0,2*np.pi+step,step, dtype=precision.real)
#         k = 2*np.pi/wvl
#         O_wrap = angle_wrap(X,Y)
#         pyr = torch.zeros((nPx,nPx), dtype=precision.complex,device=device)
#         beforeExp = torch.zeros_like(pyr, dtype=precision.real,device=device)
#         for i in range( len(nTheta)-1 ):
#             mask = (Heaviside( O_wrap-nTheta[i] )*Heaviside( nTheta[i+1]-O_wrap ))#< mask bool
#             phase = (nTheta[i]+nTheta[i+1])/2
#             cor = torch.cos(phase)*X + torch.sin(phase)*Y
#             cor = ( (cor-rooftop)*(Heaviside(cor-rooftop).to(precision.real)) )# heaviside is made real}
#             beforeExp += (mask.to(precision.real))*(k*torch.tan(alpha)*cor)
#     elif nhead==0:
#         beforeExp = torch.zeros((nPx,nPx),dtype=precision.complex,device=device)
#     else:
#         raise KeyError('Incorrect number of heads')
#     afterExp = torch.exp( 1j*( beforeExp ) )
#     fourierMask = UNZ( UNZ( torch.fft.fftshift( afterExp/torch.sum( torch.abs( afterExp.flatten() ) ) ) ,0) ,0)
#     return torch.fft.fftshift(fourierMask)
