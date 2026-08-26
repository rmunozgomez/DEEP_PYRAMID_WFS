import numpy as np
import torch
def zernIndex(mode_j):
    n = int((-1.+np.sqrt(8*(mode_j-1)+1))/2.)
    p = (mode_j-(n*(n+1))/2.)
    k = n % 2
    m = int((p+k)/2.)*2 - k
    if m != 0:
        if mode_j % 2 == 0:
            s = 1
        else:
            s = -1
        m *= s
    return [n, m]
def zernikeRadialFunc(n, m, r):
    try:
        factorial = np.math.factorial
    except:
        import scipy
        factorial = scipy.special.factorial
    R = np.zeros(r.shape)
    for i in range(0, int((n - m) / 2) + 1):
        R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                                        factorial(n - i)) /
                        (factorial(i) *
                            factorial(int(0.5 * (n + m) - i)) *
                            factorial(int(0.5 * (n - m) - i))),
                        dtype='float')
    return R
def get_zernike(pupil, diameter, nModes, remove_piston=1,type="numpy"):
    resolution = pupil.shape[0]
    pupilLogical = np.where(np.reshape(pupil, resolution*resolution) > 0)
    X,Y = np.where(pupil > 0)
    X = (X-(resolution + resolution %
        2-1)/2) / resolution * diameter
    Y = (Y-(resolution + resolution %
            2-1)/2) / resolution * diameter
    
    R = np.sqrt(X**2 + Y**2)
    R = R/R.max()
    theta = np.arctan2(Y,X)
    outFullRes = np.zeros([resolution**2,nModes])

    for i in range(remove_piston,nModes + remove_piston):
        n, m = zernIndex(i+1)
        if m == 0:
            Z = np.sqrt(n+1) * zernikeRadialFunc(n, 0, R)
        else:
            if m > 0:  # j is even
                Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n,
                                                    m, R) * np.cos(m * theta)
            else:  # i is odd
                m = abs(m)
                Z = np.sqrt(2*(n+1)) * zernikeRadialFunc(n,
                                                    m, R) * np.sin(m * theta)
        # if n!=0 and m!=0:
        if n != 0:
            Z -= Z.mean()
            Z *= (1/np.std(Z))
        outFullRes[pupilLogical, i-remove_piston] = Z
    outFullRes = np.reshape(
    outFullRes, [resolution, resolution, nModes])
    zDecomposeMat = np.linalg.pinv(outFullRes.reshape(resolution**2,nModes)) # generar matriz para descompocicion en zernike
    zComposeMat = outFullRes # matriz de modos para mapear coeficientes

    if type == "torch":
        zDecomposeMat, zComposeMat = torch.from_numpy(zDecomposeMat), torch.from_numpy(zComposeMat)
    return zDecomposeMat, zComposeMat

def zernike_decompose_np(phi, zDecomposeMat):
    return zDecomposeMat@phi.reshape(-1)
def zernike_compose_np(zernike_phi_vector, zComposeMat):
    return zComposeMat@zernike_phi_vector

def zernike_decompose_torch(phi, zDecomposeMat):
    # phi: (B,1,N,N)
    # zDecomposeMat: (nModes, N*N)
    phi_flat = phi.flatten(1)                  # (B, N*N)
    return phi_flat @ zDecomposeMat.T          # (B, nModes)

def zernike_compose_torch(zernike_phi_vector, zComposeMat):
    # zernike_phi_vector: (B, nModes)
    # zComposeMat: (N, N, nModes)
    B = zernike_phi_vector.shape[0]
    N = zComposeMat.shape[0]

    zComposeFlat = zComposeMat.reshape(-1, zComposeMat.shape[-1])   # (N*N, nModes)
    phi_flat = zernike_phi_vector @ zComposeFlat.T                  # (B, N*N)
    return phi_flat.reshape(B, 1, N, N)                             # (B,1,N,N)