import numpy as np

EPS = 1e-8


def prolate_lut_deapod(W, G, alpha, interp, S=None, eps_s=1e-3):
    """Return LUT and deapodization matrix for prolate spheroidals.

    For definitions of parameters see `[A+2016]
    <https://doi.org/10.1109/TIP.2016.2516945>`_.

    Parameters
    ----------
    W : float
        Width of the Fourier convolution kernel in pixels.
    G : int
        Size of the oversampled Euclidean Fourier grid.
    alpha : float
        Fourier grid oversampling factor.
    interp : {"nn", "lin"}
        Interpolation used to calculate kernel values from the LUT.
    S : int, optional
        Number of samples for the kernel LUT. By default chosen according to
        formula (12) in the referenced paper.
    eps_s : positive float, optional
        Maximum allowed sampling error.

    Returns
    -------
    lut : 1D numpy.ndarray, dtype float32
        LUT for the PSWF kernel.
    deapod : 2D numpy.ndarray, dtype float32
        PSWF deapodization multiplier.

    References
    ----------
    [A+2016]
        F Arcadu et al. *A Forward Regridding Method With Minimal
        Oversampling for Accurate and Efficient Iterative Tomographic
        Algorithms*. IEEE Trans. on Imag. Proc., 25(3), 2016.
    """
    # Compute the optimal kernel density
    if S is None:
        if interp == "nn":
            S = int(0.91 / (eps_s * alpha))
        elif interp == "lin":
            S = int((0.37 / eps_s) / alpha)

    # Compute kernel LUT using Legendre polynomial approximation
    leg_coeffs = np.array(
        [0.5767616e02, 0.0,
         -0.8931343e02, 0.0,
         0.4167596e02, 0.0,
         -0.1053599e02, 0.0,
         0.1662374e01, 0.0,
         -0.1780527e-00, 0.0,
         0.1372983e-01, 0.0,
         -0.7963169e-03, 0.0,
         0.3593372e-04, 0.0,
         -0.1295941e-05, 0.0,
         0.3817796e-07]
    )
    x = np.linspace(0, 1, S, endpoint=False)
    lut = (
        np.polynomial.legendre.legval(x, leg_coeffs)
        / np.polynomial.legendre.legval(0.0, leg_coeffs)
    )
    lut = np.pad(lut, (0, 5), "constant", constant_values=0)
    lut = lut.astype("float32")

    # Compute deapodization matrix
    nh = G // 2
    lmbda = 0.99998546
    norm = np.sqrt(1 / (W * lmbda))
    scale_ratio = S / (nh + 0.5)
    deapod = np.zeros(G)

    deapod[nh] = norm / lut[0]
    for i in range(1, nh + 1):
        sign = 1 - 2 * np.mod(i, 2)
        if i != nh:
            val = norm / (lut[round(i * scale_ratio)] + eps_s) * sign
            deapod[nh + i] = val
            deapod[nh - i] = val
        elif i == nh:
            deapod[0] = norm / (lut[round(i * scale_ratio)] + eps_s) * sign

    deapod = deapod.astype("float32")

    return lut, np.outer(deapod, deapod)


def kb_lut_deapod(W, G, alpha, interp, S=None, eps_s=1e-3):
    """Return LUT and deapodization matrix for Kaiser-Bessel kernel.

    For definitions of parameters see `[A+2016]
    <https://doi.org/10.1109/TIP.2016.2516945>`_.

    Parameters
    ----------
    W : int
        Width of the Fourier convolution kernel in pixels.
    G : int
        Size of the oversampled Euclidean Fourier grid.
    alpha : float
        Fourier grid oversampling factor.
    interp : {"nn", "lin"}
        Interpolation used to calculate kernel values from the LUT.
    S : int, optional
        Number of samples for the kernel LUT. By default chosen according to
        formula (12) in the referenced paper.
    eps_s : positive float, optional
        Maximum allowed sampling error.

    Returns
    -------
    lut : 1D numpy.ndarray, dtype float32
        LUT for the Kaiser-Bessel kernel.
    deapod : 2D numpy.ndarray, dtype float32
        KB deapodization multiplier.

    References
    ----------
    [A+2016]
        F Arcadu et al. *A Forward Regridding Method With Minimal
        Oversampling for Accurate and Efficient Iterative Tomographic
        Algorithms*. IEEE Trans. on Imag. Proc., 25(3), 2016.
    """
    # Compute the optimal kernel density
    if S is None:
        if interp == "nn":
            S = int(0.91 / (eps_s * alpha))
        elif interp == "lin":
            S = int(np.sqrt(0.37 / eps_s) / alpha)

    # NB: beta / pi in paper notation
    beta = np.sqrt(((W / alpha) * (alpha - 0.5)) ** 2 - 0.8)
    if np.any(np.isnan(beta)):
        raise RuntimeError(
            "choice of kernel size and oversampling lead to NaN values "
            "for computed taper parameter beta"
        )

    ##  Compute kernel LUT
    kmax = W / (2 * G)
    k = np.linspace(0, kmax, S)
    lut = (G / W) * np.i0(
        np.pi * beta * np.sqrt(np.maximum(1 - (2 * G * k / W) ** 2, 0))
    )
    lut /= lut[0]
    lut = np.pad(lut, (0, 4), "constant", constant_values=0)
    lut = lut.astype("float32")

    ##  Compute deapodization matrix
    x = np.linspace(-G // 2, G // 2, G, dtype=complex)
    f = np.sqrt((W * x / G) ** 2 - beta ** 2)
    deapod = np.abs(np.sinc(f))
    deapod[:] /= deapod[G // 2]
    deapod += EPS

    # Normalization factor for KB
    lmbda_lut = np.array(
        [[2.0, 0.6735177854455913],
         [1.9, 0.6859972251713252],
         [1.8, 0.7016852052327338],
         [1.7, 0.7186085894388523],
         [1.6, 0.7391321589920385],
         [1.5, 0.7631378768943125],
         [1.4, 0.7915122594523216],
         [1.3, 0.8296167052898601],
         [1.2, 0.8762210440212161],
         [1.1, 0.9373402868152573]]
    )[::-1]
    lmbda = np.interp(alpha, lmbda_lut[:, 0], lmbda_lut[:, 1])

    norm = np.sqrt(1 / (W * lmbda))
    deapod[:] = norm / deapod
    deapod = deapod.astype("float32")

    # Multiply with +-1 alternatingly, middle one should be +1
    sign_arr = 1 - 2 * np.mod(np.arange(G) - G // 2, 2)
    deapod *= sign_arr

    return lut, np.outer(deapod, deapod)


def next_fast_oversamp_size(n, alpha):
    size = int(2 ** (np.ceil(np.log2(n))) * alpha)
    if size % 4 != 0:
        size += 2
    assert size % 4 == 0
    return size
