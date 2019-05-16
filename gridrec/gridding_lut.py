import numpy as np
import sys

myfloat = np.float32
myint = np.int
pi = np.pi
eps = 1e-8


class gridding:
    def __init__(self, kernel, ns_grid, ns_ker, width_ker, oversampl, interp, errs):
        self.kernel = kernel
        self.ns_grid = ns_grid
        self.ns_ker = ns_ker
        self.width_ker = width_ker
        self.oversampl = oversampl
        self.interp = interp
        self.errs = errs

    def create_lut_deapod(self):
        if self.kernel == "prolate" or self.kernel == "pswf":
            lut, deapod = self.prolate()

        elif self.kernel == "kaiser-bessel" or self.kernel == "kb":
            lut, deapod = self.kaiser_bessel()

        nh = int(self.ns_grid * 0.5)

        for i in range(1, nh + 1):
            if i % 2 == 0:
                sign = 1.0
            else:
                sign = -1.0

            if i != nh:
                deapod[nh + i] *= sign
                deapod[nh - i] *= sign
            elif i == nh:
                deapod[0] *= sign
        deapod[nh] *= sign

        deapod = np.outer(deapod, deapod)

        return lut, deapod

    def kaiser_bessel(self):
        ##  Compute the optimal beta
        W = self.width_ker
        alpha = self.oversampl
        err = self.errs
        interp = self.interp
        ns_grid = self.ns_grid

        arg = ((W / alpha) * (alpha - 0.5)) ** 2 - 0.8
        ii = np.argwhere(arg < 0.0)
        if len(ii) > 0:
            sys.exit(
                "\n\nERROR inside gridrec_lut.py: the choice of the kernel "
                + "size and oversampling generate nan values for beta!\n\n"
            )

        beta = pi * np.sqrt(arg)

        ##  Compute the optimal sampling density for the kernel
        if interp == "nn":
            ns_ker = int(0.91 / (err * alpha))
        elif interp == "lin":
            ns_ker = int(np.sqrt(0.37 / err) * 1.0 / alpha)

        ##  Compute kernel LUT
        k_max = W / myfloat(2 * ns_grid)
        k = np.linspace(0, k_max, ns_ker)
        arg = 1 - (2 * ns_grid * k / W) ** 2
        arg = np.clip(arg, 0, np.max(arg))
        lut = ns_grid / myfloat(W) * np.i0(beta * np.sqrt(arg))
        lut /= lut[0]
        lut = np.pad(lut, (0, 4), "constant", constant_values=0)

        ##  Compute deapodization matrix
        nh = int(ns_grid * 0.5)
        x = np.linspace(-nh, nh, ns_grid)
        f = (np.pi * W * x / ns_grid) ** 2 - beta ** 2
        f = np.sqrt(f.astype(complex))
        deapod = np.sin(f) / f
        deapod = np.abs(deapod)

        deapod[:] /= deapod[nh]
        deapod += eps

        lmbda = self.normalize_kaiser_bessel()

        norm = myfloat(np.sqrt(1.0 / (W * lmbda)))
        deapod[:] = norm / deapod

        ##  Assign values to the class
        self.ns_ker = ns_ker

        return lut, deapod

    def normalize_kaiser_bessel(self):
        alpha = self.oversampl

        lmbda_lut = np.array(
            [
                [2.0, 0.6735177854455913],
                [1.9, 0.6859972251713252],
                [1.8, 0.7016852052327338],
                [1.7, 0.7186085894388523],
                [1.6, 0.7391321589920385],
                [1.5, 0.7631378768943125],
                [1.4, 0.7915122594523216],
                [1.3, 0.8296167052898601],
                [1.2, 0.8762210440212161],
                [1.1, 0.9373402868152573],
            ],
            dtype=myfloat,
        )

        lmbda_lut[:, :] = lmbda_lut[::-1, :]

        ind_lmbda = np.argwhere(
            np.abs(lmbda_lut - alpha) == np.min(np.abs(lmbda_lut - alpha))
        )

        if alpha != lmbda_lut[ind_lmbda[0, 0], 0]:
            if alpha > lmbda_lut[ind_lmbda[0, 0], 0]:
                d = alpha - lmbda_lut[ind_lmbda[0, 0], 0]
                norm_factor = (1 - d) * lmbda_lut[ind_lmbda[0, 0], 1] + d * lmbda_lut[
                    ind_lmbda[0, 0], 1
                ]
            else:
                d = alpha - lmbda_lut[ind_lmbda[0, 0] - 1, 0]
                norm_factor = (1 - d) * lmbda_lut[
                    ind_lmbda[0, 0] - 1, 1
                ] + d * lmbda_lut[ind_lmbda[0, 0], 1]
        else:
            norm_factor = lmbda_lut[ind_lmbda[0, 0], 1]

        return norm_factor

    def prolate(self):
        ##  Get parameters
        W = self.width_ker
        alpha = self.oversampl
        err = self.errs
        interp = self.interp
        ns_grid = self.ns_grid
        ns_ker = self.ns_ker

        ##  Compute the optimal kernel density
        if ns_ker != 2048:
            if interp == "nn":
                ns_ker = int(0.91 / (err * alpha))
            elif interp == "lin":
                ns_ker = int((0.37 / err) / alpha)

        ##  Compute kernel LUT
        coeffs = np.array(
            [
                0.5767616e02,
                0.0,
                -0.8931343e02,
                0.0,
                0.4167596e02,
                0.0,
                -0.1053599e02,
                0.0,
                0.1662374e01,
                0.0,
                -0.1780527e-00,
                0.0,
                0.1372983e-01,
                0.0,
                -0.7963169e-03,
                0.0,
                0.3593372e-04,
                0.0,
                -0.1295941e-05,
                0.0,
                0.3817796e-07,
            ]
        )

        x = np.arange(0, ns_ker) / myfloat(ns_ker)

        lut = np.polynomial.legendre.legval(x, coeffs) / np.polynomial.legendre.legval(
            0.0, coeffs
        )

        lut = np.pad(lut, (0, 5), "constant", constant_values=0)

        ##  Compute deapodization matrix
        nh = int(ns_grid * 0.5)
        lmbda = 0.99998546
        norm = myfloat(np.sqrt(1.0 / (W * lmbda)))
        scale_ratio = myfloat(ns_ker) / myfloat(nh + 0.5)
        deapod = np.zeros(ns_grid, dtype=myfloat)

        deapod[nh] = norm / lut[nn(0.0)]
        for i in range(1, nh + 1):
            if i != nh:
                deapod[nh + i] = norm / (lut[nn(i * scale_ratio)] + eps)
                deapod[nh - i] = norm / (lut[nn(i * scale_ratio)] + eps)
            elif i == nh:
                deapod[0] = norm / (lut[nn(i * scale_ratio)] + eps)

        return lut, deapod


def nn(x):
    return np.round(x).astype(myint)


def configure_regridding(npix, kernel, oversampl, interp, W, errs):
    ##  Kernel size
    ##  this quantity is fixed, since the regridding
    ##  does not work with smaller and bigger sizes
    W *= 2 / np.pi

    ##  Get size of the Fourier Cartesian grid
    nfreq = int(2 ** (np.ceil(np.log2(npix))) * oversampl)
    while nfreq % 4 != 0:
        nfreq += 1

    ##  Fixed configuration for gridrec
    if kernel == "prolate" and oversampl == 2.0 and interp == "nn":
        ltbl = 2048
    else:
        ltbl = None

    ##  Create regridding object
    gridd = gridding(kernel, nfreq, ltbl, W, oversampl, interp, errs)

    ##  Create LUT and deapodizer for regridding
    lut, deapod = gridd.create_lut_deapod()

    return W, lut, deapod
