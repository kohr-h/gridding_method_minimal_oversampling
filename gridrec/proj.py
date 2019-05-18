from __future__ import division, print_function
import numpy as np

from . import lut
from . import _gridrec

filters = {
    0: "none",
    1: "ramp",
    2: "shepp-logan",
    3: "hanning",
    4: "hamming",
    5: "lanczos",
    6: "parzen",
}
filter_nums = {v: k for k, v in filters.items()}


def gridding_fwdproj_backproj(
    num_px,
    angles_deg,
    kernel="kb",
    oversampl=1.25,
    W=7.0,
    err_samp=1e-3,
    interp="lin",
    radon_degree=0,
):
    """TODO: doc."""
    num_px = int(num_px)
    angles_deg = np.asarray(angles_deg, dtype="float32")
    kernel, kernel_in = str(kernel).lower(), kernel
    oversampl = float(oversampl)
    W = W * 2 / np.pi
    err_samp = float(err_samp)
    interp, interp_in = str(interp).lower(), interp

    G = lut.next_fast_oversamp_size(num_px, oversampl)
    if kernel in {"pswf", "prolate"}:
        ker_lut, ker_deapod = lut.prolate_lut_deapod(
            W, G, oversampl, interp, eps_s=err_samp
        )
    elif kernel in {"kb", "kaiser-bessel"}:
        ker_lut, ker_deapod = lut.kb_lut_deapod(
            W, G, oversampl, interp, eps_s=err_samp
        )
    else:
        raise ValueError("unknown kernel {!r}".format(kernel_in))

    if interp in {"nn", "nearest"}:
        interp_num = 0
    elif interp in {"lin", "linear"}:
        interp_num = 1
    else:
        raise ValueError("unknown interp {!r}".format(interp_in))

    # Settings array for algorithm
    param = np.array(
        [0, 0, 0, oversampl, interp_num, ker_lut.size - 5, W, radon_degree],
        dtype="float32",
    )

    def forward_projector(image):
        image = image.astype("float32", copy=False)
        return _gridrec.fwdproj(image, angles_deg, param, ker_lut, ker_deapod)

    def back_projector(sino):
        sino = sino.astype("float32", copy=False)
        return _gridrec.backproj(sino, angles_deg, param, ker_lut, ker_deapod)

    return forward_projector, back_projector


def gridding_scipy_operator(
    num_px,
    angles_deg,
    kernel="kb",
    oversampl=1.25,
    W=7.0,
    err_samp=1e-3,
    interp="lin",
    radon_degree=0,
):
    from scipy.sparse.linalg import LinearOperator

    angles_deg = np.asarray(angles_deg, dtype="float32")
    num_angles = angles_deg.size

    fwd_proj, back_proj = gridding_fwdproj_backproj(
        num_px, angles_deg, kernel, oversampl, W, err_samp, interp, radon_degree
    )


    def matvec(a):
        image = np.asarray(a, dtype="float32").reshape((num_px, num_px))
        return fwd_proj(image).ravel()

    def rmatvec(a):
        sino = np.asarray(a, dtype="float32").reshape((num_angles, num_px))
        return back_proj(sino).ravel()

    return LinearOperator(
        shape=(num_angles * num_px, num_px * num_px),
        dtype="float32",
        matvec=matvec,
        rmatvec=rmatvec,
    )


def gridding_fbp(
    num_px,
    angles_deg,
    filter="ramp",
    kernel="kb",
    oversampl=1.25,
    W=7.0,
    err_samp=1e-3,
    interp="lin",
    radon_degree=0,
):
    """TODO: doc."""
    num_px = int(num_px)
    angles_deg = np.asarray(angles_deg, dtype="float32")
    filter, filter_in = str(filter).lower(), filter
    kernel, kernel_in = str(kernel).lower(), kernel
    oversampl = float(oversampl)
    W = W * 2 / np.pi
    err_samp = float(err_samp)
    interp, interp_in = str(interp).lower(), interp

    filter_num = filter_nums.get(filter, None)
    if filter_num is None:
        raise ValueError("unknown filter {!r}".format(filter_in))

    G = lut.next_fast_oversamp_size(num_px, oversampl)
    if kernel in {"pswf", "prolate"}:
        ker_lut, ker_deapod = lut.prolate_lut_deapod(
            W, G, oversampl, interp, eps_s=err_samp
        )
    elif kernel in {"kb", "kaiser-bessel"}:
        ker_lut, ker_deapod = lut.kb_lut_deapod(
            W, G, oversampl, interp, eps_s=err_samp
        )
    else:
        raise ValueError("unknown kernel {!r}".format(kernel_in))

    if interp in {"nn", "nearest"}:
        interp_num = 0
    elif interp in {"lin", "linear"}:
        interp_num = 1
    else:
        raise ValueError("unknown interp {!r}".format(interp_in))

    # Settings array for algorithm
    param = np.array(
        [0, filter_num, 0, oversampl, interp_num, ker_lut.size - 5, W, radon_degree],
        dtype="float32",
    )

    def fbp(sino):
        sino = sino.astype("float32", copy=False)
        return _gridrec.backproj(sino, angles_deg, param, ker_lut, ker_deapod)

    return fbp
