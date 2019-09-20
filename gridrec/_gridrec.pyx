import cython

import numpy as np
cimport numpy as np

cdef extern void gridrec_v4_backproj(
    float *S,
    int npix,
    int nang,
    float *angles,
    float *param,
    float *lut,
    float *deapod,
    float *filt,
    float *I,
    char *fftwfn,
) nogil

cdef extern void gridrec_v4_forwproj(
    float *S,
    int npix,
    int nang,
    float *angles,
    float *param,
    float *lut,
    float *deapod,
    float *I,
    char *fftwfn,
) nogil

cdef extern void create_fftw_wisdom_file(char *fn, int npix)


@cython.boundscheck(False)
@cython.wraparound(False)
def _create_fftw_wisdom_file(npix, fftw_wisdom_file_name):
    cdef char* cfn = fftw_wisdom_file_name
    create_fftw_wisdom_file(cfn, npix);


@cython.boundscheck(False)
@cython.wraparound(False)
def backproj(
    float[:, ::1] sino not None,
    float[::1] angles not None,
    float[::1] param not None,
    float[::1] lut not None,
    float[:, ::1] deapod not None,
    float[:, ::1] image_out=None,
    float[::1] filt=None,
):
    cdef:
        int nang, npix
        char *cfn

    # TODO: what to do with this?
    fftw_wisdom_file_name = None
    if fftw_wisdom_file_name is None:
        cfn = "/dev/null"
    else:
        cfn = "~/tomcat/Programs/pymodule_gridrec_v4/profile.wis"

    nang, npix = sino.shape[0], sino.shape[1]
    if image_out is None:
        image_out = np.zeros((npix, npix), dtype='float32', order='C')
    else:
        nx_out = image_out.shape[0]
        ny_out = image_out.shape[1]
        if (nx_out, ny_out) != (npix, npix):
            raise ValueError(
                "incompatible `image_out` shape: expected {}, got {}"
                "".format((npix, npix), (nx_out, ny_out))
            )

    with nogil:
        gridrec_v4_backproj(
            &sino[0, 0],
            npix,
            nang,
            &angles[0],
            &param[0],
            &lut[0],
            &deapod[0, 0],
            &filt[0],
            &image_out[0, 0],
            cfn
        )

    return image_out


@cython.boundscheck(False)
@cython.wraparound(False)
def fwdproj(
    float[:, ::1] image not None,
    float[::1] angles not None,
    float[::1] param not None,
    float[::1] lut not None,
    float[:, ::1] deapod not None,
    float[:, ::1] sino_out=None,
    fftw_wisdom_file_name=None,
):
    cdef:
        int nang, npix
        char *cfn

    if fftw_wisdom_file_name is None:
        cfn = "/dev/null"
    else:
        cfn = fftw_wisdom_file_name

    npix = image.shape[0]
    nang = len(angles)
    if sino_out is None:
        sino_out = np.zeros((nang, npix), dtype='float32', order='C')
    else:
        nang_out = sino_out.shape[0]
        npix_out = sino_out.shape[1]
        if (nang_out, npix_out) != (nang, npix):
            raise ValueError(
                "incompatible `sino_out` shape: expected {}, got {}"
                "".format((nang, npix), (nang_out, npix_out))
            )

    with nogil:
        gridrec_v4_forwproj(
            &sino_out[0, 0],
            npix,
            nang,
            &angles[0],
            &param[0],
            &lut[0],
            &deapod[0, 0],
            &image[0, 0],
            cfn,
        )

    return sino_out
