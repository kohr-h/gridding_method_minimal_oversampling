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
)

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
)

cdef extern void create_fftw_wisdom_file(char *fn, int npix)


@cython.boundscheck(False)
@cython.wraparound(False)
def _create_fftw_wisdom_file(npix, fftw_wisdom_file_name):
    cdef char* cfn = fftw_wisdom_file_name
    create_fftw_wisdom_file(cfn, npix);


@cython.boundscheck(False)
@cython.wraparound(False)
def backproj(
    float[:, ::1] sinogram not None,
    float[::1] angles not None,
    float[::1] param not None,
    float[::1] lut not None,
    float[:, ::1] deapod not None,
    float[::1] filt=None,
):
    cdef:
        int nang, npix
        char *cfn

    fftw_wisdom_file_name = None
    if fftw_wisdom_file_name is None:
        cfn = "/dev/null"
    else:
        cfn = "~/tomcat/Programs/pymodule_gridrec_v4/profile.wis"

    nang, npix = sinogram.shape[0], sinogram.shape[1]
    image = np.zeros((npix, npix), dtype='float32', order='C')

    cdef float [:,::1] cimage = image

    gridrec_v4_backproj(
        &sinogram[0, 0],
        npix,
        nang,
        &angles[0],
        &param[0],
        &lut[0],
        &deapod[0, 0],
        &filt[0],
        &cimage[0, 0],
        cfn
    )

    return image


@cython.boundscheck(False)
@cython.wraparound(False)
def fwdproj(
    float[:, ::1] image not None,
    float[::1] angles not None,
    float[::1] param not None,
    float[::1] lut not None,
    float[:, ::1] deapod not None,
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
    sino = np.zeros((nang, npix), dtype='float32', order='C')

    cdef float [:, ::1] csino = sino

    gridrec_v4_forwproj(
        &csino[0, 0],
        npix,
        nang,
        &angles[0],
        &param[0],
        &lut[0],
        &deapod[0, 0],
        &image[0, 0],
        cfn,
    )

    return sino
