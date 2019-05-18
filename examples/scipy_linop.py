"""Example for using the SciPy LinearOperator interface."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.sparse.linalg

import gridrec

# Generate example image
image = scipy.misc.ascent().astype("float32")
assert image.ndim ==  2 and image.shape[0] == image.shape[1]
num_px = image.shape[0]

# Use 1 degree increment between projection angles
angles_deg = np.linspace(0, 180, 180, endpoint=False)

# Create function for forward projection and generate sinogram
fproj, _ = gridrec.gridding_fwdproj_backproj(num_px, angles_deg)
sino = fproj(image)

# Make SciPy LinearOperator
linop = gridrec.gridding_scipy_operator(num_px, angles_deg)

# Run 100 iterations of some least-squares solver.
# NB: Input and output are flat vectors, so we need to reshape.
# See documentation of the "lsmr" solver for info on the return values.
result = scipy.sparse.linalg.lsmr(linop, sino.ravel(), maxiter=100)
x, istop, itn, normr, normar, norma, conda, normx = result
rec = x.reshape(num_px, num_px)

# Display results
fig, ax = plt.subplots()
ax.imshow(image, cmap="gray")
ax.set_title("Original Image")
fig, ax = plt.subplots()
ax.imshow(sino, cmap="gray")
ax.set_title("Sinogram")
fig, ax = plt.subplots()
ax.imshow(rec, cmap="gray")
ax.set_title("Least-squares Reconstruction after 100 iterations")



