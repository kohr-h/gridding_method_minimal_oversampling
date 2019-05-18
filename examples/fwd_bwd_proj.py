"""Example demonstrating the basic usage of forward and backward projectors."""

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

# Create functions for forward projection and back-projection
fproj, bproj = gridrec.gridding_fwdproj_backproj(num_px, angles_deg)

# Generate sinogram and (unfiltered) back-projection
sino = fproj(image)
bp = bproj(sino)

# Display results
fig, ax = plt.subplots()
ax.imshow(image, cmap="gray")
ax.set_title("Original Image")
fig, ax = plt.subplots()
ax.imshow(sino, cmap="gray")
ax.set_title("Sinogram")
fig, ax = plt.subplots()
ax.imshow(bp, cmap="gray")
ax.set_title("Back-projection")
