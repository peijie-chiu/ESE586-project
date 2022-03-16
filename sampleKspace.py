import numpy as np
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

from tools.mriutils import *


# Phantom Demo
orig = shepp_logan_phantom()[1:, 1:]
kspace = img2kspace(orig)

fig, axes = plt.subplots(4, 3, figsize=(10, 10))
axes[0, 0].imshow(orig, cmap='gray')
axes[0, 1].imshow(np.log(np.abs(kspace)), cmap='gray')
axes[0, 2].imshow(kspace2img(fftshift2d(kspace, ifft=True)), cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Full K-space")
axes[0, 2].set_title("Reconstructed Image(IFFT)")

# Undersampled K-space with p=0.60
kspace_keep, keep_mask, recon = sampleKspace(kspace, p_at_edge=0.5)
axes[1, 0].imshow(keep_mask, cmap='gray')
axes[1, 1].imshow(np.log(np.abs(kspace_keep)), cmap='gray')
axes[1, 2].imshow(recon, cmap='gray')
axes[1, 0].set_title("Mask for Sampling K-space")
axes[1, 1].set_title(f"Undersampled K-space(p={np.mean(keep_mask):.2f})")
# axes[1, 2].set_title("Reconstructed Image(IFFT)")

# Undersampled K-space with p=0.37
kspace_keep, keep_mask, recon = sampleKspace(kspace, p_at_edge=0.1)
axes[2, 0].imshow(keep_mask, cmap='gray')
axes[2, 1].imshow(np.log(np.abs(kspace_keep)), cmap='gray')
axes[2, 2].imshow(recon, cmap='gray')
# axes[2, 0].set_title("Mask for Sampling K-space")
axes[2, 1].set_title(f"Undersampled K-space(p={np.mean(keep_mask):.2f})")
# axes[2, 2].set_title("Reconstructed Image(IFFT)")

# Undersampled K-space with p=0.37
kspace_keep, keep_mask, recon = sampleKspace(kspace, p_at_edge=0.025)
axes[3, 0].imshow(keep_mask, cmap='gray')
axes[3, 1].imshow(np.log(np.abs(kspace_keep)), cmap='gray')
axes[3, 2].imshow(recon, cmap='gray')
# axes[3, 0].set_title("Mask for Sampling K-space")
axes[3, 1].set_title(f"Undersampled K-space(p={np.mean(keep_mask):.2f})")
# axes[3, 2].set_title("Reconstructed Image(IFFT)")


# Turn Axes off 
axes = axes.flatten()
for ax in axes:
    ax.axis(False)

plt.suptitle("Under-sampled K-space Phantom Demo")
# plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.01)
# plt.show()
plt.savefig("under_sampled.jpg", dpi=300)