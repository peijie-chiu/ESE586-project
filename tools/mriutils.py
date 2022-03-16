import numpy as np
import torch


def fftshift2d(x:np.ndarray, ifft=False) -> np.ndarray:
    """
    Shift the zero frequency to the center of the K-space
    """
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def fftshift3d(x:torch.Tensor, ifft=False) -> torch.Tensor:
    """
    Shift the zero frequency to the center of the K-space
    """
    assert (len(x.shape) == 3) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.size[1] // 2) + (0 if ifft else 1)
    s1 = (x.size[2] // 2) + (0 if ifft else 1)
    x = torch.cat([x[:, s0:, :], x[:, :s0, :]], dim=1)
    x = torch.cat([x[:, :, s1:], x[:, :, :s1]], dim=2)
    return x

def img2kspace(img):
    """
    Project the MRI image to its K-space via a simple Fourier Transform
    """
    spec = np.fft.fft2(img).astype(np.complex64)
    spec = fftshift2d(spec)

    return spec

def kspace2img(spec):
    """
    Project the K-space data back to its image representation
    """
    return np.real(np.fft.ifft2(spec)).astype(np.float32)


def sampleKspace(spec, p_at_edge=0.025):
    h = [s // 2 for s in spec.shape]
    r = [np.arange(s, dtype=np.float32) - h for s, h in zip(spec.shape, h)]
    r = [x ** 2 for x in r]

    r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
    m = (p_at_edge ** (1./h[1])) ** r

    print('Bernoulli probability at edge = %.5f' % m[h[0], 0])
    print('Average Bernoulli probability = %.5f' % np.mean(m))

    keep = (np.random.uniform(0.0, 1.0, size=spec.shape) ** 2 < m)
    keep = keep & keep[::-1, ::-1]
    spec_keep = spec * keep
    keep_mask = keep.astype(np.float32)
    spec = fftshift2d(spec_keep / (m + ~keep), ifft=True) # Add 1.0 to not-kept values to prevent div-by-zero.
    image = kspace2img(spec)
    return spec_keep, keep_mask, image

