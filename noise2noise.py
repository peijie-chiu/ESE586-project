import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

from network.unet import UNet
from tools.mriutils import fftshift3d

class Noise2Noise():
    def __init__(self, model, optim, device):
        self.device = device
        self.model = model
        self.optim = optim

    def train_epoch(self, train_loader):
        for batch_idx, (source, target, spec_keep, spec_mask) in enumerate(train_loader):
            source, target = source.to(self.device), target.to(self.device)
            # Denoise image
            denoised = self.model(source)
            denoised_orig = denoised

            denoised_spec = torch.fft.fft2(torch.complex(denoised, torch.zeros_like(denoised)))
            denoised_spec = fftshift3d(denoised_spec, ifft=False)
            denoised_spec = spec_keep * spec_mask + denoised_spec * (1. - spec_mask)
            denoised = torch.real(torch.fft.ifft2(fftshift3d(denoised_spec, ifft=True)))

            # Keep MSE for each item in the minibatch for PSNR computation:
            # targets_clamped = torch.clip(target, 0, 1)
            # denoised_clamped = torch.clip(denoised, 0, 1)
            # loss_clamped = torch.mean((targets_clamped - denoised_clamped)**2, dim=(1, 2))
            loss = F.mse_loss(target, denoised)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def val_epoch(self, val_loader):
        pass