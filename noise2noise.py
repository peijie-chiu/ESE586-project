import yaml
import numpy as np
import os
import argparse
import math
from pathlib import Path
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import torchvision.utils as vutils
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from network.unet import UNet
from tools.mriutils import fftshift3d, psnr
from dataloader import MyDataset
from tools.commoutils import *


class Noise2Noise():
    def __init__(self, args, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(params['trainer_params']['gpus']) > 0 else "cpu")
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        # self.model._init_weights()
        self.optim = Adam(self.model.parameters(), 
                        lr=params['exp_params']['LR'], 
                        betas=(0.5, 0.999), 
                        weight_decay=params['exp_params']['weight_decay'])

        self.scheduler = lr_scheduler.ExponentialLR(self.optim, params['exp_params']['scheduler_gamma'])
        self.params = params


        # get dataloader
        self.dataset = MyDataset(params['data_params']['data_dir'], 
                            params['data_params']['dataset'],
                            params['data_params']['modality'],
                            params['data_params']['p_at_edge'],
                            params['data_params']['batch_size'],
                            params['data_params']['num_workers'], True)
        
        save_dir = os.path.join(params['logging_params']['save_dir'],params['logging_params']['name'])
        try:
            os.makedirs(save_dir)
        except:
            pass

        self.save_dir = make_dirs(save_dir)
        Path(f"{self.save_dir}/val_vis").mkdir(exist_ok=True, parents=True)
        Path(f"{self.save_dir}/weights").mkdir(exist_ok=True, parents=True)
        logging_path = os.path.join(self.save_dir, 'Train_log.log')
        self.logger = get_logger(logging_path)
        shutil.copy2(args.filename, self.save_dir)

    def train_epoch(self):
        train_loader = self.dataset.get_train_loader()
        dispatcher = len(train_loader) // 5
        train_epoch_loss = 0.
        train_epoch_loss_clamped = 0.
        psnr_targets = 0.
        psnr_gts = 0.
        self.model.train()
        for batch_idx, sample in enumerate(train_loader):
            source, target = sample['source'].to(self.device), sample['target'].to(self.device)
            spec_keep, keep_mask = sample['spec_val'].to(self.device), sample['spec_mask'].type(torch.complex64).to(self.device)
            # print(source.size(), target.size())
            # Denoise image
            denoised = self.model(source[:, None, :, :])
            denoised_orig = denoised
            denoised_spec = torch.fft.fft2(torch.complex(denoised, torch.zeros_like(denoised)))
            denoised_spec = fftshift3d(denoised_spec, ifft=False)
            denoised_spec = spec_keep * keep_mask + denoised_spec * (1. - keep_mask)
            denoised = torch.real(torch.fft.ifft2(fftshift3d(denoised_spec, ifft=True))).type(target.dtype)

            loss = torch.mean((denoised - target)**2)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep MSE for each item in the minibatch for PSNR computation:
            orig_clamped = torch.clip(sample['orig_img'], -0.5, 0.5).detach().cpu().numpy()
            target_clamped = torch.clip(target, -0.5, 0.5).detach().cpu().numpy()
            denoised_clamped = torch.clip(denoised, -0.5, 0.5).detach().cpu().numpy()
            denoised_orig_clamped = torch.clip(denoised_orig, -0.5, 0.5).detach().cpu().numpy()
            loss_clamped = np.mean((target_clamped-denoised_clamped) ** 2)
            psnr_target = psnr(target_clamped, denoised_clamped)
            psnr_gt = psnr(orig_clamped, denoised_orig_clamped)
            psnr_input = psnr(orig_clamped, target_clamped)

            train_epoch_loss += loss.item()
            train_epoch_loss_clamped += loss_clamped.item()
            psnr_targets += psnr_target
            psnr_gts += psnr_gt

            if (batch_idx+1) % dispatcher == 0:
                print(f"[{batch_idx+1}/{len(train_loader)}] Training Loss: {loss.item():.4f} Clamped Loss:{loss_clamped.item():.4f} PSNR_target:{psnr_target:.3f} PSNR_input:{psnr_input:.3f} PSNR_gt:{psnr_gt:.3f}")

        train_epoch_loss = train_epoch_loss / len(train_loader)
        train_epoch_loss_clamped = train_epoch_loss_clamped / len(train_loader)
        psnr_targets = psnr_targets / len(train_loader)
        psnr_gts = psnr_gts / len(train_loader)

        return train_epoch_loss, train_epoch_loss_clamped, psnr_targets, psnr_gts

    def val_epoch(self):
        val_loader = self.dataset.get_val_loader()
        val_epoch_loss = 0.
        val_epoch_loss_clamped = 0.
        psnr_inputs = 0.
        psnr_targets = 0.
        psnr_gts = 0.
        psnr_targets2orig = 0.
        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                source, target = sample['source'].to(self.device), sample['target'].to(self.device)
                spec_keep, keep_mask = sample['spec_val'].to(self.device), sample['spec_mask'].to(self.device)
                orig_image = sample['orig_img'].to(self.device)
                denoised = self.model(source[:, None, :, :])
                denoised_orig = denoised
                denoised_spec = torch.fft.fft2(torch.complex(denoised, torch.zeros_like(denoised)))
                
                denoised_spec = fftshift3d(denoised_spec, ifft=False)
                denoised_spec = spec_keep * keep_mask + denoised_spec * (1. - keep_mask)
                denoised = torch.real(torch.fft.ifft2(fftshift3d(denoised_spec, ifft=True))).type(target.dtype)

                loss = torch.mean((denoised - target)**2)
                
                source_clamped = torch.clip(source, -0.5, 0.5).detach().cpu().numpy()
                orig_clamped = torch.clip(orig_image, -0.5, 0.5).detach().cpu().numpy()
                target_clamped = torch.clip(target, -0.5, 0.5).detach().cpu().numpy()
                denoised_clamped = torch.clip(denoised, -0.5, 0.5).detach().cpu().numpy()
                denoised_orig_clamped = torch.clip(denoised_orig, -0.5, 0.5).detach().cpu().numpy()
                loss_clamped = np.mean((target_clamped-denoised_clamped) ** 2)
                psnr_target = psnr(target_clamped, denoised_clamped)
                psnr_gt = psnr(orig_clamped, denoised_orig_clamped)
                psnr_input = psnr(orig_clamped, source_clamped)
                
                val_epoch_loss += loss.item()
                val_epoch_loss_clamped += loss_clamped.item()
                psnr_inputs += psnr_input
                psnr_targets += psnr_target
                psnr_gts += psnr_gt
                psnr_targets2orig += psnr(orig_clamped, denoised_clamped)
               
                if batch_idx == 0:
                    bsz = orig_clamped.shape[0]
                    plot = torch.cat([orig_image[:bsz//2, None, :, :] + 0.5, torch.clip(denoised[:bsz//2] + 0.5, 0, 1)[:, None, :, :]], dim=0)
                    plot_grid = vutils.make_grid(plot, padding=2, nrow=int(math.sqrt(bsz)), normalize=True)
                    
        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_epoch_loss_clamped = val_epoch_loss_clamped / len(val_loader)
        psnr_inputs = psnr_inputs / len(val_loader)
        psnr_targets = psnr_targets / len(val_loader)
        psnr_gts = psnr_gts / len(val_loader)
        psnr_targets2orig = psnr_targets2orig / len(val_loader)

        return plot_grid, val_epoch_loss, val_epoch_loss_clamped, psnr_inputs, psnr_targets, psnr_gts, psnr_targets2orig

    def train(self):
        max_epochs = self.params['trainer_params']['max_epochs']
        for ep in tqdm(range(max_epochs)):
            train_epoch_loss, train_epoch_loss_clamped, psnr_targets, psnr_gts = self.train_epoch()
            plot_grid, val_epoch_loss, val_epoch_loss_clamped, psnr_inputs, psnr_targets, psnr_gts, psnr_targets2orig = self.val_epoch()
            vutils.save_image(plot_grid, f"{self.save_dir}/val_vis/ep{ep}.jpg")

            self.scheduler.step()

            print(f"== [{ep}/{max_epochs}] Val Loss: {val_epoch_loss:.4f} Clamped Loss:{val_epoch_loss_clamped:.4f} PSNR_denoised:{psnr_targets2orig:.3f} PSNR_target:{psnr_targets:.3f} ==")
            torch.save(self.model.state_dict(), f"{self.save_dir}/weights/ep{ep}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input arguments to train the Unet for classification')   
    parser.add_argument('--config',  '-c',
                            dest="filename",
                            metavar='FILE',
                            help =  'path to the config file',
                            default='configs/noise2noise.yaml')
    
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    seed = config['exp_params']['manual_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Noise2Noise(args, config)
    model.train()