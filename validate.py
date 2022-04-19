import torch
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

from network import unet_seg, unet
from dataloader import MyDataset
from tools.commoutils import *
from tools.mriutils import fftshift3d, psnr
from tools.commoutils import make_one_hot, DiceCoeff


def validate(noise2noise_model, seg_model, val_loader, device, seg=True):
    dsc_metric = DiceCoeff()
    dsc = np.empty((3, len(val_loader), 3))
    truth_seg = np.empty((len(val_loader), 255, 255))
    denoised_seg = np.empty((len(val_loader), 255, 255))
    source_seg = np.empty((len(val_loader), 255, 255))
    orig_seg = np.empty((len(val_loader), 255, 255))
    orig_images = np.empty((len(val_loader), 255, 255))
    denoised_images = np.empty((len(val_loader), 255, 255))
    source_images = np.empty((len(val_loader), 255, 255))
    target_images = np.empty((len(val_loader), 255, 255))
    noise2noise_model.eval()
    seg_model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(val_loader)):
            source, target = sample['source'].to(device), sample['target'].to(device)
            spec_keep, keep_mask = sample['spec_val'].to(device), sample['spec_mask'].to(device)
            orig_image = sample['orig_img'].to(device)
            denoised = noise2noise_model(source[:, None, :, :])
            denoised_spec = torch.fft.fft2(torch.complex(denoised, torch.zeros_like(denoised)))
            denoised_spec = fftshift3d(denoised_spec, ifft=False)
            denoised_spec = spec_keep * keep_mask + denoised_spec * (1. - keep_mask)
            denoised = torch.real(torch.fft.ifft2(fftshift3d(denoised_spec, ifft=True))).type(target.dtype)
            denoised = torch.clip(denoised, -0.5, 0.5)

            orig_images[idx, ...] = orig_image.squeeze().cpu().numpy()
            source_images[idx, ...] = source.squeeze().cpu().numpy()
            target_images[idx, ...] = target.squeeze().cpu().numpy()
            denoised_images[idx, ...] = denoised.squeeze().cpu().numpy()

            if seg:
                truth = sample['truth'].to(device)
                source = F.pad(source, (0, 1, 0, 1), "constant", -0.5)
                orig_image = F.pad(orig_image, (0, 1, 0, 1), "constant", -0.5)
                denoised = F.pad(denoised, (0, 1, 0, 1), "constant", -0.5)

                source_prob = seg_model(source[:, None, ...])
                orig_prob = seg_model(orig_image[:, None, ...])
                denoised_prob = seg_model(denoised[:, None, ...])

                source_pred = source_prob.argmax(dim=1)[:, :-1, :-1]
                orig_pred = orig_prob.argmax(dim=1)[:, :-1, :-1]
                denoised_pred = denoised_prob.argmax(dim=1)[:, :-1, :-1]

                truth1hot = make_one_hot(truth.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)
                source_pred1hot = make_one_hot(source_pred.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)
                orig_pred1hot = make_one_hot(orig_pred.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)
                denoised_pred1hot = make_one_hot(denoised_pred.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)

                WT_source = source_pred1hot[1, ...] + source_pred1hot[2, ...] + source_pred1hot[3, ...] 
                TC_source = source_pred1hot[1, ...] + source_pred1hot[3, ...] 
                ET_source = source_pred1hot[3, ...] 

                WT_orig = orig_pred1hot[1, ...] + orig_pred1hot[2, ...] + orig_pred1hot[3, ...] 
                TC_orig = orig_pred1hot[1, ...] + orig_pred1hot[3, ...] 
                ET_orig = orig_pred1hot[3, ...] 

                WT_denoised = denoised_pred1hot[1, ...] + orig_pred1hot[2, ...] + orig_pred1hot[3, ...] 
                TC_denoised = denoised_pred1hot[1, ...] + orig_pred1hot[3, ...] 
                ET_denoised = denoised_pred1hot[3, ...] 

                WT_truth = truth1hot[1, ...] + truth1hot[2, ...] + truth1hot[3, ...] 
                TC_truth = truth1hot[1, ...] + truth1hot[3, ...] 
                ET_truth = truth1hot[3, ...] 

                dsc[0, idx, 0] = dsc_metric(WT_source, WT_truth).item()
                dsc[0, idx, 1] = dsc_metric(TC_source, TC_truth).item()
                dsc[0, idx, 2] = dsc_metric(ET_source, ET_truth).item()

                dsc[1, idx, 0] = dsc_metric(WT_denoised, WT_truth).item()
                dsc[1, idx, 1] = dsc_metric(TC_denoised, TC_truth).item()
                dsc[1, idx, 2] = dsc_metric(ET_denoised, ET_truth).item()

                dsc[2, idx, 0] = dsc_metric(WT_orig, WT_truth).item()
                dsc[2, idx, 1] = dsc_metric(TC_orig, TC_truth).item()
                dsc[2, idx, 2] = dsc_metric(ET_orig, ET_truth).item()

                source_seg[idx, ...] = source_pred.squeeze().cpu().numpy()
                denoised_seg[idx, ...] = denoised_pred.squeeze().cpu().numpy()
                orig_seg[idx, ...] = orig_pred.squeeze().cpu().numpy()
                truth_seg[idx, ...] = truth.squeeze().cpu().numpy()

    noise2noise_plots = np.stack([orig_images, source_images, target_images, denoised_images])
    segmentation_plots = np.stack([source_seg, denoised_seg, orig_seg, truth_seg])

    return noise2noise_plots, segmentation_plots, dsc


def validate_segementation(model, val_loader, device):
    dsc_metric = DiceCoeff()
    dsc = np.empty((2, len(val_loader), 3))
    truth_seg = np.empty((len(val_loader), 255, 255))
    source_seg = np.empty((len(val_loader), 255, 255))
    orig_seg = np.empty((len(val_loader), 255, 255))
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(val_loader)):
            source, orig_image, truth = sample['source'].to(device), sample['orig_img'].to(device), sample['truth'].to(device)
            source = F.pad(source, (0, 1, 0, 1), "constant", -0.5)
            orig_image = F.pad(orig_image, (0, 1, 0, 1), "constant", -0.5)

            source_prob = model(source[:, None, ...])
            orig_prob = model(orig_image[:, None, ...])

            source_pred = source_prob.argmax(dim=1)[:, :-1, :-1]
            orig_pred = orig_prob.argmax(dim=1)[:, :-1, :-1]

            truth1hot = make_one_hot(truth.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)
            source_pred1hot = make_one_hot(source_pred.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)
            orig_pred1hot = make_one_hot(orig_pred.unsqueeze(dim=1), source_prob.size(1)).squeeze(0)

            # print(truth1hot.size(), source_pred1hot.size(), orig_pred1hot.size())

            WT_source = source_pred1hot[1, ...] + source_pred1hot[2, ...] + source_pred1hot[3, ...] 
            TC_source = source_pred1hot[1, ...] + source_pred1hot[3, ...] 
            ET_source = source_pred1hot[3, ...] 

            WT_orig = orig_pred1hot[1, ...] + orig_pred1hot[2, ...] + orig_pred1hot[3, ...] 
            TC_orig = orig_pred1hot[1, ...] + orig_pred1hot[3, ...] 
            ET_orig = orig_pred1hot[3, ...] 

            WT_truth = truth1hot[1, ...] + truth1hot[2, ...] + truth1hot[3, ...] 
            TC_truth = truth1hot[1, ...] + truth1hot[3, ...] 
            ET_truth = truth1hot[3, ...] 

            dsc[0, idx, 0] = dsc_metric(WT_source, WT_truth)
            dsc[0, idx, 1] = dsc_metric(TC_source, TC_truth)
            dsc[0, idx, 2] = dsc_metric(ET_source, ET_truth)

            dsc[1, idx, 0] = dsc_metric(WT_orig, WT_truth)
            dsc[1, idx, 1] = dsc_metric(TC_orig, TC_truth)
            dsc[1, idx, 2] = dsc_metric(ET_orig, ET_truth)

            source_seg[idx, ...] = source_pred.squeeze().cpu().numpy()
            orig_seg[idx, ...] = orig_pred.squeeze().cpu().numpy()
            truth_seg[idx, ...] = truth.squeeze().cpu().numpy()

    return np.stack([orig_seg, source_seg, truth_seg]), dsc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input arguments to train the Unet for classification')   
    parser.add_argument('--config',  '-c',
                            dest="filename",
                            metavar='FILE',
                            help =  'path to the config file',
                            default='configs/validate.yaml')
    
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

    device = torch.device("cuda" if torch.cuda.is_available() and len(config['exp_params']['gpus']) > 0 else "cpu")
    dataset = MyDataset(config['data_params']['data_dir'], 
                        config['data_params']['dataset'],
                        config['data_params']['modality'],
                        config['data_params']['p_at_edge'],
                        config['data_params']['batch_size'],
                        config['data_params']['num_workers'], True)

    val_dataloader = dataset.get_val_loader()
    # print(len(val_dataloader))

    noise2noise_model = unet.UNet(in_channels=1, out_channels=1).to(device)
    segmentation_model = unet_seg.UNet(in_channels=1, n_classes=4, c=8).to(device)

    # load the pretrained weights
    noise2noise_model_checkpoint = torch.load(config['model_params']['noise2noise_weights_path'])
    segmentation_model_checkpoint = torch.load(config['model_params']['segmentation_weights_path'])['model_state_dict']
    noise2noise_model.load_state_dict(noise2noise_model_checkpoint)
    segmentation_model.load_state_dict(segmentation_model_checkpoint)

    save_dir = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
    try:
        os.makedirs(save_dir)
    except:
        pass

    save_dir = make_dirs(save_dir)
    Path(f"{save_dir}/val_vis").mkdir(exist_ok=True, parents=True)
    Path(f"{save_dir}/weights").mkdir(exist_ok=True, parents=True)

    if config['data_params']['dataset'] == 'brats':
        noise2noise_plots, segmentation_plots, dsc = validate(noise2noise_model, segmentation_model, val_dataloader, device)
        # segmentation_plots, dsc = validate_segementation(segmentation_model, val_dataloader, device)

        with open(f"{save_dir}/weights/plots.npz", "wb") as f:
            np.savez(f, noise2noise_plots=noise2noise_plots, segmentation_plots=segmentation_plots, dsc=dsc)
