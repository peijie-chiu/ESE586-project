import numpy as np
from tqdm import tqdm 
import matplotlib
matplotlib.use('Agg') # dsiable the GUI of matplotlib
import matplotlib.pyplot as plt

from tools.commoutils import MatplotlibClearMemory
from tools.mriutils import *

if __name__ == '__main__':
    data_path = './logs/Prediction/exp_0/weights/plots.npz'
    save_path = './logs/Prediction/exp_0/val_vis'
    data = np.load(data_path)
    noise2noise_plots = data['noise2noise_plots']
    segmentation_plots = data['segmentation_plots']
    dsc = data['dsc']
    
    ################################## plots ##################################
    for idx in tqdm(range(noise2noise_plots.shape[1])):
        orig_image = noise2noise_plots[0, idx, ...] + 0.5 
        source_image = noise2noise_plots[1, idx, ...] + 0.5 
        target_image = noise2noise_plots[2, idx, ...] + 0.5 
        denoised_image = noise2noise_plots[3, idx, ...] + 0.5 
        source_seg = segmentation_plots[0, idx, ...]
        denoised_seg = segmentation_plots[1, idx, ...]
        orig_seg = segmentation_plots[2, idx, ...]
        truth_seg = segmentation_plots[3, idx, ...]

        dsc_source = dsc[0, idx, ...]
        dsc_denoised = dsc[1, idx, ...]
        dsc_orig = dsc[2, idx, ...]

        orig_spec = np.log(np.abs(img2kspace(orig_image)))
        source_spec = np.log(np.abs(img2kspace(source_image)))
        target_spec = np.log(np.abs(img2kspace(target_image)))
        denoised_spec = np.log(np.abs(img2kspace(denoised_image)))

        fig, axes = plt.subplots(3, 4, figsize=(13, 10))
        axes = axes.reshape(-1, 4)
        axes[0, 0].imshow(orig_image, cmap='gray')
        axes[0, 1].imshow(source_image, cmap='gray')
        axes[0, 2].imshow(denoised_image, cmap='gray')
        axes[0, 3].imshow(target_image, cmap='gray')
        axes[1, 0].imshow(orig_spec, cmap='gray')
        axes[1, 1].imshow(source_spec, cmap='gray')
        axes[1, 2].imshow(denoised_spec, cmap='gray')
        axes[1, 3].imshow(target_spec, cmap='gray')
        axes[2, 0].imshow(orig_seg)
        axes[2, 1].imshow(source_seg)
        axes[2, 2].imshow(denoised_seg)
        axes[2, 3].imshow(truth_seg)
        
        axes[0, 0].set_title(f"original")
        axes[0, 1].set_title(f"source [PSNR={psnr(source_image, orig_image):.2f}]")
        axes[0, 2].set_title(f"denoised [PSNR={psnr(denoised_image, orig_image):.2f}]")
        axes[0, 3].set_title(f"target [PSNR={psnr(target_image, orig_image):.2f}]")

        for axis in axes.flatten():
            axis.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.001)
        plt.figtext(0.045, 0.003, f'[WT={dsc_orig[0]:.2f} TC={dsc_orig[1]:.2f} ET={dsc_orig[2]:.2f}]', fontsize=10)
        plt.figtext(0.295, 0.003, f'[WT={dsc_source[0]:.2f} TC={dsc_source[1]:.2f} ET={dsc_source[2]:.2f}]', fontsize=10)
        plt.figtext(0.545, 0.003, f'[WT={dsc_denoised[0]:.2f} TC={dsc_denoised[1]:.2f} ET={dsc_denoised[2]:.2f}]', fontsize=10)
        plt.figtext(0.855, 0.003, 'Truth', fontsize=10)
        plt.savefig(f"{save_path}/{idx}.jpg", dpi=300)

        MatplotlibClearMemory()

    