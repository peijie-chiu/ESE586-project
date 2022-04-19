import numpy as np
from scipy import stats
from tqdm import tqdm 
import matplotlib
import matplotlib.patches as mpatches
# matplotlib.use('Agg') # dsiable the GUI of matplotlib
import matplotlib.pyplot as plt

font = {'size': 14}

matplotlib.rc('font', **font)

from tools.commoutils import MatplotlibClearMemory
from tools.mriutils import *

if __name__ == '__main__':
    data_path = './logs/Prediction/exp_0/weights/plots.npz'
    save_path = './logs/Prediction/exp_0'
    data = np.load(data_path)
    noise2noise_plots = data['noise2noise_plots']
    segmentation_plots = data['segmentation_plots']
    dsc = data['dsc']

    dsc_source = dsc[0, ...]
    dsc_denoised = dsc[1, ...]
    dsc_orig = dsc[2, ...]

    #################################### PSNR ####################################
    # psnr_entire_source = []
    # psnr_entire_denoised = []
    # psnr_tumor_source = []
    # psnr_tumor_denoised = []
    # for idx in tqdm(range(noise2noise_plots.shape[1])):
    #     orig_image = noise2noise_plots[0, idx, ...] + 0.5 
    #     source_image = noise2noise_plots[1, idx, ...] + 0.5 
    #     denoised_image = noise2noise_plots[3, idx, ...] + 0.5 
    #     truth_seg = segmentation_plots[3, idx, ...]
    #     binary_mask = truth_seg > 0

    #     if binary_mask.sum() > 0:
    #         psnr_entire_source.append(psnr(source_image, orig_image))
    #         psnr_entire_denoised.append(psnr(denoised_image, orig_image))
    #         psnr_tumor_source.append(psnr(source_image[binary_mask], orig_image[binary_mask]))
    #         psnr_tumor_denoised.append(psnr(denoised_image[binary_mask], orig_image[binary_mask]))
    
    # colors = ['#0000FF', '#FF00FF', '#00FF00', '#FFA500']

    # plt.figure(figsize=(14, 12))
    # bp = plt.boxplot([psnr_entire_source, psnr_tumor_source, psnr_entire_denoised, psnr_tumor_denoised], vert=1, patch_artist = True, notch ='True')
    # for i, patch in enumerate(bp['boxes']):
    #     c = i % len(colors)
    #     patch.set_facecolor(colors[c])
    # plt.ylabel("PSNR")
    # xx = range(1, 5, 1)
    # xticks = ['source', 'source [tumor regions]', 'denoised', 'denoised [tumor regions]']
    # plt.xticks(xx, xticks, rotation=0)
    # plt.savefig(f"{save_path}/psnr_stats.jpg", dpi=300)

    #################################### DSC ####################################
    dsc_source2 = []
    dsc_denoised2 = []
    dsc_orig2 = []
 
    for idx in tqdm(range(noise2noise_plots.shape[1])):
        truth_seg = segmentation_plots[3, idx, ...]
        binary_mask = truth_seg > 0

        if binary_mask.sum() > 0:
            dsc_source2.append(dsc_source[idx, ...])
            dsc_denoised2.append(dsc_denoised[idx, ...])
            dsc_orig2.append(dsc_orig[idx, ...])

    dsc_source = np.asarray(dsc_source2)
    dsc_denoised = np.asarray(dsc_denoised2)
    dsc_orig = np.asarray(dsc_orig2)

    # print(dsc_denoised.shape)

    colors = ['#0000FF', '#FF00FF', '#00FF00']
    dsc_WT = [dsc_source[:, 0], dsc_denoised[:, 0], dsc_orig[:, 0]]
    dsc_TC = [dsc_source[:, 1], dsc_denoised[:, 1], dsc_orig[:, 1]]
    dsc_ET = [dsc_source[:, 2], dsc_denoised[:, 2], dsc_orig[:, 2]]

    ttest_WT_source = stats.ttest_rel(dsc_source[:, 0], dsc_orig[:, 0]).pvalue
    ttest_TC_source = stats.ttest_rel(dsc_source[:, 1], dsc_orig[:, 1]).pvalue
    ttest_ET_source = stats.ttest_rel(dsc_source[:, 2], dsc_orig[:, 2]).pvalue

    ttest_WT_denoised = stats.ttest_rel(dsc_denoised[:, 0], dsc_orig[:, 0]).pvalue
    ttest_TC_denoised = stats.ttest_rel(dsc_denoised[:, 1], dsc_orig[:, 1]).pvalue
    ttest_ET_denoised = stats.ttest_rel(dsc_denoised[:, 2], dsc_orig[:, 2]).pvalue

    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    bp = axes[0].boxplot(dsc_WT, vert=1, patch_artist = True, notch ='True')
    for i, patch in enumerate(bp['boxes']):
        c = i % 3
        patch.set_facecolor(colors[c])

    xx = range(1, 3, 1)
    xticks = [f'{ttest_WT_source:.3f}', f'{ttest_WT_denoised:.3f}']
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(xticks)
    axes[0].tick_params(axis='x', colors='red')

    bp = axes[1].boxplot(dsc_TC, vert=1, patch_artist = True, notch ='True')
    for i, patch in enumerate(bp['boxes']):
        c = i % 3
        patch.set_facecolor(colors[c])

    xx = range(1, 3, 1)
    xticks = [f'{ttest_TC_source:.3f}', f'{ttest_TC_denoised:.3f}']
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(xticks)
    axes[1].tick_params(axis='x', colors='red')

    bp = axes[2].boxplot(dsc_ET, vert=1, patch_artist = True, notch ='True')
    for i, patch in enumerate(bp['boxes']):
        c = i % 3
        patch.set_facecolor(colors[c])

    xx = range(1, 3, 1)
    xticks = [f'{ttest_ET_source:.3f}', f'{ttest_ET_denoised:.3f}']
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(xticks)
    axes[2].tick_params(axis='x', colors='red')

    axes[0].set_ylabel('WT DSC')
    axes[1].set_ylabel('TC DSC')
    axes[2].set_ylabel('ET DSC')

    patch1 = mpatches.Patch(color=colors[0], label='source')
    patch2 = mpatches.Patch(color=colors[1], label='denoised')
    patch3 = mpatches.Patch(color=colors[2], label='original')
    patch4 = mpatches.Patch(color='#FF0000', label='Piared T-Test P-value')
    fig.legend(handles=[patch1, patch2, patch3, patch4],loc='center right')
    plt.tight_layout()

    plt.savefig(f"{save_path}/dsc_stats.jpg", dpi=300)