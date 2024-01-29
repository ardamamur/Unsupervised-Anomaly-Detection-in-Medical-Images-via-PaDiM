import yaml
import os
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None
    
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def plot_fig(test_img, scores, gts_full, gts_pos, gts_neg, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt_pos = gts_pos[i].transpose(1, 2, 0).squeeze()
        gt_neg = gts_neg[i].transpose(1, 2, 0).squeeze()
        gt_full = gts_full[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]

        # pos_mask
        pos_mask = mask.copy()
        pos_mask[pos_mask < threshold] = 0
        pos_mask[pos_mask >= threshold] = 1

        # neg_mask
        neg_mask = mask.copy()
        neg_mask[neg_mask > threshold] = 0
        neg_mask[neg_mask <= threshold] = 1

        kernel = morphology.disk(4)
        pos_mask = morphology.opening(pos_mask, kernel)
        neg_mask = morphology.opening(neg_mask, kernel)
        
        pos_mask *= 255
        neg_mask *= 255

        pos_vis_img = mark_boundaries(img, pos_mask, color=(1, 0, 0), mode='thick')
        neg_vis_img = mark_boundaries(img, neg_mask, color=(1, 0, 0), mode='thick')

        fig_img, ax_img = plt.subplots(1, 9, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt_full, cmap='gray')
        ax_img[1].title.set_text('GroundTruth Full')
        ax_img[2].imshow(gt_pos, cmap='gray')
        ax_img[2].title.set_text('GroundTruth Positive')
        ax_img[3].imshow(gt_neg, cmap='gray')
        ax_img[3].title.set_text('GroundTruth Negative')
        ax = ax_img[4].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[4].imshow(img, cmap='gray', interpolation='none')
        ax_img[4].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[4].title.set_text('Predicted heat map')
        ax_img[5].imshow(pos_mask, cmap='gray')
        ax_img[5].title.set_text('Predicted mask Positive')
        ax_img[6].imshow(neg_mask, cmap='gray')
        ax_img[6].title.set_text('Predicted mask Negative')
        ax_img[7].imshow(pos_vis_img)
        ax_img[7].title.set_text('Segmentation result Positive')
        ax_img[8].imshow(neg_vis_img)
        ax_img[8].title.set_text('Segmentation result Negative')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()