import yaml
import os
import torch
import numpy as np
from torchvision.models import wide_resnet50_2, resnet18
from data_loader import TrainDataModule
import torch.nn.functional as F
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import cv2


def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None

def load_pretrained_CNN(opt):
    
    model_dict = {
        'resnet18': resnet18,
        'wide_resnet50_2': wide_resnet50_2
    }
    
    # load pretrained CNN
    model_name = opt['model']['backbone']
    t_d = opt['model']['target_dimension']
    r_d = opt['model']['output_dimension']

    model = model_dict[model_name](pretrained=True,  progress=True)
    print('Backbone: {}'.format(opt['model']['backbone']))
    print('Input dim size: {}'.format(t_d))
    print('Output dim size after reduced: {}'.format(r_d))

    return model, t_d, r_d

def load_train_dataset(opt):

    train_data_module = TrainDataModule(
        split_dir=opt['dataset']['ann_path'],
        target_size=opt['dataset']['target_size'],
        batch_size=opt['dataset']['batch_size'])
    
    return train_data_module


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z




def visualize_images(test_img, scores, gts_neg, gts, threshold, save_dir, class_name):
    num = len(test_img)
    if num == 1:
        scores = np.asarray([scores])

    vmax = scores.max() * 255.
    vmin = scores.min() * 255.


    for i in range(num):
        img = test_img[i]
        img = img.transpose(1, 2, 0)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        gt_neg = gts_neg[i].transpose(1, 2, 0).squeeze()
        anno_map = scores[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, gridspec_kw={'wspace': 0, 'hspace': 0})
        fig_img.set_size_inches(6 * 4, 4)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        bboxes = cv2.cvtColor(gt_neg * 255, cv2.COLOR_GRAY2RGB)
        cnts_gt = cv2.findContours((gt * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
        gt_box = []

        for c_gt in cnts_gt:
            x, y, w, h = cv2.boundingRect(c_gt)
            gt_box.append([x, y, x + w, y + h])
            cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
        x_pos = anno_map * gt
        x_neg = anno_map * gt_neg
        res_anomaly = np.sum(x_pos)
        res_healthy = np.sum(x_neg)


        amount_anomaly = np.count_nonzero(x_pos)
        amount_mask = np.count_nonzero(gt)

        tp = 1 if amount_anomaly > 0.1 * amount_mask else 0  # 10% overlap due to large bboxes e.g. for enlarged ventricles
        fn = 1 if tp == 0 else 0
        fp = int(res_healthy / max(res_anomaly, 1))
        precision = tp / max((tp + fp), 1)

        
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Input')

        ax_img[1].imshow(vis_img)
        ax_img[1].title.set_text('Localization')

        
        ax_img[2].imshow(bboxes.astype(np.int64), cmap='gray')
        ax_img[2].title.set_text('GT')

        
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Anomaly Map')

        
        ax_img[4].imshow(x_pos, cmap='gray')
        ax_img[4].title.set_text(str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp))

        
        ax_img[5].imshow(x_neg, cmap='gray')
        ax_img[5].title.set_text(str(np.round(res_healthy, 2)) + ', FP: ' + str(tp))
        
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