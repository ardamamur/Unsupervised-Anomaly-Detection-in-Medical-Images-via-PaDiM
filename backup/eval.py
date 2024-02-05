import cv2
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from skimage.segmentation import mark_boundaries
from skimage.metrics import structural_similarity as ssim
from torch.nn import L1Loss
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve
from matplotlib.colors import Normalize

import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np  
import torch
import torch.nn.functional as F


matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Evaluator:
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, opt, model, outputs, train_outputs, idx, device, test_data_dict):
        # super(Evaluator, self).__init__(model, device, test_data_dict)
        super(Evaluator, self).__init__()

        self.model = model
        self.opt = opt
        self.device = device
        self.train_outputs = train_outputs
        self.idx = idx
        self.test_data_dict = test_data_dict
        self.outputs = outputs
        self.test_output_dict = {key: None for key in self.test_data_dict.keys()}
        
    def embedding_concat(self, x, y):
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


    def extract_features(self, dataset_key, dataset):
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        # extract test set features
        for (img, _, _, _) in tqdm(dataset, '| feature extraction | test | %s |' % 'brainmri'):
            # get the model prediction
            with torch.no_grad():
                _ = self.model(img.to(self.device))
            
            # get intermediate outputs
            for key, value in zip(test_outputs.keys(), outputs):
                test_outputs[key].append(value.cpu().detach())

            # initialize hook outputs
            outputs = []

        for key, value in test_outputs.items():
            test_outputs[key] = torch.cat(value, 0)

        print('first layer shape:', test_outputs['layer1'].shape)
        print('second layer shape:', test_outputs['layer2'].shape)
        print('third layer shape:', test_outputs['layer3'].shape)

        self.test_output_dict[dataset_key] = test_outputs

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])
        
        print('embedding_vectors shape:', embedding_vectors.shape)
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
        return embedding_vectors


    def calculate_mahalanobis_distance(self, embedding_vectors, train_outputs):
        # calculate mahalanobis distance between train_outputs to give anomaly score to each patch position of the test images
        B, C, H, W = embedding_vectors.size()
        print('embedding_vectors shape:', embedding_vectors.shape)

        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
        return dist_list


    def calculate_anomaly_score(self, dataset_key, dataset):
        embedding_vectors = self.extract_features(dataset_key, dataset)
        dist_list = self.calculate_mahalanobis_distance(embedding_vectors, self.train_outputs)
        # upsample to image size to get anomaly score map
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.opt['dataset']['target_size'][0], mode='bicubic',
                                    align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalize the score map
        max_score = score_map.max()
        min_score = score_map.min()
        anomaly_scores = (score_map - min_score) / (max_score - min_score)
        return anomaly_scores
    
    
    def get_optimal_threshold(self, anomaly_maps, masks):
        precision, recall, thresholds = precision_recall_curve(masks.flatten(), anomaly_maps.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        return threshold

    def evaluate(self):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("############### Object Localzation TEST ################")
        self.model.eval()
        metrics = {
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'ROCAUC' : [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
                'ROCAUC' : [],
                'PRO-SCORE': [],
            }
            print('******************* DATASET: {} ****************'.format(dataset_key))
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            anomaly_maps = self.calculate_anomaly_score(dataset_key, dataset)


            roc, pro, tps, fns, fps = 0, 0, 0, 0, []
            print('anomaly_maps shape:', anomaly_maps.shape)
            for idx, data in enumerate(dataset):
                inputs, label, masks, neg_masks = data
                nr_batches, nr_slices, width, height = inputs.shape
                neg_masks[neg_masks > 0.5] = 1
                neg_masks[neg_masks < 1] = 0
 
                #threshold = self.get_optimal_threshold(anomaly_maps[idx], masks[0].cpu().detach().numpy())

                print('nr_batches:', nr_batches)

                for i in range(nr_batches):
                    count = str(idx * nr_batches + i)
                    x_i = inputs[i][0].cpu().detach().numpy()
                    
                    if len(dataset) > 1:
                        ano_map_i = anomaly_maps[i]
                    else:
                        ano_map_i = anomaly_maps

                    mask_i = masks[i][0].cpu().detach().numpy()
                    neg_mask_i = neg_masks[i][0].cpu().detach().numpy()


                    bboxes = cv2.cvtColor(neg_mask_i * 255, cv2.COLOR_GRAY2RGB)
                    cnts_gt = cv2.findContours((mask_i * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []

                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x + w, y + h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)

                    x_pos = ano_map_i * mask_i
                    x_neg = ano_map_i * neg_mask_i
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)

                    x_i_local = mark_boundaries(x_i, res_anomaly, color=(1, 0, 0), mode='thick')
                    x_i_local = mark_boundaries(x_i_local, res_healthy, color=(0, 1, 0), mode='thick')

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_i)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0  # 10% overlap due to large bboxes e.g. for enlarged ventricles
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly, 1))
                    fps.append(fp)
                    precision = tp / max((tp + fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    if int(count) == 0:

                        rows = 1
                        cols = 3

                        elements = [x_i, x_i_local, ano_map_i, bboxes.astype(np.int64), x_pos, x_neg]
                        v_maxs = [1, 1, 0.99, 1, np.max(ano_map_i), np.max(ano_map_i)]

                        titles = ['Input', 'Localization', 'Anomaly Map', 'GT',
                                  str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp),
                                  str(np.round(res_healthy, 2)) + ', FP: ' + str(fp)]
                        
                        norm = Normalize(vmin=0, vmax=max(v_maxs))

                        diffp, axarr = plt.subplots(rows, cols, gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(cols * 4, rows * 4)

                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

        #     for metric in test_metrics:
        #         print('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
        #                                           np.nanstd(test_metrics[metric])))
        #         if metric == 'TP':
        #             print(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
        #         if metric == 'FP':
        #             print(f'FP: {np.sum(test_metrics[metric])} missed')
        #         metrics[metric].append(test_metrics[metric])

        # print('Writing plots...')
        # fig_bps = dict()
        # for metric in metrics:
        #     fig_bp = go.Figure()
        #     x = []
        #     y = []
        #     for idx, dataset_values in enumerate(metrics[metric]):
        #         dataset_name = list(self.test_data_dict)[idx]
        #         for dataset_val in dataset_values:
        #             y.append(dataset_val)
        #             x.append(dataset_name)

        #     fig_bp.add_trace(go.Box(
        #         y=y,
        #         x=x,
        #         name=metric,
        #         boxmean='sd'
        #     ))
        #     title = 'score'
        #     fig_bp.update_layout(
        #         yaxis_title=title,
        #         boxmode='group',  # group together boxes of the different traces for each value of x
        #         yaxis=dict(range=[0, 1]),
        #     )
        #     fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
        #     fig_bps[metric] = fig_bp
        # return metrics, fig_bps, diffp
        return diffp