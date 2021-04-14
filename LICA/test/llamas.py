import os
import torch
import cv2
import json
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from config import system_configs

from utils import crop_image, normalize_

from sample.vis import *

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PLUM = (255, 187, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)
CORAL = (86, 114, 255)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)

id2str = {1: 'l1',
          2: 'l0',
          3: 'r0',
          4: 'r1'}
GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [CORAL, GREEN, DARK_GREEN, PLUM, CHOCOLATE, PEACHPUFF, STATEGRAY]

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        labels[labels == 5] = 0
        scores[labels == 0] = -1
        results = torch.cat([labels.unsqueeze(-1).float(),
                             scores.unsqueeze(-1).float(),
                             out_bbox], dim=-1)

        return results

def kp_detection(db, nnet, result_dir, debug=False, evaluator=None):
    if db.split != "train":
        db_inds = db.db_inds if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds
    num_images = db_inds.size
    # num_images = 10

    multi_scales = db.configs["test_scales"]
    # categories = db.configs["categories"]
    # fvv_categories = 9

    input_size  = db.configs["input_size"]  # [h w]
    # output_size = db.configs["output_sizes"][0]  # [[h,w]]
    # num_roi     = db.num_roi

    postprocessors = {'bbox': PostProcess()}

    lane_outputs = {}
    kappa_outputs = {}

    countnumbers = [0] * 70
    countdists = [0] * 70
    used_kappas = []
    used_dists = []

    for ind in tqdm(range(0, num_images), ncols=67, desc="locating kps"):
        db_ind        = db_inds[ind]
        # image_id      = db.image_ids(db_ind)
        image_file    = db.image_file(db_ind)
        image         = cv2.imread(image_file)
        raw_img = image.copy()
        height, width = image.shape[0:2]
        # item  = db.detections(db_ind) # all in the raw coordinate

        for scale in multi_scales:
            images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
            masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
            orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
            # new_height = int(height * scale)  # 720
            # new_width = int(width * scale)  # 1280
            # new_center = np.array([new_height // 2, new_width // 2])  # 360 640
            # inp_height = input_size[0]
            # inp_width  = input_size[1]
            # pad_image, pad_mask, border, offset = crop_image(image, new_center, [inp_height, inp_width])
            pad_image     = image.copy()
            pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
            resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
            resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
            masks[0][0]   = resized_mask.squeeze()
            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)
            resized_image = resized_image.transpose(2, 0, 1)
            images[0]     = resized_image
            images        = torch.from_numpy(images)
            masks         = torch.from_numpy(masks)

            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            hooks = [
                nnet.model.module.layer4[-1].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)),
                nnet.model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])),
                nnet.model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1]))
            ]

            t0            = time.time()
            outputs, weights = nnet.test([images, masks])
            t             = time.time() - t0

            for hook in hooks:
                hook.remove()
            conv_features = conv_features[0]
            enc_attn_weights = enc_attn_weights[0]
            dec_attn_weights = dec_attn_weights[0]

            results = postprocessors['bbox'](outputs, orig_target_sizes)

            if evaluator is not None:
                evaluator.add_prediction(ind, results.cpu().numpy(), t)

        if debug:
            img_lst = image_file.split('/')
            lane_debug_dir = os.path.join(result_dir, "lane_debug")
            if not os.path.exists(lane_debug_dir):
                os.makedirs(lane_debug_dir)
            img = pad_image
            img_h, img_w, _ = img.shape

            # # --------------------------------------------------------------------------------------------------------

            # # Draw predicted curve kappas
            # pred = results[0].cpu().numpy()
            # pred = pred[pred[:, 0].astype(int) >= 1]
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # cv2.rectangle(img, (5, 10), (5 + 1100, 25 + 30 * pred.shape[0] + 10), (255, 255, 255), thickness=-1)
            # cv2.putText(img, 'Predicted curve parameters:', (10, 30), fontFace=cv2.FONT_HERSHEY_PLAIN,
            #             fontScale=1.5, color=(0, 0, 0), thickness=2)
            # for i, lane in enumerate(pred):
            #     sbc = lane[0]
            #     color = subclasses[int(sbc)]
            #     lane = lane[2:]  # remove label conf
            #     lower, upper = lane[0], lane[1]
            #     lane = lane[2:]  # remove upper, lower positions
            #     ys = np.linspace(lower, upper, num=100)
            #     points = np.zeros((len(ys), 2), dtype=np.int32)
            #     points[:, 1] = (ys * img_h).astype(int)
            #     points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
            #                      lane[5]) * img_w).astype(int)
            #     diff1 = -2 * lane[0] / (ys - lane[1]) ** 3 \
            #             - lane[2] / (ys - lane[1]) ** 2 \
            #             + lane[4]
            #     diff2 = 6 * lane[0] / (ys - lane[1]) ** 4 \
            #             + 2 * lane[2] / (ys - lane[1]) ** 3
            #     kappa = np.abs(diff2) / (1 + diff1 ** 2) ** 1.5
            #     kappa = kappa[(points[:, 0] > 0) & (points[:, 0] < img_w)]
            #     ys    = ys[(points[:, 0] > 0) & (points[:, 0] < img_w)] * img_h
            #     ax.plot(ys, kappa, color=np.array(color)[::-1] / 255, label=id2str[int(sbc)])
            #     # points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]
            # ax.set_yticks(np.arange(65)[::10])
            # ax.set_title('Perceived Curvature')
            # ax.legend(loc='upper right')
            # ax.set_xlabel('v')
            # ax.set_ylabel('Curvature')
            # save_kappa_path = os.path.join(lane_debug_dir, 'kappa_'
            #                                            '{}.jpg'.format(os.path.basename(image_file)[:-4]))
            # kappa_outputs[image_file] = save_kappa_path
            # fig.savefig(save_kappa_path)
            # plt.close(fig)
            # # stat = cv2.imread(save_kappa_path)
            # # # print(stat.shape)
            # # cv2.imshow('1', stat)
            # # cv2.waitKey()
            # # exit()

            # # --------------------------------------------------------------------------------------------------------

            # # Draw dec attn
            # h, w = conv_features.shape[-2:]
            # keep = results[0, :, 0].cpu() == 1.
            # fig, axs = plt.subplots(ncols=keep.nonzero().shape[0] + 1, nrows=2, figsize=(44, 14))
            # # print(keep.nonzero().shape[0], image_file)
            # # colors = COLORS * 100
            # for idx, ax_i in zip(keep.nonzero(), axs.T):
            #     ax = ax_i[0]
            #     ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
            #     ax.axis('off')
            #     ax.set_title('query id: [{}]'.format(idx))
            #     ax = ax_i[1]
            #     preds = db.draw_annotation(ind, pred=results[0][idx].cpu().numpy(), cls_pred=None, img=raw_img)
            #     ax.imshow(preds)
            #     ax.axis('off')
            # fig.tight_layout()
            # img_path = os.path.join(lane_debug_dir, 'decAttn_{}_{}_{}.jpg'.format(
            #     img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
            # plt.savefig(img_path)
            # plt.close(fig)
            # exit()

            # # --------------------------------------------------------------------------------------------------------

            # Draw enc attn
            # if img_lst[-2] != '1492638317092059953_0':
            #     continue
            # img_dir = os.path.join(lane_debug_dir, '{}_{}_{}'.format(
            #     img_lst[-3], img_lst[-2], os.path.basename(image_file[:-4])))
            # if not os.path.exists(img_dir):
            #     os.makedirs(img_dir)
            # f_map = conv_features
            # # print('encoder attention: {}'.format(enc_attn_weights[0].shape))
            # # print('feature map: {}'.format(f_map.shape))
            # shape = f_map.shape[-2:]
            # image_height, image_width, _ = raw_img.shape
            # sattn = enc_attn_weights[0].reshape(shape + shape).cpu()
            # _, label, _ = db.__getitem__(ind)  # 4, 115
            # # print(db.max_points)  # 56
            # for i, lane in enumerate(label):
            #     if lane[0] == 0:  # Skip invalid lanes
            #         continue
            #     lane = lane[3:]  # remove conf, upper and lower positions
            #     xs = lane[:len(lane) // 2]
            #     ys = lane[len(lane) // 2:]
            #     ys = ys[xs >= 0]
            #     xs = xs[xs >= 0]
            #     # norm_idxs = zip(ys, xs)
            #     idxs      = np.stack([ys * image_height, xs * image_width], axis=-1)
            #     attn_idxs = np.stack([ys * shape[0], xs * shape[1]], axis=-1)
            #
            #     for idx_o, idx, num in zip(idxs, attn_idxs, range(xs.shape[0])):
            #         fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(20, 14))
            #         ax_i = axs.T
            #         ax = ax_i[0]
            #         ax.imshow(sattn[..., int(idx[0]), int(idx[1])], cmap='cividis', interpolation='nearest')
            #         ax.axis('off')
            #         ax.set_title('{}'.format(idx_o.astype(int)))
            #         ax = ax_i[1]
            #         ax.imshow(raw_img)
            #         ax.add_patch(plt.Circle((int(idx_o[1]), int(idx_o[0])), color='r', radius=16))
            #         ax.axis('off')
            #         fig.tight_layout()
            #
            #         img_path = os.path.join(img_dir, 'encAttn_lane{}_{}_{}.jpg'.format(
            #             i, num, idx_o.astype(int)))
            #         plt.savefig(img_path)
            #         plt.close(fig)
            # exit()

            # # --------------------------------------------------------------------------------------------------------

            # Draw only prediction
            # pred = results[0].cpu().numpy()
            # preds, lane_points = db.draw_annotation(ind, pred=results[0].cpu().numpy(), cls_pred=None, img=image,
            #                                         draw_gt=False)
            # cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
            #                          + img_lst[-2] + '_'
            #                          + os.path.basename(image_file[:-4]) + '_PRED.jpg'), preds)

            # # --------------------------------------------------------------------------------------------------------

            # # # Draw gt and prediction
            preds, lane_points = db.draw_annotation(ind, pred=results[0].cpu().numpy(), cls_pred=None, img=image)
            cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
                                     + img_lst[-2] + '_'
                                     + os.path.basename(image_file[:-4]) + '_GT+PRED.jpg'), preds)
            # lane_outputs[image_file] = lane_points
            # # --------------------------------------------------------------------------------------------------------

            # # Draw only ground truth
            # preds = db.draw_annotation(ind, pred=None, cls_pred=None, img=image)
            # cv2.imwrite(os.path.join(lane_debug_dir, img_lst[-3] + '_'
            #                          + img_lst[-2] + '_'
            #                          + os.path.basename(image_file[:-4]) + '_GT.jpg'), preds)
            # # --------------------------------------------------------------------------------------------------------

            # # Draw statistically
            # pred = results[0].cpu().numpy()
            # pred = pred[pred[:, 0].astype(int) >= 1]
            # # corKappas, data_file, corDists, pred_lanes, corIDs, corAccs, org_lanes \
            # #     = db.get_corresponding_kappas_for_gts(ind, pred)
            # data_file, pred_lanes, scores, org_lanes, pred_kappas = db.get_corresponding_kappas_for_gts(ind, pred)

            # # This part test the function of get kappas and gts
            # y_samples = list(range(db.img_h))
            # ys = np.array(y_samples) / db.img_h
            # img = pad_image
            # overlay = img.copy()
            # for idc, lane_key in enumerate(pred_lanes):
            #     score = scores[idc]
            #     if score < 0:
            #         continue
            #     lane_value = pred_lanes[lane_key]; lane_value = np.array(lane_value)
            #     org_lane = org_lanes[idc]; org_lane = np.array(org_lane)
            #     pred_kappa = pred_kappas[idc]; pred_kappa = np.array(pred_kappa)
            #     dist = np.abs(lane_value - org_lane)
            #     used_kappa = pred_kappa[np.logical_and(lane_value >= 0, org_lane >= 0)]
            #     used_dist = dist[np.logical_and(lane_value >= 0, org_lane >= 0)]
            #     used_kappas.extend(used_kappa)
            #     used_dists.extend(used_dist)
            #
            #     for uk, ud in zip(used_kappa, used_dist):
            #         bin_idx = int(uk % 55)
            #         countnumbers[bin_idx] += 1
            #         countdists[bin_idx] += ud

            #     fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
            #     ax.plot(used_kappa, used_dist)
            #     save_kappa_path = os.path.join(lane_debug_dir,
            #                                    '{}_{}_{}_kappa_{}_{}.jpg'.format(img_lst[-2], img_lst[-1],
            #                                                                      os.path.basename(image_file[:-4]),
            #                                                                         idc, score))
            #     fig.savefig(save_kappa_path)  # save the figure to file
            #     plt.close(fig)  # close the figure window
            # exit()

            #     points = np.zeros((len(lane_value), 2), dtype=np.int32)
            #     points[:, 1] = (ys * db.img_h).astype(int)
            #     points[:, 0] = np.array(lane_value).astype(int)
            #     points = points[(points[:, 0] > 0) & (points[:, 0] < db.img_w)]
            #     if score > 0:
            #         # draw lane with a polyline on the overlay
            #         for current_point, next_point in zip(points[:-1], points[1:]):
            #             overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=(0, 255, 0), thickness=10)
            # w = 0.6
            # img = ((1. - w) * img + w * overlay).astype(np.uint8)
            # cv2.imshow('data_file', img)
            # cv2.waitKey(0)
            # exit()
            # # --------------------------------------------------------------------------------------------------------

        # # # below is ready to draw comparative results
        # lane_outputs_path = os.path.join(result_dir, 'kappa_figures.json')
        # with open(lane_outputs_path, 'w') as f:
        #     json.dump(kappa_outputs, f)

    # # ----------------------------------------------------------------------------------------------------------------

    # # Draw statistically
    # if debug:
    #     # print('max(kappa): {}'.format(max(used_kappas)))
    #     viznumbers = [0] * 70
    #     vizdists = [0] * 70
    #     kid = 0
    #     for cn, cd in zip(countnumbers, countdists):
    #         if cn > 0:
    #             cd = cd / cn
    #             viznumbers[kid] = float(cn)
    #             vizdists[kid] = cd
    #             # viznumbers.append(float(cn))
    #             # vizdists.append(cd)
    #         kid += 1
    #     print(viznumbers)
    #     print(vizdists)
    #     print(np.mean(np.array(vizdists)))
    #
    #     number_path = os.path.join(result_dir, 'number.json')
    #     with open(number_path, "w") as f:
    #         json.dump(viznumbers, f)
    #
    #     dists_path = os.path.join(result_dir, 'dists.json')
    #     with open(dists_path, "w") as f:
    #         json.dump(vizdists, f)

    if not debug:
        exp_name = 'llamas'
        evaluator.exp_name = exp_name
        eval_str, _ = evaluator.eval(label='{}'.format(os.path.basename(exp_name)))

    return 0

def testing(db, nnet, result_dir, debug=False, evaluator=None):
    return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug, evaluator=evaluator)