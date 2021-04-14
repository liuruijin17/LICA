# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

from utils.inference import get_max_preds

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)
BLACK = (0, 0, 0)

# SBC_colors = [ORANGE, RED, CYAN, DARK_GREEN, GREEN, BLUE, YELLOW, PURPLE, PINK]
SBC_colors = [ORANGE, ORANGE, ORANGE, RED, RED, RED, CYAN, CYAN, CYAN]

KPS_colors = [DARK_GREEN, DARK_GREEN, YELLOW, YELLOW, PINK]

subclasses = [BLACK, ORANGE, CYAN, PINK, DARK_GREEN, RED]


def save_batch_image_with_joints(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 file_name,
                                 nrow=8,
                                 padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            i = 0
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, KPS_colors[i], 2)
                i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_boxes(batch_image,
                                batch_boxes,
                                batch_labels,
                                file_name,
                                nrow=2,
                                padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_boxes[k]
            labels = batch_labels[k]
            num_box = boxes.shape[0]
            i = 0
            for n in range(num_box):
                lane = boxes[:, 3:][n]
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0] * H
                xs = xs[xs >= 0] * W
                cls = labels[n]
                # print(cls)
                if (cls > 0 and cls < 10):
                    for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                        j_x = x * width + padding + xcoord
                        j_y = y * height + padding + ycoord
                        cv2.circle(ndarr, (int(j_x), int(j_y)), 2, subclasses[cls], 10)
                    i += 1
            # exit()
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_dbs(batch_image,
                              batch_boxes,
                              batch_labels,
                              file_name,
                              nrow=2,
                              padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            pred = batch_boxes[k].cpu().numpy()  # 10 7
            labels = batch_labels[k].cpu().numpy()  # 10
            pred = pred[labels <= 4]  # 4 for llamas

            num_pred = pred.shape[0]
            if num_pred > 0:
                for n, lane in enumerate(pred):
                    # print('pred, lane: {}'.format(lane))
                    cls = labels[n]
                    lane = lane[1:]
                    # lower, upper = lane[0], lane[1]
                    lower = np.minimum(lane[0], lane[1])
                    upper = np.maximum(lane[0], lane[1])
                    upper = 1.
                    lane = lane[2:]
                    ys = np.linspace(lower, upper, num=100)
                    points = np.zeros((len(ys), 2), dtype=np.int32)
                    points[:, 1] = (ys * H).astype(int)
                    # Calculate the predicted xs
                    # points[:, 0] = (np.polyval(lane, ys) * W).astype(int)
                    # points[:, 0] = ((lane[0] / (ys - lane[1]) + lane[2] + lane[3] * ys - lane[4]) * W).astype(int)
                    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[5])
                                    * W).astype(int)
                    points = points[(points[:, 0] > 0) & (points[:, 0] < W)]
                    points[:, 0] += x * width + padding
                    points[:, 1] += y * height + padding
                    for current_point, next_point in zip(points[:-1], points[1:]):
                        cv2.line(ndarr, tuple(current_point), tuple(next_point), color=subclasses[cls], thickness=2)
            k = k + 1
    # exit()

    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    # print(file_name)
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_debug_images(input, target, output,
                      prefix):

    # save_batch_image_with_joints(
    #     input, meta['joints'], meta['joints_vis'],
    #     '{}_gt.jpg'.format(prefix)
    # )
    #
    # save_batch_image_with_joints(
    #     input, joints_pred, meta['joints_vis'],
    #     '{}_pred.jpg'.format(prefix)
    # )
    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )

def save_debug_images_training(input, target, output, prefix):

    # save_batch_image_with_joints(
    #     input, joints, joints_vis,
    #     '{}_gt.jpg'.format(prefix)
    # )

    # save_batch_image_with_joints(
    #     input, joints_pred, joints_vis,
    #     '{}_pred.jpg'.format(prefix)
    # )

    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )

def save_debug_images_joints(input, gt_joints, gt_joints_vis,
                             joints_pred=None, joints_vis_pred=None, prefix=None):

    save_batch_image_with_joints(
        input, gt_joints, gt_joints_vis,
        '{}_gt.jpg'.format(prefix)
    )

    save_batch_image_with_joints(
        input, joints_pred, gt_joints_vis,
        '{}_pred.jpg'.format(prefix)
    )

def save_debug_images_boxes(input, tgt_boxes, tgt_labels,
                            pred_boxes, pred_labels, prefix=None):
    save_batch_image_with_boxes(
        input, tgt_boxes, tgt_labels,
        '{}_gt.jpg'.format(prefix)
    )

    save_batch_image_with_dbs(
        input, pred_boxes, pred_labels,
        '{}_pred.jpg'.format(prefix)
    )

