import cv2
import math
import numpy as np
import torch
import random
import string
import torchvision.transforms as transforms
from copy import deepcopy
from torch.autograd import Variable
import os
from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_, \
    get_affine_transform, affine_transform, fliplr_joints
from .utils import *
from imgaug.augmentables.lines import LineString, LineStringsOnImage


GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def kp_detection(db, k_ind, lane_debug=False):
    data_rng     = system_configs.data_rng
    batch_size   = system_configs.batch_size
    input_size   = db.configs["input_size"] # [h w]
    # border       = db.configs["border"] # 128
    lighting     = db.configs["lighting"] # true
    # rand_crop    = db.configs["rand_crop"] # true
    rand_color   = db.configs["rand_color"] # color
    # rand_scales  = db.configs["rand_scales"] # null
    # allocating memory
    images   = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32) # b, 3, H, W
    masks    = np.zeros((batch_size, 1, input_size[0], input_size[1]), dtype=np.float32)  # b, 1, H, W
    # gt_lanes = np.zeros((batch_size, db.max_lanes, 1 + 2 + 2 * db.max_points), dtype=np.float32) # b 5 115
    gt_lanes = []

    db_size = db.db_inds.size # 3268 | 2782

    for b_ind in range(batch_size):

        if k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading ground truth
        item  = db.detections(db_ind) # all in the raw coordinate
        img   = cv2.imread(item['path'])
        mask  = np.ones((1, img.shape[0], img.shape[1], 1), dtype=np.bool)
        label = item['label']
        # print('label: {}'.format(label))
        transform = True
        if transform:
            line_strings = db.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings, mask = db.transform(image=img, line_strings=line_strings, segmentation_maps=mask)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': db.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            # print(item['categories'])
            label = db._transform_annotation(new_anno, img_wh=(input_size[1], input_size[0]))['label']
        # exit()

        draw_label = deepcopy(label)

        # clip polys
        tgt_ids   = label[:, 0]
        label = label[tgt_ids > 0]

        # make lower the same
        label[:, 1][label[:, 1] < 0] = 1
        label[:, 1][...] = np.min(label[:, 1])

        label = np.stack([label] * batch_size, axis=0)
        gt_lanes.append(torch.from_numpy(label.astype(np.float32)))

        img = (img / 255.).astype(np.float32)
        # if db.normalize:
        #     img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # print(img.shape)
        if rand_color:
            color_jittering_(data_rng, img)
            if lighting:
                lighting_(data_rng, img, 0.1, db.eig_val, db.eig_vec)
        normalize_(img, db.mean, db.std)
        images[b_ind]   = img.transpose((2, 0, 1))
        masks[b_ind]    = np.logical_not(mask[:, :, :, 0])
        # gt_lanes[b_ind] = label

        # debug
        if lane_debug:
            # img = img.permute(1, 2, 0).numpy()
            # if db.normalize:
            #     img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)

            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = (img * 255).astype(np.uint8)
            img_h, img_w, _ = img.shape
            # Draw label
            for i, lane in enumerate(draw_label):
                if lane[0] == 0:  # Skip invalid lanes
                    continue
                lane = lane[3:]  # remove conf, upper and lower positions
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0]
                xs = xs[xs >= 0]

                # draw GT points
                for p in zip(xs, ys):
                    p = (int(p[0] * img_w), int(p[1] * img_h))
                    img = cv2.circle(img, p, 5, color=GT_COLOR, thickness=-1)

                # draw GT lane ID
                cv2.putText(img,
                            str(i), (int(xs[0] * img_w), int(ys[0] * img_h)),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1,
                            color=(0, 255, 0))
            cv2.imshow('sample: {}'.format(item['path']),img)
            cv2.imshow('mask: {}'.format(item['path']), np.logical_not(mask[0, :, :, 0]).astype(np.float))
            cv2.waitKey(0)
            exit()

    images   = torch.from_numpy(images)
    masks    = torch.from_numpy(masks)
    # gt_lanes = torch.from_numpy(gt_lanes)

    return {
               "xs": [images, masks],
               "ys": [images, *gt_lanes]
           }, k_ind


def sample_data(db, k_ind):
    return globals()[system_configs.sampling_function](db, k_ind)


