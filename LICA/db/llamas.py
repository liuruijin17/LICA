import sys
sys.path.insert(0, "data/coco/PythonAPI/")
import json
import os
import numpy as np
import pickle
import cv2
# from tabulate import tabulate
from torchvision.transforms import ToTensor
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from progressbar import progressbar
# import progressbar

from db.detection import DETECTION
from config import system_configs
from db.lane_regression import evaluate
# from db.utils.lane import LaneEval
# from db.utils.metric import eval_json


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)
BLACK = (0, 0, 0)

GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [RED, GREEN, DARK_GREEN, PURPLE, CHOCOLATE, PEACHPUFF, STATEGRAY]
subclasses = [BLACK, ORANGE, CYAN, PINK, DARK_GREEN, RED]

PRED_HIT_COLOR = GREEN
PRED_MISS_COLOR = RED
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

id2str = {1: 'l1',
          2: 'l0',
          3: 'r0',
          4: 'r1'}

class LLAMAS(DETECTION):
    def __init__(self, db_config, split):
        super(LLAMAS, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        # result_dir = system_configs.result_dir
        cache_dir   = system_configs.cache_dir
        max_lanes   = system_configs.max_lanes
        self.metric = 'default'
        inp_h, inp_w = db_config['input_size']

        self._split = split
        self._dataset = {
            "train": ['train'],
            "test": ['test'],
            "val": ['valid'],
            "train+val": ['train', 'valid']
        }[self._split]

        self.root = os.path.join(data_dir, 'LLAMAS')
        if self.root is None:
            raise Exception('Please specify the root directory')
        self.img_w, self.img_h = 1276, 717  # llamas original image resolution
        self.offset = 0
        self.sample_hz = 4
        self.max_points = 0
        self.normalize = True
        self.to_tensor = ToTensor()
        self.aug_chance = 0.9090909090909091
        self._image_file = []

        # self.augmentations = [{'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
        #                       {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
        #                       {'name': 'CropToFixedSize', 'parameters': {'height': 645, 'width': 1148}}]
        self.augmentations = [{'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
                              {'name': 'CropToFixedSize', 'parameters': {'height': 645, 'width': 1148}}]


        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

        # self.anno_files = [os.path.join(self.root, 'labels', path) for path in self._dataset]
        self.anno_files = [os.path.join(self.root, 'labels', path) for path in self._dataset]
        self.png_files = [os.path.join(self.root, 'color_images', path) for path in self._dataset]

        self._data = "llamas"
        # self._mean = np.array([0.573392775, 0.58143508, 0.573799285], dtype=np.float32) # [0.59007017 0.59914317 0.58877597]  # [0.55671538 0.56372699 0.55888226]
        # self._std = np.array([0.01633231, 0.01760496, 0.01697805], dtype=np.float32) #
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self._cat_ids = [
            0
        ]  # 0 car
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._coco_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "llamas_{}.pkl".format(self._dataset))


        if self.augmentations is not None:
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in self.augmentations]  # add augmentation

        transformations = iaa.Sequential([Resize({'height': inp_h, 'width': inp_w})])
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=self.aug_chance), transformations])

        self._load_data()

        self._db_inds = np.arange(len(self._image_ids))

    def _load_data(self, debug_lane=False):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            if self._split != 'test':
                self._extract_data()
                self._transform_annotations()
            else:
                self._extract_test_data()
                self._transform_annotations()

            if debug_lane:
                pass
            else:
                with open(self._cache_file, "wb") as f:
                    pickle.dump([self._annotations,
                                 self._image_ids,
                                 self._image_file,
                                 self.max_lanes,
                                 self.max_points], f)
        else:
            with open(self._cache_file, "rb") as f:
                (self._annotations,
                 self._image_ids,
                 self._image_file,
                 self.max_lanes,
                 self.max_points) = pickle.load(f)
        print("max_points: {}".format(self.max_points))
        print("max_lanes: {}".format(self.max_lanes))

        if debug_lane:
            print('Go debug raw data')
            for i in range(len(self._image_ids)):
                img = self.draw_annotation(i)
                cv2.imshow('sample: {}'.format(i), img)
                cv2.waitKey(0)
                if i > 5:
                    break
            exit()

    def get_json_paths(self):
        json_paths = []
        for anno_file in self.anno_files:
            for root, dirs, files in os.walk(anno_file):
                for file in files:
                    if file.endswith(".json"):
                        json_paths.append(os.path.join(root, file))
        return json_paths

    def get_png_paths(self):
        png_paths = []
        for png_file in self.png_files:
            for root, dirs, files in os.walk(png_file):
                for file in files:
                    if file.endswith(".png"):
                        png_paths.append(os.path.join(root, file))
        return png_paths

    def get_img_path(self, json_path):
        # /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext
        base_name = '/'.join(json_path.split('/')[-3:])
        image_path = os.path.join('color_images', base_name.replace('.json', '_color_rect.png'))
        return image_path

    def _extract_test_data(self):
        image_id = 0
        self._old_annotations = {}
        self.annotations = []
        self.max_points = 1
        self.max_lanes = 4

        print("Searching testing images...")
        png_paths = self.get_png_paths()
        print('{} png images found.'.format(len(png_paths)))
        for png_path in progressbar(png_paths):
            # fake_lanes = [list(range(1)) for i in range(4)]
            relative_path = self.get_img_path(png_path)
            fake_lanes = [[(x, y) for x, y in zip(list(range(1),),
                                                  list(range(1)))] for i in range(4)]
            self._image_file.append(png_path)
            self._image_ids.append(image_id)
            self._old_annotations[image_id] = {
                'path': png_path,
                'aug': False,
                'lanes': fake_lanes,
                'relative_path': relative_path,
            }
            image_id += 1


    def _extract_data(self):

        image_id  = 0
        self._old_annotations = {}
        self.annotations = []
        self.max_points = 0
        self.max_lanes = 0

        print("Searching annotation files...")
        json_paths = self.get_json_paths()
        print('{} annotations found.'.format(len(json_paths)))
        for json_path in progressbar(json_paths):
            lanes = get_horizontal_values_for_four_lanes(json_path)
            categories = [1, 2, 3, 4]
            # init_lanes = lanes.copy()
            lanes = [[(x, y) for x, y in zip(
                lane[::self.sample_hz], list(range(self.img_h))[::self.sample_hz]) if x >= 0] for lane in lanes]
            remain_categories = []
            for lane, category in zip(lanes, categories):
                if len(lane) > 0:
                    remain_categories.append(category)
            lanes = [lane for lane in lanes if len(lane) > 0]
            relative_path = self.get_img_path(json_path)
            img_path = os.path.join(self.root, relative_path)
            # self.max_points = max(self.max_points, max(len(lane for lane in lanes)))
            self.max_points = max(self.max_points, max([len(lane) for lane in lanes]))
            self.max_lanes = max(self.max_lanes, len(lanes))
            self._image_file.append(img_path)
            self._image_ids.append(image_id)
            self._old_annotations[image_id] = {
                'path': img_path,
                'lanes': lanes,
                'aug': False,
                'relative_path': relative_path,
                'categories': remain_categories
            }
            image_id += 1


    def _get_img_heigth(self, path):
        return self.img_h

    def _get_img_width(self, path):
        return self.img_w

    def _transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self._get_img_heigth(anno['path'])
            img_w = self._get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes = zip(old_lanes, categories)
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'],
            'label': lanes,
            'old_anno': anno,
            'categories': [cat for _, cat in old_lanes]
        }

        return new_anno

    def _transform_annotations(self):
        print('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_annotation(old_anno)

    def detections(self, ind):
        image_id  = self._image_ids[ind]
        item      = self._annotations[image_id]
        return item

    def __len__(self):
        return len(self._annotations)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def class_name(self, cid):
        cat_id = self._classes[cid]
        return cat_id

    def get_metrics(self, lanes, idx):
        # Placeholders
        return [1] * len(lanes), [1] * len(lanes), None

    def __getitem__(self, idx, transform=False):

        item = self._annotations[idx]
        img = cv2.imread(item['path'])
        label = item['label']
        if transform:
            line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings = LineStringsOnImage(line_strings, shape=img.shape)
            img, line_strings = self.transform(image=img, line_strings=line_strings)
            line_strings.clip_out_of_image_()
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            new_anno['categories'] = item['categories']
            label = self._transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes


    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None,
                        draw_gt=True):
        if img is None:
            img, label, _ = self.__getitem__(idx, transform=True)
            # Tensor to opencv image
            img = img.permute(1, 2, 0).numpy()
            # Unnormalize
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            # img = img.transpose(1, 2, 0)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            _, label, _ = self.__getitem__(idx)
            img = (img * 255).astype(np.uint8)

        img_h, img_w, _ = img.shape

        if draw_gt:
            # Draw label
            for i, lane in enumerate(label):
                if lane[0] == 0:  # Skip invalid lanes
                    continue
                sbc = lane[0]
                lane = lane[3:]  # remove conf, upper and lower positions
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:]
                ys = ys[xs >= 0]
                xs = xs[xs >= 0]

                # draw GT points
                for p in zip(xs, ys):
                    p = (int(p[0] * img_w), int(p[1] * img_h))
                    img = cv2.circle(img, p, 5, color=subclasses[int(sbc)], thickness=-1)

                # draw GT lane ID
                cv2.putText(img,
                            id2str[int(sbc)], (int(xs[len(xs) * 2 // 3] * img_w), int(ys[len(ys) * 2 // 3] * img_h)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=subclasses[int(sbc)],
                            thickness=3)

        if pred is None:
            return img

        # Draw predictions
        # pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        pred = pred[pred[:, 0].astype(int) >= 1]
        matches, accs, _ = self.get_metrics(pred, idx)
        overlay = img.copy()
        colors = [(102, 106, 255),
                  (115, 181, 34),
                  (255, 255, 0),
                  (120, 190, 250),]
        lane_points = {}
        for i, lane in enumerate(pred):
            # if matches[i]:
            #     # color = colors[i]
            #     color = PRED_HIT_COLOR
            # else:
            #     color = PRED_MISS_COLOR
            # print(lane.shape)
            # lane = lane[1:]  # remove conf
            sbc = lane[0]
            color = subclasses[int(sbc)]
            # print(int(sbc))
            lane = lane[2:]  # remove label conf
            # print(lane)
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions
            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            # points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            # points[:, 0] = ((lane[0] / (ys - lane[1]) + lane[2] + lane[3] * ys - lane[4]) * img_w).astype(int)
            points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                             lane[5]) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]
            lane_points[id2str[int(sbc)]] = [list(points[:, 0].astype(float)), list(points[:, 1].astype(float))]

            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=5)

            # # draw class icon
            # if cls_pred is not None and len(points) > 0:
            #     class_icon = self.get_class_icon(cls_pred[i])
            #     class_icon = cv2.resize(class_icon, (32, 32))
            #     mid = tuple(points[len(points) // 2] - 60)
            #     x, y = mid
            #
            #     img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon
            # print(type(points), points.shape)
            # exit()

            # # draw lane ID
            if len(points) > 0:
                cv2.putText(img, id2str[int(sbc)], tuple(points[len(points)//3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color,
                            thickness=3)

            # # draw lane accuracy
            # if len(points) > 0:
            #     cv2.putText(img,
            #                 '{:.2f}'.format(accs[i] * 100),
            #                 tuple(points[len(points) // 2] - 30),
            #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale=1,
            #                 color=color,
            #                 thickness=3)
        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)
        # exit()

        return img, lane_points

    def get_final_lane(self, xxs, subclass, temp=None):
        y_samples = list(range(self.img_h))
        ys = np.array(y_samples)

        if len(xxs) == 0:
            # if temp:
            #     return subclass, temp
            # else:
                if subclass == 'l1':
                    lane_pred = (ys - 717) / -717 * 319
                    return subclass, [[-1., list(lane_pred)]]
                elif subclass == 'l0':
                    lane_pred = (ys - 717) / -717 * 319 + 319
                    return subclass, [[-1, list(lane_pred)]]
                elif subclass == 'r0':
                    lane_pred = ys / 717 * 319 + 319 * 2
                    return subclass, [[-1., list(lane_pred)]]
                elif subclass == 'r1':
                    lane_pred = ys / 717 * 319 + 319 * 3
                    return subclass, [[-1, list(lane_pred)]]

        elif len(xxs) == 1:
            return subclass, xxs
        elif len(xxs) > 1:
            max_score = 0
            cp_lane = None
            for score, cad_lane in xxs:
                if score > max_score:
                    max_score = score
                    cp_lane = cad_lane
            return subclass, [[max_score, cp_lane]]

    def pred2lanes(self, path, pred, y_samples):
        ys = np.array(y_samples) / self.img_h
        lanes = {}
        l1s = []
        l0s = []
        r0s = []
        r1s = []
        for lane in pred:
            if lane[0] == 0:
                continue
            lanepoly = lane[4:]
            lane_pred = (lanepoly[0] / (ys - lanepoly[1]) ** 2 + lanepoly[2] / (ys - lanepoly[1]) + lanepoly[3] +
                         lanepoly[4] * ys - lanepoly[5]) * self.img_w

            if int(lane[0]) == 1:
                score = lane[1]
                l1s.append([score, list(lane_pred)])
            elif int(lane[0]) == 2:
                score = lane[1]
                l0s.append([score, list(lane_pred)])
            elif int(lane[0]) == 3:
                score = lane[1]
                r0s.append([score, list(lane_pred)])
            elif int(lane[0]) == 4:
                score = lane[1]
                r1s.append([score, list(lane_pred)])
            else:
                raise ValueError('invalid lane[0]: {}'.format(lane[0]))

        l1, l1_lane = self.get_final_lane(l1s, subclass='l1')
        # print(l1, len(l1_lane[0][1]))
        l0, l0_lane = self.get_final_lane(l0s, subclass='l0')
        # print(l0, len(l0_lane[0][1]))
        r0, r0_lane = self.get_final_lane(r0s, subclass='r0')
        # print(r0, len(r0_lane[0][1]))
        r1, r1_lane = self.get_final_lane(r1s, subclass='r1')
        # print(r1, len(r1_lane[0][1]))
        lanes[l1] = l1_lane[0][1]
        lanes[l0] = l0_lane[0][1]
        lanes[r0] = r0_lane[0][1]
        lanes[r1] = r1_lane[0][1]
        return lanes

    def pred2llamasformat(self, idx, pred, runtime, output):
        runtime *= 1000.  # s to ms
        old_anno = self._annotations[idx]['old_anno']
        if self._split == 'test':
            rel_path = old_anno['relative_path'][18:-15] + '.json'
        else:
            rel_path = old_anno['relative_path'][19:-15] + '.json'
        h_samples = list(range(self.img_h))
        lanes = self.pred2lanes(rel_path, pred, h_samples)
        output[rel_path] = lanes
        return output

    def save_llamas_predictions(self, predictions, runtimes, filename):
        self.l1_temp = []
        self.l0_temp = []
        self.r0_temp = []
        self.r1_temp = []
        output = {}
        for idx in range(len(predictions)):
            output = self.pred2llamasformat(idx, predictions[idx], runtimes[idx], output)
        with open(filename, 'w') as output_file:
            json.dump(output, output_file)
        print('Evaluating...')
        if not self._dataset == 'test':
            result = evaluate.evaluate(eval_file=filename, split=self._dataset[0])
            return result
        else:
            return {'l1': -1, 'l0': -1, 'r0': -1, 'r1': -1}

    def eval(self, exp_dir, predictions, runtimes, label=None, only_metrics=False):
        # Placeholder
        pred_filename = 'llamas_{}_predictions_{}.json'.format(self.split, label)
        pred_filename = os.path.join(exp_dir, pred_filename)
        result = self.save_llamas_predictions(predictions, runtimes, pred_filename)
        # result = 'mean_dis: {}'.format(mean_dist)
        str_result = 'mean_dis: {}'.format(result)
        if not only_metrics:
            filename = 'llamas_{}_eval_result_{}.json'.format(self.split, label)
            with open(os.path.join(exp_dir, filename), 'w') as out_file:
                json.dump(str_result, out_file)
        return result, None

    def get_final_lane_with_kappa(self, xxs, subclass):
        y_samples = list(range(self.img_h))
        ys = np.array(y_samples)
        if len(xxs) == 0:
            # print('subclass: {}'.format(subclass))
            # if subclass == 'l1':
            #     return subclass, [[-1., list(np.arange(self.img_h).astype(np.float64))]]
            # elif subclass == 'r1':
            #     # return subclass, [[-1, list(np.ones(self.img_h) * self.img_w)]]
            #     return subclass, [[-1, list(np.arange(self.img_h - 1 - self.img_w, self.img_h - 1)[::-1].astype(np.float64))]]
            # elif subclass == 'l0':
            #     return subclass, [[-1., list(np.arange(self.img_h).astype(np.float64))]]
            # elif subclass == 'r0':
            #     return subclass, [[-1, list(np.arange(self.img_h - 1 - self.img_w, self.img_h - 1)[::-1].astype(np.float64))]]
            if subclass == 'l1':
                lane_pred = (ys - 717) / -717 * 319
                return subclass, [[-1., list(lane_pred), [-10.] * len(lane_pred)]]
            elif subclass == 'l0':
                lane_pred = (ys - 717) / -717 * 319 + 319
                return subclass, [[-1, list(lane_pred), [-10.] * len(lane_pred)]]
            elif subclass == 'r0':
                lane_pred = ys / 717 * 319 + 319 * 2
                return subclass, [[-1., list(lane_pred), [-10.] * len(lane_pred)]]
            elif subclass == 'r1':
                lane_pred = ys / 717 * 319 + 319 * 3
                return subclass, [[-1, list(lane_pred), [-10.] * len(lane_pred)]]

        elif len(xxs) == 1:
            return subclass, xxs
        elif len(xxs) > 1:
            max_score = 0
            cp_lane = None
            cp_kappa = None
            for score, cad_lane, cad_kappa in xxs:
                if score > max_score:
                    max_score = score
                    cp_lane = cad_lane
                    cp_kappa = cad_kappa
            return subclass, [[max_score, cp_lane, cp_kappa]]

    def pred2laneskappas(self, path, pred, y_samples):

        ys = np.array(y_samples) / self.img_h
        lanes = {}
        l1s = []
        l0s = []
        r0s = []
        r1s = []
        for lane in pred:
            if lane[0] == 0:
                continue

            bdlane = lane[2:]  # remove label conf
            lower, upper = bdlane[0], bdlane[1]
            lanepoly = lane[4:]
            lane_pred = (lanepoly[0] / (ys - lanepoly[1]) ** 2 + lanepoly[2] / (ys - lanepoly[1]) + lanepoly[3] +
                         lanepoly[4] * ys - lanepoly[5]) * self.img_w
            lane_pred[(ys < lower) | (ys > upper)] = -2
            # lanes.append(list(lane_pred))
            diff1 = -2 * lanepoly[0] / (ys - lanepoly[1]) ** 3 \
                    - lanepoly[2] / (ys - lanepoly[1]) ** 2 \
                    + lanepoly[4]
            diff2 = 6 * lanepoly[0] / (ys - lanepoly[1]) ** 4 \
                    + 2 * lanepoly[2] / (ys - lanepoly[1]) ** 3
            kappa = np.abs(diff2) / (1 + diff1 ** 2) ** 1.5
            kappa += 1

            if int(lane[0]) == 1:
                score = lane[1]
                l1s.append([score, list(lane_pred), kappa])
            elif int(lane[0]) == 2:
                score = lane[1]
                l0s.append([score, list(lane_pred), kappa])
            elif int(lane[0]) == 3:
                score = lane[1]
                r0s.append([score, list(lane_pred), kappa])
            elif int(lane[0]) == 4:
                score = lane[1]
                r1s.append([score, list(lane_pred), kappa])
            else:
                raise ValueError('invalid lane[0]: {}'.format(lane[0]))

        l1, l1_lane = self.get_final_lane_with_kappa(l1s, subclass='l1')
        # print(l1, len(l1_lane[0][1]))
        l0, l0_lane = self.get_final_lane_with_kappa(l0s, subclass='l0')
        # print(l0, len(l0_lane[0][1]))
        r0, r0_lane = self.get_final_lane_with_kappa(r0s, subclass='r0')
        # print(r0, len(r0_lane[0][1]))
        r1, r1_lane = self.get_final_lane_with_kappa(r1s, subclass='r1')
        # print(r1, len(r1_lane[0][1]))
        lanes[l1] = l1_lane[0][1]
        lanes[l0] = l0_lane[0][1]
        lanes[r0] = r0_lane[0][1]
        lanes[r1] = r1_lane[0][1]
        scores = [l1_lane[0][0], l0_lane[0][0], r0_lane[0][0], r1_lane[0][0]]
        kappas = [l1_lane[0][-1], l0_lane[0][-1], r0_lane[0][-1], r1_lane[0][-1]]

        return lanes, scores, kappas

    def get_corresponding_kappas_for_gts(self, idx, pred):

        old_anno = self._annotations[idx]['old_anno']
        if self._dataset == 'test':
            rel_path = old_anno['relative_path'][18:-15] + '.json'
        else:
            rel_path = old_anno['relative_path'][19:-15] + '.json'
        h_samples = list(range(self.img_h))
        lanes, scores, kappas = self.pred2laneskappas(rel_path, pred, h_samples)
        json_path = os.path.join('/media/ruijin/NVME2TB/vision01/Datasets/LLAMAS/labels/valid', rel_path)
        gtlanes = get_horizontal_values_for_four_lanes(json_path)

        return old_anno['relative_path'], lanes, scores, gtlanes, kappas
        # return corKappas, org_anno['org_path'], corDists, pred, corIDs, corAccs, org_anno['org_lanes']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _extend_lane(lane, projection_matrix):
    """Extends marker closest to the camera

    Adds an extra marker that reaches the end of the image

    Parameters
    ----------
    lane : iterable of markers
    projection_matrix : 3x3 projection matrix
    """
    # Unfortunately, we did not store markers beyond the image plane. That hurts us now
    # z is the orthongal distance to the car. It's good enough

    # The markers are automatically detected, mapped, and labeled. There exist faulty ones,
    # e.g., horizontal markers which need to be filtered
    filtered_markers = filter(
        lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and x['pixel_start']['x'] != x['pixel_end']['x']),
        lane['markers'])
    # might be the first marker in the list but not guaranteed
    closest_marker = min(filtered_markers, key=lambda x: x['world_start']['z'])

    if closest_marker['world_start']['z'] < 0:  # This one likely equals "if False"
        return lane

    # World marker extension approximation
    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    zero_x = closest_marker['world_start']['x'] - (closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (closest_marker['world_start']['z'] - 1) * y_gradient

    # Pixel marker extension approximation
    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) /\
        (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) /\
        (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    pixel_zero_x = closest_marker['pixel_start']['x'] + (716 - closest_marker['pixel_start']['y']) * pixel_x_gradient
    if pixel_zero_x < 0:
        left_y = closest_marker['pixel_start']['y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        right_y = closest_marker['pixel_start']['y'] + (1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {
        'lane_marker_id': 'FAKE',
        'world_end': {
            'x': closest_marker['world_start']['x'],
            'y': closest_marker['world_start']['y'],
            'z': closest_marker['world_start']['z']
        },
        'world_start': {
            'x': zero_x,
            'y': zero_y,
            'z': 1
        },
        'pixel_end': {
            'x': closest_marker['pixel_start']['x'],
            'y': closest_marker['pixel_start']['y']
        },
        'pixel_start': {
            'x': ir(new_pixel_point[0]),
            'y': ir(new_pixel_point[1])
        }
    }
    lane['markers'].insert(0, new_marker)

    return lane


class SplineCreator():
    """
    For each lane divder
      - all lines are projected
      - linearly interpolated to limit oscillations
      - interpolated by a spline
      - subsampled to receive individual pixel values

    The spline creation can be optimized!
      - Better spline parameters
      - Extend lowest marker to reach bottom of image would also help
      - Extending last marker may in some cases be interesting too
    Any help is welcome.

    Call create_all_points and get the points in self.sampled_points
    It has an x coordinate for each value for each lane

    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}  # <--- the interesting part
        self.debug_image = np.zeros((717, 1276, 3), dtype=np.uint8)

    def _sample_points(self, lane, ypp=5, between_markers=True):
        """ Markers are given by start and endpoint. This one adds extra points
        which need to be considered for the interpolation. Otherwise the spline
        could arbitrarily oscillate between start and end of the individual markers

        Parameters
        ----------
        lane: polyline, in theory but there are artifacts which lead to inconsistencies
              in ordering. There may be parallel lines. The lines may be dashed. It's messy.
        ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels
        between_markers : bool, interpolates inbetween dashes

        Notes
        -----
        Especially, adding points in the lower parts of the image (high y-values) because
        the start and end points are too sparse.
        Removing upper lane markers that have starting and end points mapped into the same pixel.
        """

        # Collect all x values from all markers along a given line. There may be multiple
        # intersecting markers, i.e., multiple entries for some y values
        x_values = [[] for i in range(717)]
        for marker in lane['markers']:
            x_values[marker['pixel_start']['y']].append(marker['pixel_start']['x'])

            height = marker['pixel_start']['y'] - marker['pixel_end']['y']
            if height > 2:
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(height)
                for i in range(height + 1):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i
                    x_values[ir(y)].append(ir(x))

        # Calculate average x values for each y value
        for y, xs in enumerate(x_values):
            if not xs:
                x_values[y] = -1
            else:
                x_values[y] = sum(xs) / float(len(xs))

        # In the following, we will only interpolate between markers if needed
        if not between_markers:
            return x_values  # TODO ypp

        # # interpolate between markers
        current_y = 0
        while x_values[current_y] == -1:  # skip missing first entries
            current_y += 1

        # Also possible using numpy.interp when accounting for beginning and end
        next_set_y = 0
        try:
            while current_y < 717:
                if x_values[current_y] != -1:  # set. Nothing to be done
                    current_y += 1
                    continue

                # Finds target x value for interpolation
                while next_set_y <= current_y or x_values[next_set_y] == -1:
                    next_set_y += 1
                    if next_set_y >= 717:
                        raise StopIteration

                x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) /\
                    (next_set_y - current_y + 1)
                current_y += 1

        except StopIteration:
            pass  # Done with lane

        return x_values

    def _lane_points_fit(self, lane):
        # TODO name and docstring
        """ Fits spline in image space for the markers of a single lane (side)

        Parameters
        ----------
        lane: dict as specified in label

        Returns
        -------
        Pixel level values for curve along the y-axis

        Notes
        -----
        This one can be drastically improved. Probably fairly easy as well.
        """
        # NOTE all variable names represent image coordinates, interpolation coordinates are swapped!
        lane = _extend_lane(lane, self.json_content['projection_matrix'])
        sampled_points = self._sample_points(lane, ypp=1)
        self.sampled_points[lane['lane_id']] = sampled_points

        return sampled_points

    def create_all_points(self, ):
        """ Creates splines for given label """
        for lane in self.lanes:
            self._lane_points_fit(lane)


def get_horizontal_values_for_four_lanes(json_path):
    """ Gets an x value for every y coordinate for l1, l0, r0, r1

    This allows to easily train a direct curve approximation. For each value along
    the y-axis, the respective x-values can be compared, e.g. squared distance.
    Missing values are filled with -1. Missing values are values missing from the spline.
    There is no extrapolation to the image start/end (yet).
    But values are interpolated between markers. Space between dashed markers is not missing.

    Parameters
    ----------
    json_path: str
               path to label-file

    Returns
    -------
    List of [l1, l0, r0, r1], each of which represents a list of ints the length of
    the number of vertical pixels of the image

    Notes
    -----
    The points are currently based on the splines. The splines are interpolated based on the
    segmentation values. The spline interpolation has lots of room for improvement, e.g.
    the lines could be interpolated in 3D, a better approach to spline interpolation could
    be used, there is barely any error checking, sometimes the splines oscillate too much.
    This was used for a quick poly-line regression training only.
    """

    sc = SplineCreator(json_path)
    sc.create_all_points()

    l1 = sc.sampled_points.get('l1', [-1] * 717)
    l0 = sc.sampled_points.get('l0', [-1] * 717)
    r0 = sc.sampled_points.get('r0', [-1] * 717)
    r1 = sc.sampled_points.get('r1', [-1] * 717)

    lanes = [l1, l0, r0, r1]
    return lanes



def _filter_lanes_by_size(label, min_height=40):
    """ May need some tuning """
    filtered_lanes = []
    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
        if (lane_end - lane_start) < min_height:
            continue
        filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _filter_few_markers(label, min_markers=2):
    """Filter lines that consist of only few markers"""
    filtered_lanes = []
    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _fix_lane_names(label):
    """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

    # Create mapping
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]
    for key in sorted(lane_ids):
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1
        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1
    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]


def read_json(json_path, min_lane_height=20):
    """ Reads and cleans label file information by path"""
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    _filter_lanes_by_size(label_content, min_height=min_lane_height)
    _filter_few_markers(label_content, min_markers=2)
    _fix_lane_names(label_content)

    content = {'projection_matrix': label_content['projection_matrix'], 'lanes': label_content['lanes']}

    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
    return content


def ir(some_value):
    """ Rounds and casts to int
    Useful for pixel values that cannot be floats
    Parameters
    ----------
    some_value : float
                 numeric value
    Returns
    --------
    Rounded integer
    Raises
    ------
    ValueError for non scalar types
    """
    return int(round(some_value))


# End code under the previous license




