"""Evaluates lane regression results

Submission format (as json file):
{
   "label_base": {
       'l1': [x0,x1, x2, x3, x4, ..., x399],
       'l0': [x0,x1, x2, x3, x4, ..., x399],
       'r0': [x0,x1, x2, x3, x4, ..., x399],
       'r1': [x0,x1, x2, x3, x4, ..., x399],

   },
   ... (one entry for each label / image within a set
}

Markers from left to right:
l1, l0, car / camera, r0, r1

The main metric for evaluation is mean abs distance in pixels
between regressed markers and reference markers.
"""
import sys
sys.path.append("/media/ruijin/NVME2TB/vision01/work/cvpr2021/e2elsptr")
from db.label_scripts import dataset_constants
from db.common import helper_scripts
from db.label_scripts import spline_creator
import math
import argparse
import json

import numpy


def compare_lane(reference_lane, detected_lane, vertical_cutoff=400):
    """Mean deviation in pixels"""
    assert len(reference_lane) == 717, "Reference lane is too short"
    assert len(detected_lane) >= 717 - vertical_cutoff, "Need at least 417 pixels per lane"

    # Reference lanes go from 0 to 717. If a horizontal entry is not
    # defined, it is stored as -1. We have to filter for that.

    reference_lane = reference_lane[vertical_cutoff:]
    if len(detected_lane) == 717:  # lane regressed across complete image
        detected_lane = detected_lane[vertical_cutoff:]
    elif len(detected_lane) == 417:  # lane regress across part of image that is relevant
        pass
    else:
        raise NotImplementedError(f"Evaluations not implemented for length of detected lane: {len(detected_lane)}")

    reference_lane = [x if x != -1 else float('nan') for x in reference_lane]
    # Results are only allowed to be nan where the labels also are invalid.
    # Just don't add nans to your submissions within the relevant sections of the image.
    assert all([not math.isnan(x) or math.isnan(x_ref) for x, x_ref in zip(detected_lane, reference_lane)]), "NaNs not allowe within lower part of image"

    lane_diff = numpy.subtract(reference_lane, detected_lane)
    abs_lane_diff = numpy.abs(lane_diff)
    mean_abs_diff = numpy.nanmean(abs_lane_diff)
    return mean_abs_diff


def evaluate(eval_file: str, split: str):

    assert eval_file.endswith(".json"), "Detections need to be in json file"
    with open(eval_file) as efh:
        regressions = json.load(efh)

    labels = helper_scripts.get_labels(split=split)
    # print(type(labels))  # list
    # print(len(labels))  # 20844
    # exit()
    results = {"l1": [], "l0": [], "r0": [], "r1": []}
    for label in labels:
        # label /media/ruijin/NVME2TB/vision01/Datasets/LLAMAS/labels/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/1419280849_0800952000.json
        spline_labels = spline_creator.get_horizontal_values_for_four_lanes(label)
        # print(type(spline_labels))  list 4
        # for spline_label in spline_labels:
        #     print(type(spline_label))  list
        #     print(len(spline_label))  717
        assert len(spline_labels) == 4, "Incorrect number of lanes"
        key = helper_scripts.get_label_base(label)
        # print(type(key))  # str images-2014-12-22-12-35-10_mapping_280S_ramps/1419280849_0800952000.json
        regression_lanes = regressions[key]
        for lane, lane_key in zip(spline_labels, ["l1", "l0", "r0", "r1"]):
            # print(lane)
            # print(lane_key)
            # exit()
            result = compare_lane(lane, regression_lanes[lane_key])
            results[lane_key].append(result)

    for key, value in results.items():
        results[key] = numpy.nanmean(value)
    # print(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", help="file to be evaluated", required=True)
    parser.add_argument("--split", help="train, valid, or test", default="valid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.split in ["train", "valid", "test"]
    results = evaluate(eval_file=args.eval_file, split=args.split)
    print(results)
