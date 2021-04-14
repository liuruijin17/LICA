# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 poly_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold**2, 0.)

        self.poly_weight = poly_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        tgt_ids  = torch.cat([tgt[:, 0] for tgt in targets]).long()

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_boxes"]
        tgt_uppers = torch.cat([tgt[:, 2] for tgt in targets])
        tgt_lowers = torch.cat([tgt[:, 1] for tgt in targets])
        # # Compute the L1 cost between lowers and uppers
        cost_lower = torch.cdist(out_bbox[:, :, 0].view((-1, 1)), tgt_lowers.unsqueeze(-1), p=1)
        cost_upper = torch.cdist(out_bbox[:, :, 1].view((-1, 1)), tgt_uppers.unsqueeze(-1), p=1)

        # # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 3:] for tgt in targets])
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        valid_xs = tgt_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        weights = weights / torch.max(weights)

        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]
        out_polys = out_bbox[:, :, 2:].view((-1, 6))
        tgt_ys = tgt_ys.repeat(out_polys.shape[0], 1, 1)
        tgt_ys = tgt_ys.transpose(0, 2)
        tgt_ys = tgt_ys.transpose(0, 1)

        # Calculate the predicted xs
        out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) ** 2 + out_polys[:, 2] / (tgt_ys - out_polys[:, 1]) + \
                 out_polys[:, 3] + out_polys[:, 4] * tgt_ys - out_polys[:, 5]
        tgt_xs = tgt_xs.repeat(out_polys.shape[0], 1, 1)
        tgt_xs = tgt_xs.transpose(0, 2)
        tgt_xs = tgt_xs.transpose(0, 1)

        cost_polys = torch.stack([torch.sum(torch.abs(tgt_x[valid_x] - out_x[valid_x]), dim=0) for tgt_x, out_x, valid_x in zip(tgt_xs, out_xs, valid_xs)], dim=-1)
        cost_polys = cost_polys * weights

        # # Final cost matrix
        C = self.cost_class * cost_class + self.poly_weight * cost_polys + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper


        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class, set_cost_bbox, set_cost_giou,
                  poly_weight, lower_weight, upper_weight):
    """
    args.set_cost_class: class coefficient in the matching cost
    args.set_cost_bbox: l1 box coefficient in the matching cost
    args.set_cost_giou: giou box coefficient in the mathcing cost
    """
    return HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou,
                            poly_weight=poly_weight, lower_weight=lower_weight, upper_weight=upper_weight)
