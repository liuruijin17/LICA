import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .matcher import build_matcher
from .misc import *

from sample.vis import save_debug_images_boxes

BN_MOMENTUM = 0.1

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SelfAttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(SelfAttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, \
            "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
        self.key_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = self.key_conv
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()  # B C H W
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])   # B C H+padding W+padding
        q_out = self.query_conv(padded_x)  # B C H+padding W+padding
        k_out = self.key_conv(padded_x)  # B C H+padding W+padding
        v_out = self.value_conv(padded_x)  # B C H+padding W+padding
        q_out = q_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        _, _, qw, qh, _, _ = q_out.shape  # B C ? ? kernel_size kernel_size
        _, _, kw, kh, _, _ = k_out.shape  # B C ? ? kernel_size kernel_size
        _, _, vw, vh, _, _ = v_out.shape  # B C ? ? kernel_size kernel_size
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)  # B C ? ? kernel_size kernel_size
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, kw, kh, -1)
        k_out = k_out.transpose(2, 3)
        k_out = k_out.transpose(3, 4)
        k_out = k_out.transpose(4, 5)  # B groups ? ? kernel_size*kernel_size*groups C/groups
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, vw, vh, -1)
        v_out = v_out.transpose(2, 3)
        v_out = v_out.transpose(3, 4)
        v_out = v_out.transpose(4, 5)  # B groups ? ? kernel_size*kernel_size*groups C/groups
        q_out = q_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, qw, qh, -1)
        q_out = q_out.transpose(2, 3)
        q_out = q_out.transpose(3, 4)  # B groups ? ? C/groups kernel_size*kernel_size*groups
        out = torch.matmul(k_out, q_out)   # B groups ? ? kernel_size*kernel_size kernel_size*kernel_size
        out = F.softmax(out, dim=-1)  # B groups ? ? kernel_size*kernel_size kernel_size*kernel_size
        # out = (out - torch.min(out)) / (torch.max(out) - torch.min(out) +1e-8)
        out = torch.matmul(out, v_out)  # B groups ? ? kernel_size*kernel_size C
        # /groups
        out = torch.mean(out, dim=-2)  # B groups ? ? C/groups
        out = out.squeeze(1)  # B ? ? C
        out = out.transpose(-1, -2) # B ? C ?
        out = out.contiguous().transpose(-2, -3)  # B C ? ?
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 freeze=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 kps_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 norm_layer=FrozenBatchNorm2d
                 ):
        super(kp, self).__init__()
        self.flag = flag
        # above all waste not used
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        kernel_sizes = [5, 5, 5, 5]  # if conv, the 5 is invalid
        paddings     = [2, 2, 2, 2]
        attn_groups  = [1, 1, 1, 1]

        self.layer1 = self._make_layer(block[0], res_dims[0], layers[0], stride=res_strides[0],
                                       kernel_size=kernel_sizes[0], padding=paddings[0], attn_groups=attn_groups[0],
                                       embed_shape=1)
        self.layer2 = self._make_layer(block[1], res_dims[1], layers[1], stride=res_strides[1],
                                       kernel_size=kernel_sizes[1], padding=paddings[1], attn_groups=attn_groups[1],
                                       embed_shape=1)
        self.layer3 = self._make_layer(block[2], res_dims[2], layers[2], stride=res_strides[2],
                                       kernel_size=kernel_sizes[2], padding=paddings[2], attn_groups=attn_groups[2],
                                       embed_shape=1)
        self.layer4 = self._make_layer(block[3], res_dims[3], layers[3], stride=res_strides[3],
                                       kernel_size=kernel_sizes[3], padding=paddings[3], attn_groups=attn_groups[3],
                                       embed_shape=1)

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        self.class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, kps_dim - 4, mlp_layers)
        self.scene_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

    def _make_layer(self, block, planes, blocks, stride=1,
                    kernel_size=None, padding=None, attn_groups=None, embed_shape=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            kernel_size=kernel_size, padding=padding, attn_groups=attn_groups,
                            embed_shape=embed_shape))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                kernel_size=kernel_size, padding=padding, attn_groups=attn_groups,
                                embed_shape=embed_shape))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        images = xs[0]
        masks  = xs[1]
        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos    = self.position_embedding(p, pmasks)
        hs, _, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        # output_coord = self.bbox_embed(hs).sigmoid()  # nheads B nqueries num_kps
        output_coord = self.bbox_embed(hs)
        output_scene = self.scene_embed(hs)
        output_scene = torch.mean(output_scene, dim=-2, keepdim=True)
        output_scene = output_scene.repeat(1, 1, output_coord.shape[2], 1)
        output_coord = torch.cat([output_coord[:, :, :, :2], output_scene, output_coord[:, :, :, 2:]], dim=-1)
        out = {'pred_logits': output_class[-1], 'pred_boxes': output_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_coord)
        return out, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class AELoss(nn.Module):
    def __init__(self,
                 db,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(AELoss, self).__init__()
        self.debug_path  = debug_path
        self.db = db
        weight_dict = {'loss_ce': 0, 'loss_polys': 0, 'loss_lowers': 0, 'loss_uppers': 0}
        # set_cost_bbox and set_cost_giou are not used, i will fix it in the next version
        # boxes key represents the polys, lowers and uppers
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=weight_dict['loss_ce'],
                                set_cost_bbox=0,
                                set_cost_giou=0,
                                poly_weight=weight_dict['loss_polys'],
                                lower_weight=weight_dict['loss_lowers'],
                                upper_weight=weight_dict['loss_uppers'])
        losses      = ['labels', 'boxes', 'cardinality']

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets):

        gt_cluxy = [tgt[0] for tgt in targets[1:]]
        loss_dict, indices = self.criterion(outputs, gt_cluxy)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # save = True
        if save:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                gt_viz_inputs = targets[0]
                tgt_labels = [tgt[:, 0].long() for tgt in gt_cluxy]
                pred_labels = outputs['pred_logits'].detach()
                prob = F.softmax(pred_labels, -1)
                scores, pred_labels = prob.max(-1)
                pred_boxes = outputs['pred_boxes'].detach()
                pred_clua3a2a1a0 = torch.cat([scores.unsqueeze(-1), pred_boxes], dim=-1)
                save_debug_images_boxes(gt_viz_inputs,
                                        tgt_boxes=gt_cluxy,
                                        tgt_labels=tgt_labels,
                                        pred_boxes=pred_clua3a2a1a0,
                                        pred_labels=pred_labels,
                                        prefix=save_path)
            # exit()

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)
