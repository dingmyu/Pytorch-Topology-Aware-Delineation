import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad, _Loss
from torch.autograd import Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=255, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


def discriminative_loss(input, target, n_objects, max_n_objects, usegpu):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""
    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)
    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)
    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v=0.5, norm=2)
    dist_term = calculate_distance_term(
        cluster_means, n_objects, delta_d=3, norm=2, usegpu=True)
    loss = var_term + dist_term
    return loss


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    #print pred_repeated.size(),pred_repeated.type,gt_expanded.size(),gt_expanded.type
    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        if _n_objects_sample:
            _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
            # n_loc, n_objects, 1
            _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

            _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
            if (max_n_objects - _n_objects_sample) != 0:
                n_fill_objects = max_n_objects - _n_objects_sample
                _fill_sample = torch.zeros(n_fill_objects, n_filters)
                if usegpu:
                    _fill_sample = _fill_sample.cuda()
                _fill_sample = Variable(_fill_sample)
                _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        else:
            _mean_sample = torch.zeros(max_n_objects, n_filters)
            if usegpu:
                _mean_sample = _mean_sample.cuda()
            _mean_sample = Variable(_mean_sample)
        means.append(_mean_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means



def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                        delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

    var_term = 0.0
    for i in range(bs):
        if n_objects[i]:
            _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
            _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

            var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term

def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = n_objects[i]

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        if n_objects[i]:
            _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
            _norm = torch.norm(_mean_sample, norm, 1)
            reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term
