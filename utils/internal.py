from collections import defaultdict, deque, OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler
import math


def init_scheduler(lr_scheduler, lr, num_step, optimizer, **kwargs):
    """
    Initialize learning rate scheduler.
    Milestone:
            args.milestone: milestones to decrease the learning rate
                        [milestone_1, milestone_2, ..., milestone_n]
            args.gamma: scale factor
            the learning rate is scaled by gamma when iteration reaches each milestone
    Linear:
            args.lr_e: desired learning rate at the end of training
            the learning rate decreases linearly from lr to lr_e
    Exp:
            args.lr_e: desired learning rate at the end of training
            the learning rate decreases exponentially from lr to lr_e
    Cyclic:
            args.up_ratio: ratio of training steps in the increasing half of a cycle
            args.down_ratio: ratio of training steps in the decreasing half of a cycle
            args.lr_e: Initial learning rate which is the lower boundary in the cycle for each parameter group.
    Static:
            the learning rate remains unchanged during the training
    """
    if lr_scheduler == 'milestones':
        milestones = [milestone * num_step for milestone in kwargs['milestones']]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=kwargs['gamma'])
    elif lr_scheduler == 'exp':
        gamma = math.pow(1 / 100, 1 / num_step)
        lr_scheduler = ExponentialLR(optimizer, gamma)
    elif lr_scheduler == 'cyclic':
        num_circles = 3
        up = int(num_step * 1/3 / num_circles)
        down = int(num_step * 2 / 3 / num_circles)
        lr_scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=lr,
                                step_size_up=up, step_size_down=down, mode='triangular2')
    elif lr_scheduler == 'static':
        def lambda_rule(t):
            return 1.0

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    # TODO ImageNet scheduler
    else:
        raise NameError('Scheduler {0} not found'.format(args.lr_scheduler))
    return lr_scheduler


def init_optimizer(model, optimizer, lr, **kwargs):
    """
    Initialize optimizer:
        SGD: Implements stochastic gradient descent (optionally with momentum).
             args.momentum: momentum factor (default: 0.9)
             args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
        Adam: Implements Adam algorithm.
            args.beta_1, beta_2:
                coefficients used for computing running averages of gradient and its square, default (0.9, 0.99)
            args.eps: term added to the denominator to improve numerical stability (default: 1e-8)
            args.weight_decay: weight decay (L2 penalty) (default: 5e-4)
    """
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr + 1e-8, momentum=kwargs['momentum'],
                                    weight_decay=kwargs['weight_decay'])
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr + 1e-8, betas=(0.9, 0.99),
                                     weight_decay=1e-4)
    else:
        raise NameError('Optimizer {0} not found'.format(optimizer))
    return optimizer


class LLR(_LRScheduler):
    def __init__(self, optimizer, lr_st, lr_ed, steps, last_epoch=-1, verbose=False):
        self.lr_st = lr_st
        self.lr_ed = lr_ed
        self.steps = steps
        self.diff = lr_st - lr_ed
        super(LLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.lr_st * (self.lr_st - (self.last_epoch / self.steps) * self.diff) / self.lr_st
                for group in self.optimizer.param_groups]
