from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)     # 按列进行softmax归一化，但比softmax多做一次log
        p_t = F.softmax(y_t/self.T, dim=1)
        # 第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵，以计算kl散度
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
