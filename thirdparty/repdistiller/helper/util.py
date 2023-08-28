from __future__ import print_function

import torch
import numpy as np

def param_dist(model, swa_model, p):        # 返回model和swa_model之前参数距离
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:      # 动态调整优化器中的learning-rate
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:     # 每轮调整时，lr_new为上一轮的0.1倍
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):      # output = model(input),topk=(1,5)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)            # 5
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)      # topk函数返回output中最大的前 maxk 个值和下标，pred中存放下标
        pred = pred.t()                                 # 转置
        correct = pred.eq(target.view(1, -1).expand_as(pred))     # view(1, -1)是构造一个行数为 1，列数自动推断的二维张量；expand_as(pred)是将该张量扩展为pred大小，即将target.view(1, -1)复制len（output[0]）维

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)   # 将所有正确的 1 相加
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    pass
