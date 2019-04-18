from collections import OrderedDict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.legacy.nn.Module import Module
from torch.legacy.nn.utils import clear


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l


def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l


def CrossMapLRN(device, size, alpha, beta, k=1.0):
    lrn = SpatialCrossMapLRN(size, device, alpha, beta, k)
    n = Lambda(lambda x, lrn=lrn: Variable(lrn.forward(x.data).to(device)))
    return n


def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l


class Inception(nn.Module):
    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm,
                 reduceStride=None, padding=True):
        super(Inception, self).__init__()
        #
        self.seq_list = []
        self.outputSize = outputSize

        #
        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernelSize)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1,
                                  (0, 0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernelSize)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1,
                                  (0, 0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1,
                                  (0, 0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)

    def forward(self, input):
        x = input

        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            # print(seq)
            # print(self.outputSize)
            # print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            # print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        # print('target_size:', target_size)

        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))

        output = torch.cat(ys, 1)

        return output


# This is a simple modification of https://github.com/pytorch/pytorch/blob/master/torch/legacy/nn/SpatialCrossMapLRN.py.
class SpatialCrossMapLRN(Module):

    def __init__(self, size, device, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()

        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.scale = None
        self.paddedRatio = None
        self.accumRatio = None
        self.device = device

    def updateOutput(self, input):
        assert input.dim() == 4

        if self.scale is None:
            self.scale = input.new()

        if self.output is None:
            self.output = input.new()

        batchSize = input.size(0)
        channels = input.size(1)
        inputHeight = input.size(2)
        inputWidth = input.size(3)

        self.output = self.output.to(self.device)
        self.scale = self.scale.to(self.device)

        # if input.is_cuda:
        #     self.output = self.output.cuda(self.device)
        #     self.scale = self.scale.cuda(self.device)

        self.output.resize_as_(input)
        self.scale.resize_as_(input)

        # use output storage as temporary buffer
        inputSquare = self.output
        torch.pow(input, 2, out=inputSquare)

        prePad = int((self.size - 1) / 2 + 1)
        prePadCrop = channels if prePad > channels else prePad

        scaleFirst = self.scale.select(1, 0)
        scaleFirst.zero_()
        # compute first feature map normalization
        for c in range(prePadCrop):
            scaleFirst.add_(inputSquare.select(1, c))

        # reuse computations for next feature maps normalization
        # by adding the next feature map and removing the previous
        for c in range(1, channels):
            scalePrevious = self.scale.select(1, c - 1)
            scaleCurrent = self.scale.select(1, c)
            scaleCurrent.copy_(scalePrevious)
            if c < channels - prePad + 1:
                squareNext = inputSquare.select(1, c + prePad - 1)
                scaleCurrent.add_(1, squareNext)

            if c > prePad:
                squarePrevious = inputSquare.select(1, c - prePad)
                scaleCurrent.add_(-1, squarePrevious)

        self.scale.mul_(self.alpha / self.size).add_(self.k)

        torch.pow(self.scale, -self.beta, out=self.output)
        self.output.mul_(input)

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 4

        batchSize = input.size(0)
        channels = input.size(1)
        inputHeight = input.size(2)
        inputWidth = input.size(3)

        if self.paddedRatio is None:
            self.paddedRatio = input.new()
        if self.accumRatio is None:
            self.accumRatio = input.new()
        self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
        self.accumRatio.resize_(inputHeight, inputWidth)

        cacheRatioValue = 2 * self.alpha * self.beta / self.size
        inversePrePad = int(self.size - (self.size - 1) / 2)

        self.gradInput.resize_as_(input)
        torch.pow(self.scale, -self.beta, out=self.gradInput).mul_(gradOutput)

        self.paddedRatio.zero_()
        paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
        for n in range(batchSize):
            torch.mul(gradOutput[n], self.output[n], out=paddedRatioCenter)
            paddedRatioCenter.div_(self.scale[n])
            torch.sum(self.paddedRatio.narrow(0, 0, self.size - 1), 0, out=self.accumRatio)
            for c in range(channels):
                self.accumRatio.add_(self.paddedRatio[c + self.size - 1])
                self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
                self.accumRatio.add_(-1, self.paddedRatio[c])

        return self.gradInput

    def clearState(self):
        clear(self, 'scale', 'paddedRatio', 'accumRatio')
        return super(SpatialCrossMapLRN, self).clearState()
