import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class BaseClassifier(nn.Module):
    # def __init__(self):
    #     super().__init__()

    def fresh_params(self, bn_wd):
        if bn_wd:
            return self.parameters()
        else:
            return self.named_parameters()


class LinearClassifier(BaseClassifier):
    def __init__(self, nattr, c_in, bn=False, pool='avg', scale=1):
        super().__init__()

        self.pool = pool
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool == 'cat':  # not work
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)])

        if self.pool == 'cat':
            self.logits = nn.Sequential(
                nn.Linear(c_in * 2, nattr),
            )
        else:
            self.logits = nn.Sequential(
                nn.Linear(c_in, nattr),
                nn.BatchNorm1d(nattr) if bn else nn.Identity()
            )

    def forward(self, feature, label=None):
        if self.pool == 'cat':
            feat = []
            for layer in self.pool:
                feat.append(layer(feature).view(feature.size(0), -1))
            feat = torch.cat(feat, dim=1)
        else:
            feat = self.pool(feature).view(feature.size(0), -1)

        x = self.logits(feat)

        return x, feature


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier, bn_wd=True):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd

    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)

    def finetune_params(self):

        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits, feat = self.classifier(feat_map, label)
        return logits, feat


if __name__ == '__main__':
    a = LinearClassifier(32, 32)
    b = a.fresh_params()
