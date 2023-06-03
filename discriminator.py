import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.utils.events import get_event_storage

from fvcore.nn import sigmoid_focal_loss_jit # smooth_l1_loss
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DiscriminatorRes3(nn.Module):
    def __init__(self):
        super(DiscriminatorRes3, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1, 1) ,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, kernel_size=(1, 1), bias = False)
        ).cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_label, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)

        # loss = F.binary_cross_entropy_with_logits(x, domain_label, reduction="mean")
        loss = sigmoid_focal_loss_jit(x, domain_label, alpha=0.25, gamma=2, reduction="mean")
        acc = np.exp(-loss.item())
        # storage = get_event_storage()
        # storage.put_scalar("acc_r3", acc)
        return loss

class DiscriminatorRes4(nn.Module):
    def __init__(self):
        super(DiscriminatorRes4, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(512, 128, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Linear(128, 1, bias = False).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_label, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        # loss = F.binary_cross_entropy_with_logits(x, domain_label, reduction="mean")
        loss = sigmoid_focal_loss_jit(x, domain_label,alpha=0.25,gamma=2,reduction="mean")
        acc = np.exp(-loss.item())
        # storage = get_event_storage()
        # storage.put_scalar("acc_r4", acc)
        return loss #, acc
        
class DiscriminatorRes5(nn.Module):
    def __init__(self):
        super(DiscriminatorRes5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(1024, 256, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).cuda()
        self.reducer2 = nn.Sequential(
            nn.Linear(256, 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(128, 1, bias= False)
        ).cuda() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, domain_label, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)

        # loss = F.binary_cross_entropy_with_logits(x, domain_label, reduction="mean")
        loss = sigmoid_focal_loss_jit(x,domain_label,alpha=0.25,gamma=2,reduction="mean")
        acc = np.exp(-loss.item())
        # storage = get_event_storage()
        # storage.put_scalar("acc_r5", acc)
        
        return loss #, acc
