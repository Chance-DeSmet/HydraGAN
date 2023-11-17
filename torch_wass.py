import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class torch_wasserstein_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(torch_wasserstein_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).mean()                             
        
        return intersection
    
class torch_reid_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(torch_reid_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        loss = torch.sub(inputs, targets)
        loss = torch.abs(loss)
        loss = torch.square(torch.sub(loss, 0.25))
        loss = torch.sum(loss)
        return loss
    
class torch_single_point_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(torch_single_point_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        targets = torch.mean(targets)
        inputs = torch.mean(inputs)
        loss = torch.mean(torch.abs(torch.sub(inputs, targets)))
        return loss
        