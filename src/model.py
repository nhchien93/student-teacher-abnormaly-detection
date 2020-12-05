import os

import torch
import torch.nn as nn
import torchvision.models as models

import config

class TeacherModel():
    '''
    Resnet18 model as teacher model.
    '''
    def __init__(self):
        try:
            teacher_resnet18 = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(teacher_resnet18.children())[:-5]).to(config.DEVICE).eval()
        except:
            teacher = os.path.join(config.RESNET_FOLDER, 'teacher_resnet18.pth')
            print('Can not download resnet18 pre-trained model.')
            print('Use model: {}'.format(teacher))
            self.model = torch.load(teacher, map_location=config.DEVICE)

class BasicBlock(nn.Module):
    '''
    Last group layer for Student model.
    '''
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class StudentModel(nn.Module):

    def __init__(self, num_groups, group=BasicBlock):
        super(StudentModel, self).__init__()
        self.in_planes = 64       
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            self._make_layer(group, 64, num_groups, stride=1)
        )

    def _make_layer(self, group, planes, num_groups, stride):
        strides = [stride] + [1]*(num_groups-1)
        layers = []
        for stride in strides:
            layers.append(group(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
