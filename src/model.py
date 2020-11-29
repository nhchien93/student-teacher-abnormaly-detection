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

class StudentResnetModel():
    '''
    Resnet18 model as student model.
    '''
    def __init__(self):
        try:
            student_resnet18 = models.resnet18(pretrained=False)
            self.model = nn.Sequential(*list(student_resnet18.children())[:-5]).to(config.DEVICE).eval()
        except:
            student = os.path.join(config.RESNET_FOLDER, 'student_resnet18.pth')
            print('Can not download resnet18 structure model.')
            print('Use model: {}'.format(student))
            self.model = torch.load(student, map_location=config.DEVICE)
