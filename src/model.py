import torch
import torch.nn as nn
import torchvision.models as models

import config

class TeacherModel():
    '''
    Resnet18 model as teacher model.
    '''
    def __init__(self):
        teacher_resnet18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(teacher_resnet18.children())[:-5]).to(config.DEVICE).eval()

class StudentResnetModel():
    '''
    Resnet18 model as student model.
    '''
    def __init__(self):
        student_resnet18 = models.resnet18(pretrained=False)
        self.model = nn.Sequential(*list(student_resnet18.children())[:-5]).to(config.DEVICE).eval()
