import os

import utils

import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

techer_model_base = models.resnet18(pretrained=True)
teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).cpu().eval()

student_model = torch.load('../model/grid_1.pth', map_location=device)

img_path = '/media/chiennh2/01D6871FDE9C27C0/WorkSpace/Projects/STAD/data/grid/test/bent/002.png'
img = utils.load_image(img_path)

teacher_features = utils.predict(img, teacher_model, True)
teacher_feature = teacher_features[0]
fig = plt.figure()
plt.imshow(teacher_feature.cpu().detach().numpy())
plt.savefig('../result/teacher_{}'.format(img_path.split('/')[-1]))
del(fig)
student_features = utils.predict(img, student_model, True)
student_feature = student_features[0]
fig = plt.figure()
plt.imshow(student_feature.cpu().detach().numpy())
plt.savefig('../result/student_{}'.format(img_path.split('/')[-1]))
