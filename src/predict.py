import os

import utils

import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import config

techer_model_base = models.resnet18(pretrained=True)
teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).cpu().eval()

student_model = torch.load('../trained_model/model_1_carpet_2020_11_25_14_39_58.pth', map_location=config.DEVICE)

img_path = '/media/chiennh2/01D6871FDE9C27C0/WorkSpace/Projects/STAD/data/carpet/test/hole/002.png'
img = utils.load_image(img_path)

teacher_features = utils.predict(img, teacher_model, True, device=config.DEVICE)
teacher_feature = teacher_features[0]
fig = plt.figure()
plt.imshow(teacher_features.cpu().detach().numpy())
plt.savefig('../result/teacher_{}'.format(img_path.split('/')[-1]))
del(fig)
student_features = utils.predict(img, student_model, True, device=config.DEVICE)
student_feature = student_features[0]
fig = plt.figure()
plt.imshow(teacher_features.cpu().detach().numpy())
plt.savefig('../result/student_{}'.format(img_path.split('/')[-1]))


features_teachernp = teacher_features.cpu().detach().numpy()
features_studentnp = student_features.cpu().detach().numpy()
error = np.abs(np.subtract(features_teachernp,features_studentnp))
error = np.mean(error, 1)
error_mask = error[0]
# del (fig)
fig = plt.figure()
plt.imshow(error_mask)
plt.savefig('../result/error_{}'.format(img_path.split('/')[-1]))
