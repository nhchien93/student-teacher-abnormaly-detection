import os

import utils

import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import config

techer_model_base = models.resnet18(pretrained=True)
teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).to(config.DEVICE).eval()

student_model = torch.load('../trained_model/model_0_grid_2020_11_26_02_21_50.pth', map_location=config.DEVICE)

img_path = '/media/chiennh2/01D6871FDE9C27C0/WorkSpace/Projects/STAD/data/grid/test/bent/000.png'
img = utils.load_image(img_path)

teacher_features = utils.predict(img, teacher_model, True, config.DEVICE)
student_features = utils.predict(img, student_model, True, config.DEVICE)

utils.plot_features(teacher_features, '../result/teacher_{}'.format(img_path.split('/')[-1]), 1, True)
utils.plot_features(student_features, '../result/student_{}'.format(img_path.split('/')[-1]), 1, True)

teacher_feature_np = teacher_features.to(config.DEVICE).detach().numpy()
student_feature_np = student_features.to(config.DEVICE).detach().numpy()

error = np.abs(np.subtract(teacher_feature_np,student_feature_np))
error = np.mean(error, 1)

error_mask = error[0]

plt.imshow(error_mask)
plt.show()
plt.savefig('../result/error_{}'.format(img_path.split('/')[-1]))