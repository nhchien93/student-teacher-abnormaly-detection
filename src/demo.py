import os
import argparse

import utils

import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from model import TeacherModel
import config

arg = argparse.ArgumentParser()
arg.add_argument("-img", "--img", required=True, help="Choose image to predict.")
arg.add_argument("-model", "--model", required=True, help="Choose model to predict.")
args = vars(arg.parse_args())

if __name__ == '__main__':
    teacher_model = TeacherModel().model

    student_model = torch.load('../trained_model/{}'.format(args['model']), map_location=config.DEVICE)

    img_path = '../data/{}'.format(args['img'])
    print(img_path)
    img = utils.load_image(img_path)

    teacher_features, teacher_feature_mean = utils.predict(img=img, model=teacher_model, mean=True, device=config.DEVICE)
    teacher_features = teacher_features.to(config.DEVICE).detach().numpy()
    
    student_features, student_feature_mean = utils.predict(img=img, model=student_model, mean=True, device=config.DEVICE)
    student_features = student_features.to(config.DEVICE).detach().numpy()
    
    err = utils.get_error(teacher_features, student_features)
    
    ori_img = utils.load_image_(img_path, (128, 128))
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ori_img)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(err, cmap='jet')
    plt.title('Predict heatmap')
    plt.colorbar(extend='both')
    
    plt.show()
    # fig.savefig('../result/error_{}'.format(img_path.split('/')[-1]))
    fig.savefig('../result/demo_sample.png')
