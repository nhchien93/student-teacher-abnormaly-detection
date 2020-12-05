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
arg.add_argument("-i", "--img", required=True, help="Choose image to predict.")
arg.add_argument("-m", "--model", required=True, help="Choose model to predict.")
arg.add_argument("-t", "--thresh", default=0.3, type=float, help="Choose threshold to predict.")
args = vars(arg.parse_args())

if __name__ == '__main__':
    teacher_model = TeacherModel().model

    model_file_names = os.listdir(config.MODEL_FOLDER)
    student_model_paths = [os.path.join(config.MODEL_FOLDER, name) for name in model_file_names if (args['model'] in name)]
    students_model = [torch.load(student, map_location=config.DEVICE) for student in student_model_paths]

    img_path = '../data/{}'.format(args['img'])
    print(img_path)
    img = utils.load_image(img_path)
    errs = []
    for idx, student_model in enumerate(students_model):
        teacher_features, teacher_feature_mean = utils.predict(img=img, model=teacher_model, mean=True, device=config.DEVICE)
        teacher_features = teacher_features.to(config.DEVICE).detach().numpy()
        
        student_features, student_feature_mean = utils.predict(img=img, model=student_model, mean=True, device=config.DEVICE)
        student_features = student_features.to(config.DEVICE).detach().numpy()
        
        err_temp = utils.get_error(teacher_features, student_features)
        errs.append(err_temp)
    err = sum(errs)/len(errs)
    result = np.where((err<args['thresh']), 0, 1)
    one_indices = [index for index, value in np.ndenumerate(result) if value == 1]

    ori_img = utils.load_image_(img_path, (128, 128))

    ori_img_copy = np.copy(ori_img)
    for idx in one_indices:
        ori_img_copy[idx[0],idx[1]] = 0
    fig = plt.figure(figsize = (8., 4.8))
    plt.subplot(1, 3, 1)
    plt.imshow(ori_img)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(ori_img_copy)
    plt.title('Result Image (p={})'.format(args['thresh']))
    
    plt.subplot(1, 3, 3)
    plt.imshow(err, cmap='jet')
    plt.title('Predict heatmap')
    plt.colorbar(extend='both')
    
    plt.show()
    fig.savefig(os.path.join(config.RESULT_FOLDER, 'demo_result.png'))
