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

def get_result(img):
    _, contours, _ = cv2.findContours(your_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv.imread('coins.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return unknown

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
    result = np.where((err<0.3), 0, 1)
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
    plt.title('Result Image')
    
    plt.subplot(1, 3, 3)
    plt.imshow(err, cmap='jet')
    plt.title('Predict heatmap')
    plt.colorbar(extend='both')
    
    plt.show()
    # fig.savefig('../result/error_{}'.format(img_path.split('/')[-1]))
    fig.savefig('../result/demo_sample_all.png')
