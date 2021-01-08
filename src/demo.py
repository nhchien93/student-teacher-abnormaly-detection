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
arg.add_argument("-i", "--image", required=True, help="Choose image to predict.")
arg.add_argument("-m", "--model", required=True, help="Choose model to predict.")
arg.add_argument("-t", "--thresh", type=float, required=True, help="Choose threshold to predict.")
args = vars(arg.parse_args())

if __name__ == '__main__':
  teacher_model = TeacherModel().model

  model_file_names = os.listdir(config.MODEL_FOLDER)
  model_file_paths = [os.path.join(config.MODEL_FOLDER, name) for name in model_file_names if (args['model'] in name)]
  print(model_file_paths)
  students_model = [torch.load(student, map_location=config.DEVICE) for student in model_file_paths]

  img_path = f"../data/{args['image']}"
  img = utils.load_image(img_path)

  student_list = []
  teacher_features = utils.predict(img=img, model=teacher_model, mean=False, device=config.DEVICE)
  teacher_features = teacher_features.cpu().detach().numpy()
  print('Shape teacher_features: ', teacher_features.shape)

  for idx, student_model in enumerate(students_model):
    
    student_features = utils.predict(img=img, model=student_model, mean=False, device=config.DEVICE)
    student_features = student_features.cpu().detach().numpy()

    student_list.append(student_features)

  student_out = sum(student_list)/len(student_list)

  student_plot = np.mean(student_out, axis=1)
  teacher_plot = np.mean(teacher_features, axis=1)

  err = utils.get_error(teacher_features, student_out)

  result = np.where((err<args['thresh']), 0, 1)
  one_indices = [index for index, value in np.ndenumerate(result) if value == 1]

  ori_img = utils.load_image_(img_path, (128, 128))
  ori_img_copy = np.copy(ori_img)

  for idx in one_indices:
    ori_img_copy[idx[0], idx[1], :] = (255, 0, 0)

  fig = plt.figure()
  plt.subplot(1, 3, 1)
  plt.imshow(ori_img)
  plt.title('Original Image')
  plt.axis('off')

  # plt.subplot(1, 5, 2)
  # plt.imshow(teacher_plot[0])
  # plt.title('Teacher Feature')
  # plt.axis('off')

  # plt.subplot(1, 5, 3)
  # plt.imshow(student_plot[0])
  # plt.title('Student Feature')
  # plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.imshow(ori_img_copy)
  plt.title('Result (p={})'.format(args['thresh']))
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.imshow(err, cmap='jet')
  plt.title('Predict heatmap')
  plt.colorbar(extend='both')
  plt.axis('off')

  plt.show()
  fig.savefig('../result/demo_result_2.png')
