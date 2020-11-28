import os
import json

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import config

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])

def load_image(path = "../data/grid/test/broken/008.png", show_img=False):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(512,512))
    if show_img:
        plt.imshow(img)
        plt.show()
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    return img

def predict(img, model, mean, device):
    features = model(img.to(device))
    if mean:
        features_mean = torch.mean(features, axis=1)
    return features, features_mean

def get_error(teacher_features, student_features):
    error = (np.subtract(teacher_features, student_features))**2
    err_mean = np.mean(error, axis=1)
    err = err_mean/np.max(err_mean)
    err = err[0]
    return err

def plot_features(features, path, index=0, mean=False, show_img=False):
    if mean:
        feature = features[0]
    else:
        feature = features[0,index,:,:]
    if show_img:
        fig = plt.figure()
        plt.imshow(feature.to(config.DEVICE).detach().numpy())
        plt.show()
        fig.savefig(path)

def load_image_(path = "grid/test/broken/008.png", dst_size=(512,512)):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, dst_size)
    return img

def cal_error_mask(teacher_feature, student_feature, threshold):
    teacher_feature = teacher_feature.to(config.DEVICE).detach().numpy()
    student_feature = student_feature.to(config.DEVICE).detach().numpy()
    error = np.abs(np.subtract(teacher_feature, student_feature))
    error[error<threshold] = 0
    return error

def save_history_train(train_loss, val_loss, path):
    data = {}
    data['train_loss'] = train_loss
    data['val_loss'] = val_loss
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def plot_history(loss_train, loss_val, saved_path):
    '''
    Plot training loss and validation loss versus epoches.
    Args:
        loss_train {list} - List of training loss values.
        loss_val {list} - List of validation loss values.
    '''
    fig = plt.figure()
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.title('TEACHER-STUDENT RESNET_18')
    plt.show()
    fig.savefig(saved_path)