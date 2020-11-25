import os

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])

def load_image(path = "grid/test/broken/008.png", show_img=False):
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
      features = torch.mean(features, axis=1)
    return features

def plot_features(features, index = 0, mean = False):
    if mean:
        feature = features[0]
    else:
        feature = features[0,index,:,:]
    plt.imshow(feature.cpu().detach().numpy())
    plt.show()

def load_image_(path = "grid/test/broken/008.png"):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(512,512))
    return img

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
    plt.savefig(saved_path)