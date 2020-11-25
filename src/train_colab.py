from __future__ import print_function, division
import torchvision.models as models
import torch
import torch.nn as nn
import glob
from torchvision import transforms
import cv2 as cv
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

! wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/grid.tar.xz

!tar xf grid.tar.xz

transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((65, 65)),
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

def predict(img, model, c):
    features = model(img.cuda())
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

transform_ = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Resize((65, 65)),
    transforms.ToTensor(),
    ])

def load_image_(path = "grid/test/broken/008.png"):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img,(512,512))
    return img

class Custom_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_data =  "grid/train/good", transform = None ):
        self.path_data = path_data
        self.transform = transform
        self.list_path = glob.glob(self.path_data + "/*.png")

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = load_image_(self.list_path[idx])
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}

        return sample

resnet18 = models.resnet18(pretrained=True)
resnet18_crop = nn.Sequential(*list(resnet18.children())[:-5]).cuda().eval()

import os
custom_Dataset = Custom_Dataset(path_data =  "grid/train/good", transform = transform_)
dataloader = DataLoader(custom_Dataset, batch_size=1, shuffle=True, num_workers=0)


validate_data = Custom_Dataset(path_data =  "grid/test/good", transform = transform_)
dataloader_validate = DataLoader(custom_Dataset, batch_size=1, shuffle=True, num_workers=0)

Epochs = 100

student_model = models.resnet18(pretrained=False)
student_model_crop = nn.Sequential(*list(student_model.children())[:-5]).cuda()

teacher = models.resnet18(pretrained=True)
teacher_crop = nn.Sequential(*list(teacher.children())[:-5]).cuda().eval()

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(student_model_crop.parameters(), lr=0.001)#

training_loss_list = []
validate_loss_list = []

for epoch in range(Epochs):  # loop over the dataset multiple times
    training_loss = 0.0
    validate_loss = 0.0
    for i_batch_train, sample_batched in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        batch_images = sample_batched['image'].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        represent_teacher = teacher_crop(batch_images).detach()
        represent_student = student_model_crop(batch_images)

        loss = criterion(represent_teacher, represent_student)
        loss.backward()
        optimizer.step()

        # print statistics
        training_loss += loss.item() 
    for i_batch_validate, sample_batched in enumerate(dataloader_validate):
        # get the inputs; data is a list of [inputs, labels]
        batch_images = sample_batched['image'].cuda()

        # # zero the parameter gradients
        # optimizer.zero_grad()

        # forward + backward + optimize
        represent_teacher = teacher_crop(batch_images).detach()
        represent_student = student_model_crop(batch_images).detach()

        loss = criterion(represent_teacher, represent_student)
        validate_loss += loss.item() 
    print('%d, training_loss: %.3f, validate_loss: %.3f' %(epoch + 1,  training_loss / i_batch_train, validate_loss/i_batch_validate ))
    training_loss_list.append(training_loss / i_batch_train)
    validate_loss_list.append(validate_loss/i_batch_validate)

print('Finished Training')
if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
torch.save(student_model_crop, "trained_models/grid_1.pth")

import tensorflow as tf

def create_good_gt(img_num, saved_folder, img_shape=[512, 512]):
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    for num in range(img_num):
        if num < 10:
            name = '00{}.png'.format(num)
        if 10 < num < 100:
            name = '0{}.png'.format(num)
        if num > 100:
            name = '{}.png'.format(num)
        img = np.zeros(img_shape, dtype=np.uint8)
        cv.imwrite(os.path.join(saved_folder, name), img)
    return img

s = create_good_gt(21, 'grid/ground_truth/good')

def get_ground_truth(img_folder, show_img=False):
    y_trues = np.array([], dtype=np.int16)
    img_names = sorted(os.listdir(img_folder))
    for img_name in img_names:
        img = cv.imread(os.path.join(img_folder, img_name), cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(128,128))
        img[img > 0] =1
        img[img < 0] =0
        if show_img:
            plt.figure()
            plt.imshow(img)
        y_trues = np.concatenate((y_trues, img.flatten()))
    return y_trues

def predict_img_folder(img_folder, teacher_model, student_model, mean=False, show_img=False):
    y_preds = np.array([])
    img_names = sorted(os.listdir(img_folder))
    for img_name in img_names:
        img = load_image(path = os.path.join(img_folder, img_name))
        features_student = predict(img, student_model, mean)
        # plot_features(features_student, index = 1, mean = mean)

        features_teacher = predict(img, teacher_model, mean)
        # plot_features(features_teacher, index = 1, mean = mean)

        features_teachernp = features_teacher.cpu().detach().numpy()
        features_studentnp = features_student.cpu().detach().numpy()

        error = np.abs(np.subtract(features_teachernp,features_studentnp))
        if mean == False:
            error = np.mean(error, 1)
        error_mask = error[0]
        y_preds = np.concatenate((y_preds, error_mask.flatten()))
        if show_img:
        plt.imshow(error_mask)
        plt.show()
    return y_preds

def evaluate(y_trues, y_preds, show_curve=False):
    '''
    Calculate roc curve area.
    Args:
        y_trues - {1d numpy array}
        y_preds - {1d numpy array}
    Return:
        roc_area - {float} roc curve area.
    '''
    auc = tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
        dtype=None, thresholds=None, multi_label=False, label_weights=None
    )
    auc.update_state(y_trues, y_preds)
    roc_area = auc.result().numpy()
    tpr = (auc.true_positives / (auc.true_positives + auc.false_negatives)).numpy()
    fpr = (auc.false_positives / (auc.false_positives + auc.true_negatives)).numpy()
    if show_curve:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return roc_area

def plot_history(loss_train, loss_val):
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

gt_folder = 'grid/ground_truth/metal_contamination'
test_folder = 'grid/test/metal_contamination'
y_trues = get_ground_truth(gt_folder)
y_preds = predict_img_folder(test_folder, teacher_crop, student_model_crop)
r = evaluate(y_trues, y_preds, show_curve=True)

plot_loss(training_loss_list, validate_loss_list)