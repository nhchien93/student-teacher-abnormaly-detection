import os
import argparse

import utils

import tensorflow as tf
import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import config

arg = argparse.ArgumentParser()
arg.add_argument("-teacher", "--teacher", default='resnet18', help="Choose teacher model to evaluate.")
arg.add_argument("-student", "--student", required=True, help="Choose student model to evaluate.")
arg.add_argument("-dataset", "--dataset", default='grid', help="Choose dataset to evaluate.")
arg.add_argument("-object", "--object", default='broken', help="Choose object of dataset to evaluate.")
args = vars(arg.parse_args())

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
        img = utils.load_image(path = os.path.join(img_folder, img_name))
        features_student = utils.predict(img, student_model, mean, device=config.DEVICE)

        features_teacher = utils.predict(img, teacher_model, mean, device=config.DEVICE)

        features_teachernp = features_teacher.to(config.DEVICE).detach().numpy()
        features_studentnp = features_student.to(config.DEVICE).detach().numpy()

        error = np.abs(np.subtract(features_teachernp,features_studentnp))
        if mean == False:
            error = np.mean(error, 1)
        error_mask = error[0]
        y_preds = np.concatenate((y_preds, error_mask.flatten()))
        if show_img:
            plt.figure()
            plt.imshow(error_mask)
            plt.show()
            plt.savefig('../result/diff.png')
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
        plt.savefig(os.path.join(config.RESULT_FOLDER, 'evaluate_{}_{}.png'.format(args['student'].split('.')[0], args['object'])))
    return roc_area

if __name__ == '__main__':
    

    gt_folder = '/media/chiennh2/01D6871FDE9C27C0/WorkSpace/Projects/STAD/data/{}/ground_truth/{}'.format(args['dataset'], args['object'])
    test_folder = '/media/chiennh2/01D6871FDE9C27C0/WorkSpace/Projects/STAD/data/{}/test/{}'.format(args['dataset'], args['object'])

    if args['teacher'] == 'resnet18':
        techer_model_base = models.resnet18(pretrained=True)
        teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).to(config.DEVICE).eval()
    else:
        teacher_model_path = os.path.join(config.MODEL_FOLDER, args['teacher'])
        teacher_model = torch.load(teacher_model_path)

    student_model_path = os.path.join(config.MODEL_FOLDER, args['student'])
    student_model = torch.load(student_model_path, map_location=config.DEVICE)

    y_trues = get_ground_truth(gt_folder)
    y_preds = predict_img_folder(test_folder, teacher_model, student_model, show_img=False)

    r = evaluate(y_trues, y_preds, show_curve=True)
    