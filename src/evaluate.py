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

from model import TeacherModel
import config

arg = argparse.ArgumentParser()
arg.add_argument("-teacher", "--teacher", default='resnet18', help="Choose teacher model to evaluate.")
arg.add_argument("-dataset", "--dataset", default='grid', help="Choose dataset to evaluate.")
arg.add_argument("-object", "--object", default='all', help="Choose object of dataset to evaluate.")
args = vars(arg.parse_args())

def get_ground_truth(img_folder, show_img=False):
    '''
    Get ground truth image from img_folder.
    '''
    y_trues = np.array([], dtype=np.int16)
    img_names = sorted(os.listdir(img_folder))
    for img_name in img_names:
        print(img_name)
        img = cv.imread(os.path.join(img_folder, img_name), cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(128,128))
        img[img > 0] =1
        img[img < 0] =0
        if show_img:
            plt.figure()
            plt.imshow(img)
        y_trues = np.concatenate((y_trues, img.flatten()))
    return y_trues

def get_ground_truth_dataset(dataset_folder, show_img=False):
    '''
    Get ground truth image from img_folder.
    '''
    folders = os.listdir(dataset_folder)
    for folder in folders:
        if folder == 'good':
            folders.remove(folder)
    y_trues = np.array([])
    for folder in folders:
        folder_path = os.path.join(dataset_folder, folder)
        print(folder_path)
        y_trues_temp = get_ground_truth(folder_path)
        y_trues = np.concatenate((y_trues, y_trues_temp))
    return y_trues

def get_predict(img_folder, teacher_model, students_model, mean=False, show_img=False):
    '''
    Predict image for images in img_folder. Using all student model.
    '''
    y_preds_list = []
    img_names = sorted(os.listdir(img_folder))
    for idx, student_model in enumerate(students_model):
        print('Load student model {}'.format(idx))
        y_preds_temp = np.array([])
        for img_name in img_names:
            print(img_name)
            img = utils.load_image(path = os.path.join(img_folder, img_name))
            
            teacher_features, teacher_feature_mean = utils.predict(img=img, model=teacher_model, mean=True, device=config.DEVICE)
            teacher_features = teacher_features.to(config.DEVICE).detach().numpy()
            
            student_features, student_feature_mean = utils.predict(img=img, model=student_model, mean=True, device=config.DEVICE)
            student_features = student_features.to(config.DEVICE).detach().numpy()
            
            error_mask = utils.get_error(teacher_features, student_features)
            # error_feature_mean = np.abs(np.subtract(teacher_feature, student_feature))
            
            # features_student = utils.predict(img, student_model, mean, device=config.DEVICE)

            # features_teacher = utils.predict(img, teacher_model, mean, device=config.DEVICE)

            # features_teachernp = features_teacher.to(config.DEVICE).detach().numpy()
            # features_studentnp = features_student.to(config.DEVICE).detach().numpy()

            # error = np.abs(np.subtract(features_teachernp,features_studentnp))
            # if mean == False:
            #     error = np.mean(error, 1)
            # error_mask = error_mask[0]
            y_preds_temp = np.concatenate((y_preds_temp, error_mask.flatten()))
            if show_img:
                plt.figure()
                plt.imshow(error_mask)
                plt.show()
                plt.savefig('../result/diff.png')
        y_preds_list.append(y_preds_temp)
    
    y_preds = sum(y_preds_list)/len(y_preds_list)
    return y_preds

def get_predict_dataset(dataset_folder, teacher_model, students_model, mean=False, show_img=False):
    '''
    Predict image for images in img_folder. Using all student model.
    '''
    folders = os.listdir(dataset_folder)

    for folder in folders:
        if folder == 'good':
            folders.remove(folder)

    y_preds = np.array([])
    for folder in folders:
        folder_path = os.path.join(dataset_folder, folder)
        print(folder_path)
        y_preds_mean = get_predict(folder_path, teacher_model, students_model, mean=False, show_img=False)
        y_preds = np.concatenate((y_preds, y_preds_mean))
    return y_preds

def evaluate(y_trues, y_preds, show_curve=False):
    '''
    Calculate roc curve area for each student model.
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
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Student-Teacher Model Evaluate - {} Dataset'.format(args['dataset']))
        plt.legend(loc="lower right")
        plt.show()
        fig.savefig(os.path.join(config.RESULT_FOLDER, 'evaluate_{}_{}.png'.format(args['dataset'], args['object'])))
    return roc_area

if __name__ == '__main__':
    
    if args['object'] == 'all':
        gt_folder = '../data/{}/ground_truth'.format(args['dataset'])
        test_folder = '../data/{}/test'.format(args['dataset'])

        if args['teacher'] == 'resnet18':
            # techer_model_base = models.resnet18(pretrained=True)
            # teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).to(config.DEVICE).eval()
            teacher_model = TeacherModel().model
        else:
            teacher_model_path = os.path.join(config.MODEL_FOLDER, args['teacher'])
            teacher_model = torch.load(teacher_model_path)

        model_file_name = os.listdir(config.MODEL_FOLDER)
        student_model_paths = [os.path.join(config.MODEL_FOLDER, name) for name in model_file_name if (args['dataset'] in name)]
        print(student_model_paths)
        students_model = [torch.load(student_model_path, map_location=config.DEVICE) for student_model_path in student_model_paths]
        y_trues = get_ground_truth_dataset(gt_folder)
        y_preds = get_predict_dataset(test_folder, teacher_model, students_model, show_img=False)
    else:
        gt_folder = '../data/{}/ground_truth/{}'.format(args['dataset'], args['object'])
        test_folder = '../data/{}/test/{}'.format(args['dataset'], args['object'])

        if args['teacher'] == 'resnet18':
            techer_model_base = models.resnet18(pretrained=True)
            teacher_model = nn.Sequential(*list(techer_model_base.children())[:-5]).to(config.DEVICE).eval()
        else:
            teacher_model_path = os.path.join(config.MODEL_FOLDER, args['teacher'])
            teacher_model = torch.load(teacher_model_path)

        model_file_name = os.listdir(config.MODEL_FOLDER)
        student_model_paths = [os.path.join(config.MODEL_FOLDER, name) for name in model_file_name if (args['dataset'] in name)]
        print(student_model_paths)
        students_model = [torch.load(student_model_path, map_location=config.DEVICE) for student_model_path in student_model_paths]
        y_trues = get_ground_truth(gt_folder)
        y_preds = get_predict(test_folder, teacher_model, students_model, show_img=False)

    r = evaluate(y_trues, y_preds, show_curve=True)
    