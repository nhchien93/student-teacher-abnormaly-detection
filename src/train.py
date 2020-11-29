import os
import argparse

import datetime

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import utils
from model import TeacherModel, StudentResnetModel
from process_data import CustomDataset
import config

arg = argparse.ArgumentParser()
arg.add_argument("-d", "--dataset", default='grid', help="Choose dataset to train.")
arg.add_argument("-s", "--student", default='resnet18', help="Choose dataset to train.")
arg.add_argument("-n", "--num", default=3, type=int, help="Choose number of student to train.")
arg.add_argument("-e", "--epoch", default=100, type=int, help="Choose epoch to train.")
args = vars(arg.parse_args())

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    ])

if __name__ == '__main__':
    train_data_folder = '{}/{}/train/good'.format(config.DATA_FOLDER, args['dataset'])
    val_data_folder = '{}/{}/test/good'.format(config.DATA_FOLDER, args['dataset'])

    train_dataset = CustomDataset(path_data=train_data_folder, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    validate_dataset = CustomDataset(path_data=val_data_folder, transform=transform)
    val_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=True, num_workers=0)

    num_students = args['num']

    teacher_model = TeacherModel().model

    if args['student'] == 'resnet18':
        students_model = [StudentResnetModel().model for i in range(num_students)]
    else:
        students_model = [StudentCustomModel().model for i in range(num_students)]

    criterion = torch.nn.MSELoss()
    optimizer = [optim.Adam(student_model.parameters(), lr=0.01) for student_model in students_model]
    training_loss_list = []
    validate_loss_list = []

    EPOCHS = args['epoch']
    for i, student_model in enumerate(students_model):
        training_loss_temp = []
        validate_loss_temp = []
        for epoch in range(EPOCHS):
            training_loss = 0.0
            validate_loss = 0.0
            for i_batch_train, sample_batched in enumerate(train_dataloader):

                batch_images = sample_batched['image'].to(config.DEVICE)

                optimizer[i].zero_grad()

                represent_teacher = teacher_model(batch_images).detach()
                represent_student = student_model(batch_images)

                loss = criterion(represent_teacher, represent_student)
                loss.backward()
                optimizer[i].step()

                training_loss += loss.item() 
            for i_batch_validate, sample_batched in enumerate(val_dataloader):

                batch_images = sample_batched['image'].to(config.DEVICE)

                represent_teacher = teacher_model(batch_images).detach()
                represent_student = student_model(batch_images).detach()

                loss = criterion(represent_teacher, represent_student)
                validate_loss += loss.item() 
            print('Student: %d - Epoch: %d - training_loss: %.3f - validate_loss: %.3f' %(i, epoch + 1,  training_loss / i_batch_train, validate_loss/i_batch_validate ))
            training_loss_temp.append(training_loss / i_batch_train)
            validate_loss_temp.append(validate_loss/i_batch_validate)
    

        curr_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        model_saved_name = 'model_{}_{}_{}.pth'.format(i, args['dataset'], curr_time)
        torch.save(student_model, os.path.join(config.MODEL_FOLDER, model_saved_name))
        utils.plot_history(training_loss_temp, validate_loss_temp, os.path.join(config.RESULT_FOLDER, 'history_{}_{}_{}.png'.format(i, args['dataset'], curr_time)))
