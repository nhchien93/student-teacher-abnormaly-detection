import os

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULT_FOLDER = '../result'
if not os.path.exists(RESULT_FOLDER):
    os.mkdir(RESULT_FOLDER)

MODEL_FOLDER = '../model'
if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)