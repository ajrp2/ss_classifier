import os
import random

import torch
from torch import nn, from_numpy
from torch.nn import functional as F
from torch.utils import data
import numpy as np
import imageio as io

import utils as ut
import conf
from conf import PARAMS as params


class Conv3DModel(nn.Module):

    def __init__(self):
        super(Conv3DModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(
            in_features=200704,
            out_features=1000
        )

        self.fc2 = nn.Linear(
            in_features=100,
            out_features=conf.N_CLASSES
        )

    def forward(self, input):

        input = self.layer1(input)

        x = self.layer2(input)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        output = self.fc2(x)

        return output


class SomiteStageDataset(data.Dataset):

    def __init__(self, img_folder):
        
        self.img_folder = img_folder

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):

        path = self.img_folder[index]
        image = io.volread(path).astype("float32")
        image = ut._reshape_np(image)
        print(image.shape)
        image = ut.normalize_stack(image, conf.TARGET_DIMS)
        print(image.shape)
        image = np.reshape(image, (1, 128, 128, 100))

        somite_count = str(os.path.dirname(path)[-2:])
        label = np.array(
            conf.LABEL_LOOKUP[str(somite_count)]).astype("int64")   

        return torch.FloatTensor(image), label
