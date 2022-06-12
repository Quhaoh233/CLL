# coding = utf-8

import pandas as pd
import numpy as np
import csv
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def import_data(path):
    path_list = []
    name_dict = dict()
    data_dict = dict()
    type_name_dict = dict()

    path_list.extend(glob.glob(path+'/*.csv'))
    for n, npath in enumerate(path_list):
        name_dict[n] = npath[(len(path)+1):(npath.rfind('csv')-1)]
        type_name_dict[n] = npath[(len(path)+1):(len(path)+4)]
        temp = pd.read_csv(npath)
        data = temp['OCCUPANCY'].values.astype('float64')
        data = torch.tensor(data).float()
        data = data.to(device)
        data_dict[n] = data

    information = pd.read_csv('information.csv')
    poi_density_list = information['POI_DENSITY'].values.astype('float64')
    poi_density_list = torch.tensor(poi_density_list)
    poi_density_list = poi_density_list.to(device)
    return name_dict, type_name_dict, data_dict, poi_density_list

class MyData(Dataset):
    def __init__(self, data, seq_length):
        self.sample_list = dict()
        self.label_list = dict()
        for n in range(len(data) - seq_length - seq_length):
            sample = data[n:n+seq_length]
            label = data[n+seq_length+seq_length]
            self.sample_list[n] = sample
            self.label_list[n] = label

    def __len__(self):
        return int(len(self.sample_list))

    def __getitem__(self, item):
        sample = self.sample_list[item]
        sample = torch.reshape(sample, (-1, 1))
        label = self.label_list[item]
        sample = sample.to(device)
        label = label.to(device)
        return sample, label
