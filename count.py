# coding = utf=8

import pandas as pd
import numpy as np
import csv
import torch
import matplotlib.pyplot as plt
import glob

# input data
def import_data(path):
    path_list = []
    data_dict = dict()

    path_list.extend(glob.glob(path+'/*.csv'))
    for n, npath in enumerate(path_list):
        temp = pd.read_csv(npath)
        data = temp['OCCUPANCY'].values.astype('float64')
        data_dict[n] = data
    return data_dict

time_series_features = pd.read_csv('time_series_features.csv')
MAXT = time_series_features['MAXT'].values.astype('float64')
MINT = time_series_features['MINT'].values.astype('float64')
CLIMATE = time_series_features['CLIMATE'].values.astype('float64')
WIND = time_series_features['WIND'].values.astype('float64')
time_series = np.zeros([30, 4])
time_series[:, 0] = MAXT
time_series[:, 1] = MINT
time_series[:, 2] = CLIMATE
time_series[:, 3] = WIND

path = 'members'
data_dict = import_data(path)

def count(occupancy):
    max_list = np.zeros([30])
    min_list = np.zeros([30])
    sum_list = np.zeros([30])
    output = np.zeros([30, 3])
    for i in range(30):
        max_list[i] = np.max(occupancy[0+288*i:288+288*i])
        min_list[i] = np.min(occupancy[0+288*i:288+288*i])
        sum_list[i] = np.sum(occupancy[0+288*i:288+288*i])

    output[:, 0] = max_list
    output[:, 1] = min_list
    output[:, 2] = sum_list
    return output

def perison(input, time_series):
    perison = np.zeros([12])
    for t in range(4):
        for z in range(3):
            temp = np.corrcoef(input[:, z], time_series[:, t])
            perison[3 * t + z] = temp[0, 1]
    return perison

perison_list = np.zeros([len(data_dict), 12])
for j in range(len(data_dict)):
    data = data_dict[j]
    input = count(data)
    perison_list[j, :] = perison(input, time_series)

# output loss
output = perison_list
f = open('perison_list.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output:
    csv_writer.writerow(l)
f.close()

