# coding = utf-8

import torch
import pandas as pd
import numpy as np
import read_dataset
import MyNet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import selector
import csv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ______________________ pre ______________________________
# input federation data
path = 'members'
name_dict, type_dict, data_dict, poi_density_list = read_dataset.import_data(path)


# input target data
target_name = "target1"
target_num = 0
target = pd.read_csv('target/target1.csv')
target_data = target['OCCUPANCY'].values.astype('float64')
target_data = torch.as_tensor(target_data, device=device).float()
target_type = 'res'
target_poi_density = 12

# _____________________ Selector ____________________
select_dict = selector.Eq_selector(data_dict, target_num)    # Equivalent Selector as A3C-Selector (RL)
# select_dict = selector.RL_selector(target_type, target_poi_density, data_dict, federation_state)
# select_dict = selector.type_selector(target_type, type_dict, data_dict)
# select_dict = selector.random_selector(data_dict)

times = 10
pre_training_epoch_num = 200
test_epoch_num = 400

test_size = len(target_data) - int(len(target_data)*0.8) - 6*2
occ_list = np.zeros([test_size, times])
output_matrix = torch.zeros([test_epoch_num, times*6])

for t in range(times):
    print('time =', t+1)
    # instantiation
    net = MyNet.MyLSTMNet(input_size=1, hidden_size=1, seq_len=6, output_size=1, num_layers=1).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)

    # input data
    train_task_dict = dict()
    support_set_dict = dict()
    query_set_dict = dict()
    support_dataloader_dict = dict()
    query_dataloader_dict = dict()
    task_num = len(select_dict)
    for n in range(task_num):
        train_task_dict[n] = select_dict[n]
        support_size = int(len(train_task_dict[n])*0.6)
        query_size = int(len(train_task_dict[n])*0.8)
        temp = train_task_dict[n][:support_size]
        support_set_dict[n] = read_dataset.MyData(data=temp, seq_length=6)
        support_dataloader_dict[n] = DataLoader(support_set_dict[n], batch_size=len(support_set_dict[n]), shuffle=False)
        temp = train_task_dict[n][support_size:query_size]
        query_set_dict[n] = read_dataset.MyData(data=temp, seq_length=6)
        query_dataloader_dict[n] = DataLoader(query_set_dict[n], batch_size=len(query_set_dict[n]), shuffle=False)

    # _______________ training ________________________
    # ori_ta matrix
    ori_ta = dict()
    temp_num = 0
    with torch.no_grad():
        for param in net.parameters():
            ori_ta[temp_num] = param.data
            temp_num += 1
    # outer loop
    for epoch in range(pre_training_epoch_num):
        if (epoch+1) % 10 == 0:
            print('pre_training_epoch =', epoch+1, '/', pre_training_epoch_num)
        # gradient matrix
        gradient = dict()
        temp_num = 0
        for param in net.parameters():
            if param.requires_grad:
                gradient[temp_num] = ori_ta[temp_num] * 0
                temp_num += 1
        # inner loop
        for n in range(task_num):
            # FOMAML
            # init params
            temp_num = 0
            with torch.no_grad():
                for param in net.parameters():
                    if param.requires_grad:
                        param.data = ori_ta[temp_num]
                        temp_num += 1

            for (support, query) in zip(support_dataloader_dict[n], query_dataloader_dict[n]):
                support_sample, support_label = support
                query_sample, query_label = query
                # ith task, Support_set, temporal parameter
                optimizer.zero_grad()
                output = net(support_sample)
                output = torch.squeeze(output)
                loss = loss_function(output, support_label)
                loss.backward()
                optimizer.step()
                # ith task, Query_set, gradient
                optimizer.zero_grad()
                output = net(query_sample)
                output = torch.squeeze(output)
                loss = loss_function(output, query_label)
                loss.backward()
                optimizer.step()
                # extract cumulative gradient
                with torch.no_grad():
                    gradient_index = 0
                    for param in net.parameters():
                        if param.requires_grad:
                            gradient[gradient_index] = gradient[gradient_index] + param.grad
                            gradient_index += 1

        # update ori_ta
        for index in range(len(ori_ta)):
            temp_value = ori_ta[index] - 0.03 * (gradient[index] / task_num)
            ori_ta[index] = temp_value
        # print('pretraining_epoch=', epoch+1, '/', pre_training_epoch_num)
    # ______________ testing _____________________
    support_size = int(len(target_data)*0.6)
    query_size = int(len(target_data)*0.8)
    support_target = target_data[support_size:query_size]
    query_target = target_data[query_size:]
    support_set = read_dataset.MyData(support_target, seq_length=6)
    query_set = read_dataset.MyData(query_target, seq_length=6)
    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    query_loader = DataLoader(query_set, batch_size=len(query_set), shuffle=False)

    fine_tuning_loss_list = torch.zeros([test_epoch_num])
    test_loss_list = torch.zeros([test_epoch_num])
    RMSE_list = torch.zeros([test_epoch_num])
    MAPE_list = torch.zeros([test_epoch_num])
    R2_list = torch.zeros([test_epoch_num])
    RAE_list = torch.zeros([test_epoch_num])

    # init params
    temp_num = 0
    with torch.no_grad():
        for param in net.parameters():
            if param.requires_grad:
                param.data = ori_ta[temp_num]
                temp_num += 1

    for epoch in range(test_epoch_num):
        # fine-tuning
        if (epoch+1) % 10 == 0:
            print('test_epoch =', epoch+1, '/', test_epoch_num)
        for i, support in enumerate(support_loader):
            support_sample, support_label = support
            optimizer.zero_grad()
            output = net(support_sample)
            output = torch.squeeze(output)
            loss = loss_function(output, support_label)
            loss.backward()
            optimizer.step()
            fine_tuning_loss_list[epoch] = loss.item()
            # print(epoch, '/', fine_tuning_epoch_num, 'Loss =', loss.item())

        # testing
        for i, query in enumerate(query_loader):
            query_sample, query_label = query
            output = net(query_sample)
            output = torch.squeeze(output)
            loss = loss_function(output, query_label)
            test_loss_list[epoch] = loss.item()

            # output metrics
            output = torch.reshape(output, [len(query_label)]).cpu()
            output = output.detach().numpy()
            query_label = torch.reshape(query_label, [len(query_label)]).cpu()
            query_label = query_label.detach().numpy()

            # calculate MAPE
            MAPE = np.mean(abs(output - query_label) / query_label) * 100
            # RMSE
            RMSE = np.sqrt(np.mean((output - query_label)*(output - query_label))) * 100
            # R2
            SSE = np.sum((output - query_label)*(output - query_label))
            SST =np.sum((np.mean(query_label) - query_label)*(np.mean(query_label) - query_label))
            R2_score = (1 - SSE / SST) * 100
            # RAE
            RAE = np.sum(abs(output - query_label)) / np.sum(abs(np.mean(query_label) - query_label)) * 100

            RMSE_list[epoch] = RMSE
            MAPE_list[epoch] = MAPE
            R2_list[epoch] = R2_score
            RAE_list[epoch] = RAE
    occ_list[:, t] = output
    output_matrix[:, 6*t+0] = RMSE_list
    output_matrix[:, 6*t+1] = MAPE_list
    output_matrix[:, 6*t+2] = R2_list
    output_matrix[:, 6*t+3] = RAE_list
    output_matrix[:, 6*t+4] = fine_tuning_loss_list
    output_matrix[:, 6*t+5] = test_loss_list


# ________________ output ______________________________________
print(target_name, ' result')
print('RMSE =', RMSE)
print('MAPE =', MAPE)
print('R2 =', R2_score)
print('RAE =', RAE)

output_matrix = output_matrix.detach().numpy()
# output loss
output = output_matrix
f = open('result/L3F_target1_metrics.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output:
    csv_writer.writerow(l)
f.close()

# output loss
output = occ_list
f = open('result/L3F_target1_occupancy.csv', 'w', newline='')
csv_writer = csv.writer(f)
for l in output:
    csv_writer.writerow(l)
f.close()

