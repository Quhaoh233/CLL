# coding = utf-8
import numpy as np
import torch
import csv
import time as clock
import torch.optim as optim
import MyNet
import glob
import read_dataset
import MyFunction
import meta_learner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# hyper params
action_number = 4060
state_num = 30

# input data
path = 'members'
name_dict, type_dict, data_dict, poi_density_list = read_dataset.import_data(path)
eps = np.finfo(np.float32).eps.item()

# define reward function
def reward_fn(target, action_index):
    target_data = target
    select_dict = dict()
    temp = 0
    action = MyFunction.action_transform_fn(action_index)
    for i in range(len(action)):
        if action[i] == 1:
            select_dict[temp] = data_dict[i]
            temp += 1
    ori_ta = meta_learner.FOMAML_training(select_dict, 20)  # pretraining epochs = 20
    MAPE = meta_learner.Testing(target_data, ori_ta, 10)  # pretraining epochs = 10
    reward = MAPE
    return reward

#  build state
def build_state(Newcomer, type_dict, poi_density_list):
    target_type = type_dict[Newcomer]
    target_poi_density = poi_density_list[Newcomer]
    # state_poi_density shape(30)
    SE = (poi_density_list - target_poi_density) * (poi_density_list - target_poi_density) + eps
    state_poi_density = target_poi_density / SE
    state_poi_density = torch.reshape(state_poi_density, [len(state_poi_density), 1])
    # state_type shape(30)
    state_type = np.zeros([30, 1])
    for i in range(len(state_type)):
        if target_type == type_dict[i]:
            state_type[i, :] = 1
    state_type = torch.tensor(state_type)
    state_type = state_type.to(device)
    state = torch.cat((state_poi_density, state_type), dim=1)
    return state

# instantiation
actor_net = MyNet.Actor(output_size=action_number).to(device)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-2)


for training_comer in range(state_num):
    state = build_state((training_comer), type_dict, poi_density_list)
    state = state.to(device)
    target_data = data_dict[(training_comer)]
    state = torch.reshape(state, [len(state), 2])
    R_list = torch.zeros([action_number, 1])
    R_list = R_list.to(device)
    for action_index in range(action_number):
        start_time = clock.time()
        print('state:', (training_comer), 'action:', action_index)
        # compute reward
        temp_reward = reward_fn(target_data, action_index)
        actor_net.action_index_list.append(action_index)
        reward = 15 - temp_reward  # bias = 15
        R_list[action_index] = reward
        actor_net.reward_list.append(reward)
        actor_net.state_list.append((training_comer))
        end_time = clock.time()
        time = end_time - start_time
        print("sample time =", time, 's')

    # output sample
    sample_state = np.reshape(actor_net.state_list, (-1, 1))
    sample_action_index = np.reshape(actor_net.action_index_list, (-1, 1))
    sample_reward = np.reshape(actor_net.reward_list, (-1, 1))
    sample = np.concatenate((sample_state, sample_action_index, sample_reward), axis=1)
    output_list = sample
    f = open('sample.csv', 'w', newline='')
    csv_writer = csv.writer(f)
    for l in output_list:
        csv_writer.writerow(l)
        print('writing')
    f.close()





