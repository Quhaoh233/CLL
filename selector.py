# coding = utf-8
import MyFunction
import random
import torch

def type_selector(target_type, type_dict, data_dict):
    select_dict = dict()
    temp = 0
    for n in range(len(type_dict)):
        if temp <= 2:
            if target_type == type_dict[n]:
                select_dict[temp] = data_dict[n]
                temp += 1
    return select_dict

def RL_selector(target_type, target_poi_density, data_dict, federation_state):
    selector = torch.load('actor_net.pkl')
    selected_dict = dict()
    state = MyFunction.build_state(federation_state, target_type, target_poi_density)
    prob = selector(state)
    action = torch.argmax(prob)
    action = MyFunction.action_transform_fn(action)
    temp = 0
    for i in range(len(action)):
        if action[i] == 1:
            selected_dict[temp] = data_dict[i]
            temp +=1
    return selected_dict

def Eq_selector(data_dict, target_num):
    selected_dict = dict()
    if target_num == 0:
        selected_dict[0] = data_dict[0]
        selected_dict[1] = data_dict[3]
        selected_dict[2] = data_dict[4]
    if target_num == 1:
        selected_dict[0] = data_dict[7]
        selected_dict[1] = data_dict[9]
        selected_dict[2] = data_dict[26]
    if target_num == 2:
        selected_dict[0] = data_dict[2]
        selected_dict[1] = data_dict[13]
        selected_dict[2] = data_dict[24]
    if target_num == 3:
        selected_dict[0] = data_dict[12]
        selected_dict[1] = data_dict[17]
        selected_dict[2] = data_dict[26]
    return selected_dict

def random_selector(data_dict):
    select_dict = dict()
    beg = 0
    end = 4060 - 1
    index = random.randint(beg, end)
    action = MyFunction.action_transform_fn(index)
    temp = 0
    for i in range(30):
        if action[i] == 1:
            select_dict[temp] = data_dict[i]
            print(i)
            temp += 1
    return select_dict
