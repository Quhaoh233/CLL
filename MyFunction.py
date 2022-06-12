import numpy as np
import torch

state_num = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = 0.000001 # a minimize value

def build_action_dict_fn():
    action_dict = dict()
    num = 0
    for i in range(30):
        for j in range(29-i):
            for k in range(28-i-j):
                action = np.zeros([30])
                action[i] = 1
                action[(30-29+i+j)] = 1
                action[(30-28+i+j+k)] = 1
                action_dict[num] = action
                num += 1
    return action_dict

action_dict = build_action_dict_fn()

def action_transform_fn(action_index):
    action = action_dict[action_index]
    return action

def build_state(state_i, type_l, poi_density_l):
    target_t = type_l[state_i]
    target_pd = poi_density_l[state_i]
    SE = (poi_density_l - target_pd) * (poi_density_l - target_pd)
    state_pd = target_pd / (SE + eps)
    state_pd = torch.as_tensor(state_pd)
    state_pd = torch.reshape(state_pd, [len(state_pd), 1])
    state_t = torch.zeros([state_number, 1], device=device)
    for i in range(state_number):
        if target_t == type_l[i]:
            state_t[i] = 1
    output_state = torch.cat((state_pd, state_t), dim=1)
    return output_state
