# coding = utf-8
import numpy as np
import pandas as pd
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import MyNet

# hyper params
critic_batch_size = 4060
actor_batch_size = 4060
test_batch_size = 1
action_number = 4060
state_number = 30
sample_PATH = 'dataset/RL_samples.csv'
information_PATH = 'dataset/occupancies/information.csv'
cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eps = np.finfo(float).eps

# instantiation
actor_net = MyNet.Actor(action_number).to(cuda0)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-2)

critic_net = MyNet.Critic().to(cuda0)
critic_loss_fn = torch.nn.MSELoss()
critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-2)

# input data
value_list = torch.zeros([action_number * state_number], device=cuda0)

information = pd.read_csv(information_PATH)
type_list = information['TYPE']
type_list = torch.tensor(type_list).to(cuda0)
poi_density_list = information['POI_DENSITY']
poi_density_list = torch.tensor(poi_density_list).to(cuda0)


RL_samples = pd.read_csv(sample_PATH)
state_index_list = RL_samples['STATE']
state_index_list = torch.tensor(state_index_list).to(cuda0)

def build_state(state_i, type_l, poi_density_l):
    target_t = type_l[state_i]
    target_pd = poi_density_l[state_i]
    SE = (poi_density_l - target_pd) * (poi_density_l - target_pd)
    state_pd = target_pd / (SE + eps)
    state_pd = torch.as_tensor(state_pd)
    state_pd = torch.reshape(state_pd, [len(state_pd), 1])
    state_t = torch.zeros([state_number, 1], device=cuda0)
    for i in range(state_number):
        if target_t == type_l[i]:
            state_t[i] = 1
    output_state = torch.cat((state_pd, state_t), dim=1)
    return output_state

state_dict = dict()
for i in range(len(state_index_list)):
    state_index = state_index_list[i]
    state = build_state(state_index, type_list, poi_density_list)
    state_dict[i] = state

action_list = RL_samples['ACTION']
action_list = torch.tensor(action_list)
reward_list = RL_samples['REWARD']
temp_reward_list = reward_list
for i in range(state_number):
    max_r = max(reward_list)
    min_r = min(reward_list)
    mean_r = np.mean(reward_list)
    for j in range(action_number):
        temp_reward_list[i*action_number+j] = (reward_list[i*action_number+j] - mean_r) / (max_r - min_r)
reward_list = temp_reward_list
reward_list = torch.tensor(reward_list, device=cuda0)

class BUFFER(Dataset):
    def __len__(self):
        return len(value_list)

    def __getitem__(self, item):
        state_index = state_index_list[item]
        state = state_dict[item]
        action_index = action_list[item]
        reward = reward_list[item]
        value = value_list[item]
        return item, state_index, state, action_index, reward, value

buffer = BUFFER()

critic_dataloader = DataLoader(buffer, batch_size=critic_batch_size, shuffle=True)
actor_dataloader = DataLoader(buffer, batch_size=actor_batch_size, shuffle=True)
test_dataloader = DataLoader(buffer, batch_size=test_batch_size, shuffle=True)

def update_critic():
    # sample batch
    for item, s_index, s, a, r, v in critic_dataloader:
        critic_optimizer.zero_grad()
        critic_s_tensor = torch.as_tensor(s).float()
        output_value = critic_net(critic_s_tensor)    # (batch, 1)
        v = torch.reshape(v, [-1, 1])   # (batch, 1)
        loss = critic_loss_fn(output_value, v)
        loss.backward()
        print(loss.item())
        critic_optimizer.step()

def update_actor():
    # sample batch
    for item, s_index, s, a, r, v in actor_dataloader:
        actor_optimizer.zero_grad()
        # objective function
        actor_s_tensor = torch.as_tensor(s, device=cuda0).float()
        prob_matrix = actor_net(actor_s_tensor)    # (batch, 4060)
        batch_prob = torch.zeros([len(prob_matrix)], device=cuda0)
        for j in range(len(prob_matrix)):
            act = a[j]
            batch_prob[j] = prob_matrix[j, act]
        log_p = torch.log(batch_prob)
        v = torch.reshape(v, [-1, 1])
        r = torch.as_tensor(r, device=cuda0)
        adv = -log_p * (r - v)
        obj = torch.sum(adv) / actor_batch_size
        print(obj.data)
        obj.backward()
        actor_optimizer.step()


episode = 1
for i in range(episode):
    # update value
    for training_comer in range(state_number):
        print('updating value_list')
        s = build_state(training_comer, type_list, poi_density_list)
        s_tensor = torch.as_tensor(s, device=cuda0).float()
        s_tensor = torch.reshape(s_tensor, [1, state_number, 2])
        prob = actor_net(s_tensor)
        prob = torch.squeeze(prob)
        R = reward_list[training_comer*action_number:training_comer*action_number+action_number]
        R_float = torch.as_tensor(R, device=cuda0, dtype=torch.float)
        prob_float = torch.as_tensor(prob, dtype=torch.float)
        V = torch.dot(R_float, prob_float)
        for k in range(action_number):
            value_list[training_comer*action_number+k] = V

    # update critic_net
    for j in range(1):
        start_time = time.time()
        print('updating critic_net')
        update_critic()
        end_time = time.time()
        duration = end_time - start_time
        print('critic_net time=', duration)
    # update actor_net
    for k in range(2):
        start_time = time.time()
        print('updating actor')
        update_actor()
        end_time = time.time()
        duration = end_time - start_time
        print('actor_net time=', duration)
    print('episode=', i, '/', episode)


def testing():
    for test_comer in range(state_number):
        test_s = build_state(test_comer, type_list, poi_density_list)
        test_s_tensor = torch.as_tensor(test_s, device=cuda0).float()
        test_s_tensor = torch.reshape(test_s_tensor, [1, state_number, 2])
        test_prob = actor_net(test_s_tensor)    # (1, 4060)
        test_action = torch.argmax(test_prob, dim=1, keepdim=False)
        print(test_s, test_action.data)
testing()

torch.save(actor_net, 'actor_net.pkl')
torch.save(critic_net, 'critic_net.pkl')
