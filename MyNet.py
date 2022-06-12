# coding = utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        # x is input, size (batch, seq, feature)
        x, _ = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x

class MyLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # utilize the GRU model in torch.nn
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        # x is input, size (batch, seq, feature)
        x, _ = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x

class MyFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size):
        super().__init__()
        self.backbone = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        # x is input, size (batch, seq, feature)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x

class MyTransformer(nn.Module):
    def __init__(self, d_model, nhead, seq_len, output_size):
        super().__init__()
        self.backbone = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = self.fc(x)
        return x

class MyBiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, output_size, num_layers):
        super().__init__()
        self.backbone = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)  # utilize the GRU model in torch.nn
        self.embedding = nn.Linear(hidden_size*2, hidden_size)
        self.fc = nn.Linear(seq_len, output_size)

    def forward(self, x):
        # x is input, size (batch, seq, feature)
        x, _ = self.backbone(x)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x

class Actor(nn.Module):
    def __init__(self, output_size):
        super(Actor, self).__init__()
        self.feature_net = nn.Linear(2, 1)
        # self.dropout = nn.Dropout(p=0.6)
        self.hidden1 = nn.Linear(30, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_size)

        self.action_index_list = []
        self.prob_list = []
        self.log_prob_list = []
        self.reward_list = []
        self.state_list = []
        self.value_list = []

    def forward(self, x):
        x = F.relu(x)
        x = self.feature_net(x)  # shape: [batch, 30, 1]
        # x = self.dropout(x)
        x = x.transpose(1, 2)   # shape: [batch, 1, 30]
        x = F.relu(x)
        x = self.hidden1(x)  # shape: [batch, 1, 256]
        res = x
        x = self.hidden2(x)
        x = x + res
        x = F.relu(x)
        x = self.actor(x)  # vector [batch, 1, 4060]
        b, s, h = x.shape
        action_scores = x.view(b*s, h)
        return F.softmax(action_scores, dim=1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.feature_net = nn.Linear(2, 1)
        # self.dropout = nn.Dropout(p=0.6)
        self.critic = nn.Linear(30, 1)

    def forward(self, x):
        x = F.relu(x)   # shape: [batch, 30, 2]
        x = self.feature_net(x)  # shape: [batch, 30, 1]
        # x = self.dropout(x)
        b, s, h = x.shape
        x = x.view(b, s*h)
        x = F.relu(x)
        x = self.critic(x)  # scalar, shape:[batch, 1, 1]
        return x
