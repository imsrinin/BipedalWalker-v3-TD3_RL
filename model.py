import torch.nn as nn
import torch
import torch.nn.functional as F
# from torchsummary import summary
import torch.nn.utils.prune as prune
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # self.l1 = nn.Linear(state_dim, 1000)
        # self.l2 = nn.Linear(1000, 800)
        # self.l3 = nn.Linear(800, 600)
        # self.l4 = nn.Linear(600, 400)
        # self.l5 = nn.Linear(400, 200)
        # self.l6 = nn.Linear(200, 100)
        # self.l7 = nn.Linear(100, 20)
        # self.l8 = nn.Linear(20, action_dim)
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # a = F.relu(self.l3(a))
        # a = F.relu(self.l4(a))
        # a = F.relu(self.l5(a))
        # a = F.relu(self.l6(a))
        # a = F.relu(self.l7(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


# model = Actor(24,4,4)
# parameters = (
#     (model.l1, "weight"),
#     (model.l2, "weight"),
#     (model.l3, "weight"),
#     (model.l4, "weight"),
#     (model.l5, "weight"),
#     (model.l6, "weight"),
# )
# prune.global_unstructured(
#     parameters,
#     pruning_method=prune.L1Unstructured,
#     amount=0.8,
# )

# print(model.l1.weight)