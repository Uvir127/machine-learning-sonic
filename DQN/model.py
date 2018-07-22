import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers import WarpGrayFrame, PyTorchFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.n_actions = action_space.n
        self.observation_size = observation_space.shape

        # TODO: initialize layers
        self.conv1 = nn.Conv2d(self.observation_size[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, x):
        # TODO: implement forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.size())
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
