import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from model import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Fill in default hyperparameters
class DQNAgent:
    def __init__(self,
                action_space,
                observation_space,
                eps_start=0.1,
                eps_end=0.02,
                eps_fraction=0.1,
                batch_size=32,
                gamma=0.99,
                target_update_freq=1000,
                replay_buffer_size=int(1e5),
                lr=1e-4,
                max_timesteps=int(7000000),
                learning_starts=10000,
                train_freq=1,
                print_freq=10,
                save_freq=100,
                path='latest_episode.dqn'):

        # TODO: Store hyperparameters
        self.action_space = action_space
        self.observation_space = observation_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_fraction = eps_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.replay_buffer_size = replay_buffer_size
        self.lr = lr
        self.max_timesteps = max_timesteps
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.print_freq = print_freq
        self.save_freq=save_freq
        self.path=path

        # TODO: Initialise replay memory
        self.replay_mem = ReplayBuffer(replay_buffer_size)
        # TODO: Initialise policy network
        self.policy_network = DQN(self.observation_space, self.action_space).to(device)
        # TODO: Initialise target network
        self.target_network = DQN(self.observation_space, self.action_space).to(device)
        # TODO: Copy policy network parameters to target network
        self.updateTarget()
        # TODO: Set target network to eval mode
        self.target_network.eval()

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def optimize_td_loss(self):

        # TODO: Sample a batch from replay memory
        states, actions, rewards, next_states, dones = self.replay_mem.sample(self.batch_size)

        # TODO: Convert states, actions, rewards, next_states, dones to Tensors and send to gpu
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        # TODO: Compute next Q-values using target network
        with torch.no_grad():
            nextQ = self.target_network(next_states)
            max_nextQ, _ = nextQ.max(1)
            y = rewards + (1-dones) * self.gamma * max_nextQ

        # TODO: Compute current Q-values using policy network
        currQ = self.policy_network(states)

        # TODO: Gather the Q-values for the sampled actions
        currQ = currQ.gather(1, actions.unsqueeze(1)).squeeze()

        # TODO: Compute the TD error
        loss = F.smooth_l1_loss(currQ, y)

        # TODO: Zero the optimizer's gradient buffers
        self.optimizer.zero_grad()

        # TODO: Call .backward() to compute the gradients
        loss.backward()

        # TODO: Update the policy network's parameters
        self.optimizer.step()

    def updateTarget(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state):
        # TODO: return greedy action for state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.policy_network(state)
            _, action = out.max(1)
        return action.item()
