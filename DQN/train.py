import gym
import retro
from retro_contest.local import make
import numpy as np
import random
import time
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wrappers import *
from DQNSanic import DQNAgent
from model import DQN

def main():
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    env = SonicDiscretizer(env)
    env = PyTorchFrame(env)

    agent = DQNAgent(action_space=env.action_space, observation_space=env.observation_space)

    if (os.path.exists('Best_episode.dpn')):
        print("Loading")
        params = torch.load('Best_episode.dpn')
        agent.policy_network.load_state_dict(params)
        agent.updateTarget()
    else:
        print("New")

    runSteps = agent.eps_fraction*float(agent.max_timesteps)
    episode_rewards = [0.0]
    loss = [0.0]
    bestEpisode = 0

    state = env.reset()
    for t in range(agent.max_timesteps):
        agent.eps_fraction = min(1.0, float(t)/runSteps)
        greedy = agent.eps_start + agent.eps_fraction * (agent.eps_end - agent.eps_start)
        # if (random.random() > greedy):
        #     action = agent.act(state)
        # else:
        #     action = env.action_space.sample()

        action = agent.act(state)

        new_state, rew, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)

        agent.replay_mem.add(state, action, rew, new_state, float(done))
        state = new_state

        episode_rewards[-1] += rew

        if (done):
            state = env.reset()
            episode_rewards.append(0.0)

        if ((t > agent.learning_starts) and (t % agent.train_freq == 0)):
            agent.optimize_td_loss()

        if ((t > agent.learning_starts) and (t % agent.target_update_freq == 0)):
            agent.updateTarget()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and agent.print_freq is not None and num_episodes % agent.print_freq == 0:
            print("********************************************************")
            print("Time: ", time.asctime( time.localtime(time.time())))
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * greedy)))
            print("********************************************************")

        if done and agent.save_freq is not None and num_episodes % agent.save_freq == 0:
            torch.save(agent.policy_network.state_dict(), agent.path)

        if done and episode_rewards[-2] > bestEpisode:
            bestEpisode = episode_rewards[-2]
            # torch.save(agent.policy_network.state_dict(), 'Best_episode.dpn')


if __name__ == '__main__':
    main()
