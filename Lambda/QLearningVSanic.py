import retro
import numpy as np
import random
import time
import array
import math
import os.path
from statistics import mean, median
from collections import Counter

actions = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   #Right
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #Jump
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   #Nothing
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   #Left
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],   #Left, Down
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],   #Right, Down
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   #Down
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]   #Down, Jump

actionsPrint = ['Right', 'Jump', 'Nothing', 'Left', 'Left, Down', 'Right, Down', 'Down', 'Down, Jump']

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
LR = 0.1
DF = 0.9
greedy = 0.05
xLength = 10000
ylength = 1500

startState = np.zeros([1, 2])
endState = np.zeros([1, 2])

qLoad = 'QLearningVTest.npy'

env.reset();

def main():
    global startState
    global endState
    qlearningV()
    # runQ()

def qlearningV():
    global startState
    global endState

    if(os.path.exists(qLoad)):
        print('Reading in Qs')
        Q = np.load(qLoad)
    else:
        print('New Run')
        Q = np.zeros((xLength, ylength, np.shape(actions)[0]))

    obs, rew, done, info = env.step(actions[2])
    startState = [info['x'], info['y']]
    endState = [info['screen_x_end'], info['y']]

    currState = np.array(startState)
    newState = np.array(startState)

    save = True
    numWins = 0
    episode = 0

    while(True):
        currActionIdx = pickAction(Q, currState)
        currAction = actions[currActionIdx]
        episode += 1
        rings = 0
        lives = 3
        save = True

        while(True):
            env.render()

            if(currActionIdx == 1):
                obs, rew, done, info = env.step(currAction)
                obs, rew, done, info = env.step(currAction)
                obs, rew, done, info = env.step(currAction)
                obs, rew, done, info = env.step(currAction)
                obs, rew, done, info = env.step(currAction)
                obs, rew, done, info = env.step(actions[2])
            else:
                obs, rew, done, info = env.step(currAction)

            newState = [info['x'], info['y']]
            newActionIdx = pickAction(Q, newState)
            newAction = actions[newActionIdx];

            if(lives > info['lives']):
                lives = info['lives']
            print(currState, lives)
            r = reward(currState, newState, startState, lives, rew, info['level_end_bonus'], info['rings'], info['lives'])
            # time.sleep(0.1)
            Q[currState[0]][currState[1]][currActionIdx] = updateQ(Q[currState[0]][currState[1]][currActionIdx], Q[newState[0]][newState[1]], r)

            currState = newState
            currAction = newAction
            currActionIdx = newActionIdx

            if done:
                break;

            if(info['level_end_bonus'] > 0):
                numWins += 1
                localtime = time.asctime( time.localtime(time.time()))
                print("Episode : ", episode, "Win : ", numWins, "Time : ", localtime)
                # np.save(qLoad, Q)
                break;
        env.reset()


def reward(currState, newState, startState, currLives, rew, LEB, rings, gameLives):
    dist = abs(currState[0] - startState[0])

    reward = dist + rew + LEB*100s

    if(np.array_equal(currState, newState)):
        reward -= 5000

    if(currLives > gameLives):
        currLives = gameLives
        reward -= 100000
        rings = 0

    return reward

def pickAction(Q, state):
    if(random.uniform(0,1) > greedy):
        actionIdx = np.argmax(Q[state[0]][state[1]])
    else:
        actionIdx = random.randrange(np.shape(actions)[0])
    return actionIdx

def updateQ(QCurr, Qnew, reward):
    QCurr = (1-LR)*QCurr + LR*(reward + (DF*np.max(Qnew)))
    return QCurr

def runQ():
    global startState
    global endState
    print('Reading in Qs')
    Q = np.load(qLoad)

    obs, rew, done, info = env.step(actions[2])
    startState = [info['x'], info['y']]
    endState = [info['screen_x_end'], info['y']]

    currState = np.array(startState)
    newState = np.array(startState)

    currActionIdx = pickBestAction(Q, currState)
    currAction = actions[currActionIdx]
    lives = 3
    count = 0

    while(True):
        count += 1
        env.render()

        if(currActionIdx == 1):
            for _ in range(6):
                obs, rew, done, info = env.step(currAction)
                print("Lives : ", lives, currState, actionsPrint[currActionIdx], count)
                newState = [info['x'], info['y']]
                newActionIdx = currActionIdx
                newAction = actions[newActionIdx];
                count += 1
                currState = newState
                currActionIdx = newActionIdx
                currAction = newAction
            obs, rew, done, info = env.step(actions[2])
        else:
            obs, rew, done, info = env.step(currAction)


        # print(Q[currState[0]][currState[1]])
        # if(count == 197):
            # print('I actually died at state : ', currState)
        if(lives > info['lives']):
            lives = info['lives']
            print('The game said I Died at State : ', currState)

        print("Lives : ", lives, currState, actionsPrint[currActionIdx], count)
        time.sleep(0.01)
        newState = [info['x'], info['y']]
        newActionIdx = pickBestAction(Q, newState)
        newAction = actions[newActionIdx];

        currState = newState
        currAction = newAction
        currActionIdx = newActionIdx

        if done:
            break;
    env.reset()

def pickBestAction(Q, state):
    actionIdx = np.argmax(Q[state[0]][state[1]])
    return actionIdx

if __name__ == '__main__':
    main()
