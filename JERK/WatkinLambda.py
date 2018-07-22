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


env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
LR = 0.1
DF = 0.99
greedy = 0.03
xLength = 10000
ylength = 1500

startState = np.zeros([1, 2])
endState = np.zeros([1, 2])
farX = 0

render = False

qLoad = 'WatkinLambdaTest.npy'

env.reset();

def main():
    global startState
    global endState
    watkinLambda()
    # runQ()

def watkinLambda():
    global startState
    global endState
    global farX

    if(os.path.exists(qLoad)):
        print('Reading in Qs')
        Q = np.load(qLoad)
    else:
        print('New Run')
        Q = np.zeros((xLength, ylength, np.shape(actions)[0]))

    e = np.zeros((xLength, ylength, np.shape(actions)[0]))
    indeces = []
    _, rew, done, info = env.step(actions[2])
    startState = [info['x'], info['y']]
    endState = [info['screen_x_end'], info['y']]

    currState = np.array(startState)
    newState = np.array(startState)

    numWins = 0
    episode = 0
    count = 0
    while(True):
        count += 1
        currActionIdx = pickAction(Q, currState)
        currAction = actions[currActionIdx]
        currSX = info['screen_x']
        thingsDone = []
        episode += 1
        rings = 0
        lives = 3
        jumped = False
        stuckCounter = 0
        if((numWins != 0) and ((numWins%20) == 0)):
            # print("Learn Jumps")
            Q = learnJumps(Q)

        while(True):
            if (render):
                env.render()
            count += 1

            if(currActionIdx == 1):
                doJump()
                _, rew, done, info = env.step(actions[2])
            else:
                _, rew, done, info = env.step(currAction)

            thingsDone.append([currState[0], currState[1], currActionIdx])
            newState = [info['x'], info['y']]
            newActionIdx = pickAction(Q, newState)
            newAction = actions[newActionIdx];

            r = reward(currState, startState, rew, info['level_end_bonus'], info['rings'])
            # time.sleep(0.01)

            if(newState[0] > farX):
                if(jumped and (len(thingsDone) > 2)):
                    print("DOnt get StuCK netx TimE")
                    stuck = thingsDone[-2]
                    Q[stuck[0]][stuck[1]][0] -= 5000
                    Q[stuck[0]][stuck[1]][1] += 5000
                    jumped = False
                farX = newState[0]
                stuckCounter = 0
            else:
                stuckCounter += 1
                if(jumped and (stuckCounter > 100)):
                    doLeft()
                    jumped = False

            if(stuckCounter == 2000):
                doJump()
                jumped = True
                stuckCounter = 0

            ydiff = abs(currState[1] - newState[1]);
            if(ydiff > 0):
                r += ydiff*100

            if(lives > info['lives']):
                lives = info['lives']
                currSX = 0
                nextSX = 0
                r -= 100000
                farX  = 0
                stuckCounter = 0
                rings = 0
                count = 0

                lengthTD = len(thingsDone)
                x = thingsDone[-1][0]
                for i in range(lengthTD-1, 0, -1):
                    if(thingsDone[i][0] != x):
                        currState = [thingsDone[i][0], thingsDone[i][1]]
                        currActionIdx = thingsDone[i][2]
                        newState  = [thingsDone[i+1][0], thingsDone[i+1][1]]
                        newActionIdx = thingsDone[i+1][2]

                        Q, e, indeces = update(Q, e, indeces, r, currState, currActionIdx, newState, newActionIdx)
                        thingsDone = []
                        break;
            else:
                if not (np.array_equal(currState, newState)):
                    Q, e, indeces = update(Q, e, indeces, r, currState, currActionIdx, newState, newActionIdx)

            currState = newState
            currAction = newAction
            currActionIdx = newActionIdx

            if done:
                break;

            if(info['level_end_bonus'] > 0):
                numWins += 1
                localtime = time.asctime( time.localtime(time.time()))
                print("Episode : ", episode, "Win : ", numWins, "Time : ", localtime)
                np.save(qLoad, Q)
                break;
        env.reset()

def learnJumps(Q):
    global startState
    global endState
    global farX

    _, rew, done, info = env.step(actions[2])
    startState = [info['x'], info['y']]
    endState = [info['screen_x_end'], info['y']]

    currState = np.array(startState)
    newState = np.array(startState)

    currActionIdx = pickAction(Q, currState)
    currAction = actions[currActionIdx]
    thingsDone = []
    lives = 3
    jumped = False
    stuckCounter = 0
    while(True):
        if (render):
            env.render()

        if(currActionIdx == 1):
            doJump()
            _, rew, done, info = env.step(actions[2])
        else:
            _, rew, done, info = env.step(currAction)

        thingsDone.append([currState[0], currState[1], currActionIdx])
        newState = [info['x'], info['y']]
        newActionIdx = pickBestAction(Q, newState)
        newAction = actions[newActionIdx];

        if(newState[0] > farX):
            if(jumped and (len(thingsDone) > 2)):
                print("DOnt get StuCK netx TimE")
                stuck = thingsDone[-2]
                Q[stuck[0]][stuck[1]][0] -= 5000
                Q[stuck[0]][stuck[1]][1] += 5000
                jumped = False
            farX = newState[0]
            stuckCounter = 0
        else:
            stuckCounter += 1
            if(jumped and (stuckCounter > 100)):
                doLeft()
                jumped = False

        if(stuckCounter == 2000):
            doJump()
            jumped = True
            stuckCounter = 0

        if(lives > info['lives']):
            lives = info['lives']
            currSX = 0
            nextSX = 0
            farX  = 0
            stuckCounter = 0
            thingsDone = []

        currState = newState
        currAction = newAction
        currActionIdx = newActionIdx

        if done:
            break;

        if(info['level_end_bonus'] > 0):
            localtime = time.asctime( time.localtime(time.time()))
            print("Episode : ", episode, "Win : ", numWins, "Time : ", localtime)
            np.save(qLoad, Q)
            break;

    env.reset()
    return Q


def reward(currState, startState, rew, LEB, rings):
    dist = abs(currState[0] - startState[0])
    return dist + rew + LEB*100

def pickAction(Q, state):
    if(random.uniform(0,1) > greedy):
        actionIdx = np.argmax(Q[state[0]][state[1]])
    else:
        actionIdx = random.randrange(np.shape(actions)[0])
    return actionIdx

def doLeft():
    for _ in range(150):
        left = actions[3]
        if (render):
            env.render()
        _, rew, done, info = env.step(left)

def doJump():
    for _ in range(6):
        if (render):
            env.render()
        _, rew, done, info = env.step(actions[1])

def update(Q, e, i, reward, cs, cai, ns, nai):
    l = 0.8
    bai = np.argmax(Q[ns[0]][ns[1]])
    sigma = reward + DF*Q[ns[0]][ns[1]][bai] - Q[cs[0]][cs[1]][cai]
    e[cs[0]][cs[1]][cai] = e[cs[0]][cs[1]][cai] + 1
    i.append([cs[0], cs[1], cai])

    dummyI = np.array(i)
    dim1,dim2 = np.shape(dummyI)
    for r in range(dim1):
        sx = dummyI[r][0]
        sy = dummyI[r][1]
        a  = dummyI[r][2]
        Q[sx][sy][a] += LR*sigma*e[sx][sy][a]
        if(nai == bai):
            e[sx][sy][a] = DF*l*e[sx][sy][a]
        else:
            e[sx][sy][a] = 0
    if(nai != bai):
        i = []
    return Q, e, i

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

    while(True):
        env.render()
        time.sleep(0.01)
        if(currActionIdx == 1):
            doJump()
            _, rew, done, info = env.step(actions[2])
        else:
            _, rew, done, info = env.step(currAction)

        print(Q[currState[0]][currState[1]])
        newState = [info['x'], info['y']]
        newActionIdx = pickBestAction(Q, newState)
        newAction = actions[newActionIdx];

        if(lives > info['lives']):
            lives = info['lives']

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
