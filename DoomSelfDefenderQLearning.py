import gym
import numpy
import time
import ppaquette_gym_doom
import itertools
import sys
import random
import operator


THRESHHOLD = 20;

def enemyLocation2(observation):
    v = observation[199]
    arr = [0,0,0]
    means = numpy.empty(shape=(v.shape[0]/20-1))
    for c in xrange(means.shape[0]):
        means[c] = numpy.mean(v[c*20:c*20+39,:])
    totalMean= numpy.mean(means)
    for k in xrange(means.shape[0]):
            if(abs(means[k]-totalMean) > THRESHHOLD):
                    if(k<15):
                            arr[0]=1
                            k=14
                    elif(k==15):
                            arr[1]=1
                    else:
                            arr[2]=1
                            return arr
    return arr                        










def getStateCount(sp, s):
    a = 0
    for i in xrange(sp):
        a+=s[i]*pow(2,i)
    return a


def getOptimalFuture(markov, totals, qValues, state, action):
    possibilities = markov[state][action]/((totals[:,action])+0.0)
    valueOfPossibilities = 0
    for p in xrange(len(possibilities)):
        valueOfPossibilities+=(possibilities[p]*max(qValues[p]))
    return valueOfPossibilities



env = gym.make('ppaquette/DoomDeathmatch-v0')
statePossibilities = 4
stateSize = pow(2,statePossibilities)
actionSize = 6
discount = 0.1
learning_rate = 0.5
exploitRate = 1.0
episode_count = 10


f = open('meanResults.txt', 'w')
random.seed()


markovPossibilities=numpy.ones((stateSize,actionSize,stateSize))
totalChosen=numpy.ones(shape=(stateSize,actionSize))*stateSize
qValues = numpy.zeros(shape=(stateSize,actionSize))




arrR = list(0 * i for i in range(43))
arrR[14]=1 #always return right action


arr = list(0 * i for i in range(43))
secondsToRun = sys.argv[1]
start_time = time.time()


tempRewards = numpy.empty(shape=(episode_count))
while float(secondsToRun) > (time.time()-start_time):
    if discount<0.9:
        discount+=0.000001
    for i_episode in range(episode_count):
        print("Episode: ", i_episode )
        observation = env.reset()


        # index 0 for fire
        # index 14 for right
        # index 15 for left
        newState = -1
        totalReward = 0
        done = False
        canFire = False
        while(not done): #during an episode
            env.render()
            if not newState==-1:
                state = newState
            else:
                state = enemyLocation2(observation)
                state.append(canFire)
            
            stateCount = getStateCount(statePossibilities,state)
            
            #decides whether to exploit or take a random action
            
            if random.random()<exploitRate:
                if canFire:
                    actionIndex = numpy.argmax(qValues[stateCount])
                #if can't fire reduces the set of possible actions
                else:
                    actionIndex = numpy.argmax(qValues[stateCount,0:3])
            else:
                actionIndex = random.randint(0,actionSize-1)
            
            arr[0] = actionIndex/3
            if actionIndex%3==0:
                arr[14] = 0
                arr[15] = 0
            elif actionIndex%3==1:
                arr[14] = 1
                arr[15] = 0
            else:
                arr[14] = 0
                arr[15] = 1
            #3,4,5 do not fire
            #0,1,2 stop, left, right
            #0 left exists, 1 middle, 2 right state


            observation, reward, done, info = env.step(arr)
            canFire = 1==info.get("ATTACK_READY",0) and info.get("SELECTED_WEAPON_AMMO",0)!=0
            #takes the action and gets the "canfire" info, if gun is not recently fired and there is enough ammo gun can fire
            

            if reward>0:
                reward=reward*2
            if arr[0]==1:
                reward = reward-10.0/130.0
            totalReward+=reward
            #calculates rewards for the actions


            newState =  enemyLocation2(observation)
            newState.append(canFire)
            newStateCount = getStateCount(statePossibilities,newState)
            #new state is created from the observation and info


            if canFire:
                bestNextActionIndex = numpy.argmax(qValues[newStateCount])
            else:
                bestNextActionIndex = numpy.argmax(qValues[stateCount,0:3])
            #best action for the consequent step is decided


            qValues[stateCount][actionIndex] = (1-learning_rate)*qValues[stateCount][actionIndex] + learning_rate*(reward+discount*qValues[newStateCount][bestNextActionIndex])


            #updates qvalue indice for the state-action pair depending on the next state and the reward
            #this might be revised since we "know" the next step, no need to calculate probability matrices.. 


            #these parts are omitted because of a crucial misunderstangind...
            #getOptimalFuture(markovPossibilities,totalChosen,qValues,stateCount,actionIndex)
            #totalChosen[stateCount][actionIndex] = totalChosen[stateCount][actionIndex] +1
            #markovPossibilities[stateCount][actionIndex][newStateCount] = markovPossibilities[stateCount][actionIndex][newStateCount]+1
            #possibility data is updated
            #if(reward!=0):
            #    print(reward)
        tempRewards[i_episode] = totalReward
    print(numpy.mean(tempRewards))
    f.write('%s' % numpy.mean(tempRewards))
    f.write(" ")
f.close()
f = open('qvalues.txt', 'w')
tempV = 0
for x in xrange(qValues.shape[0]):
    for y in xrange(qValues.shape[1]):
        if (not tempV==x):
            f.write("\n")
            tempV = x
        f.write(str(qValues[x][y]))
        f.write(" ")
        
f.close()


tempV = 0
tempV0 = 0
f = open('markov.txt', 'w')
for x in xrange(markovPossibilities.shape[0]):
    for y in xrange(markovPossibilities.shape[1]):
        for z in xrange(markovPossibilities.shape[2]):
            if not tempV==x:
                f.write("\n")
                tempV = x
            if (not tempV0==y):
                f.write("\t")
                tempV0 = y
            f.write(str(markovPossibilities[x][y][z]))
            f.write(" ")
            
f.close()