import os
import gymnasium as gym 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt


class DQNagent:
    def __init__(self, state_size, action_size, action_space):
        
        self.state_size = state_size        #we assign the dimension of the observation space, in our case 4
        self.action_size = action_size      #the size of the action space, in our case 2
        self.action_space = action_space    #the type of the action space

        self.memory = deque(maxlen=100_000) #we play a total of 500 episodes, each for a maximum of 500 steps, each step corresponds to a state. What we do is store these states in this big vector and then to make the agent learn we make him replay batches made of random elements of this vector. 
                                            #picking random events allow to pick different scenarios because if we just replay states in order they will jsut be too similar and the agent won't have too much new information to learn from
        self.epsilon = 1.0
        self.epsilon_floor = 0.01
        self.epsilon_max = 1.0

        self.gamma = 0.95
        self.alpha = 0.01
        self.delta_epsilon = 0.001

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential() #we build our network, it's a nn made of one input layer, one hidden and one output layer

        model.add(Dense(24 , input_dim = self.state_size , activation= 'relu'))
        model.add(Dense(24 , activation = 'relu'))
        model.add(Dense(self.action_size , activation='linear'))

        model.compile(loss = 'mse', optimizer = Adam(learning_rate=self.alpha)) #prova a runnare cross entropy e vedi la differnza

        return model 
    
    def remember(self,state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        
        if np.random.rand() < self.epsilon:
            print("random")
            return self.action_space.sample()
        print("exploit" , self.model.predict(state, verbose=0)[0])
        return np.argmax(self.model.predict(state, verbose=0)[0])
    
    def replay(self, batch_size): #TRAINING

        minibatch = random.sample(self.memory, batch_size) #we create a random sample of batch size of some of the memories we took before. 

        for state, action, reward, next_state, done in minibatch:
            target = reward                                #if we met a termination condition, so max steps(2000) or lost, we fetch the reward
            if not done:
                target = (reward + self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])) #otherwise we make an estimation of the future reward, using the NN
            target_f = self.model.predict(state, verbose=0)            
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0 )


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
action_space = env.action_space
batch_size = 32
learn_eps = 500
test_eps = 50
episode_steps = 500

agent = DQNagent(state_size, action_size, action_space)

results = []
epsilon = []

done = False

for e in range(learn_eps): #training

    totalReward = 0
    state , _= env.reset()
    state = np.reshape(state, [1, state_size])#just transposing row to column

    for time in range(episode_steps):#the cartpole can run maximum 600 timesteps

        action = agent.act(state)

        next_state, reward, done , _ , _,  = env.step(action)

        totalReward += reward

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            break

    if len(agent.memory) > batch_size:
    
        agent.replay(batch_size)       #print('end training')

    if agent.epsilon > agent.epsilon_floor:
        
        #agent.epsilon = (agent.epsilon_max - agent.epsilon_floor) * np.exp(-agent.delta_epsilon * e) + agent.epsilon_floor
        agent.epsilon = -0.00198*e +1

    if e % 20 == 0:
        
        agent.model.save('./Cartpole_model.keras')
        #print('#agent updated#')
    results.append(totalReward)
    print("episode: {}/{}, score: {}, e: {:.2}".format(e, learn_eps, totalReward, agent.epsilon))

    epsilon.append(agent.epsilon)
        #results.append(totalReward)
    #print(results, epsilon)    
list_str = str(results)
with open("learning.txt", "w") as file:
    file.write(list_str)

input("Press Enter to continue...")

play_results = []
agent.epsilon = 0.0

for e in range(test_eps): #TESTING

    state , _= env.reset()
    state = np.reshape(state, [1, state_size])#just transposing row to column
    #state = np.transpose(state)
    totalReward = 0

    for time in range(episode_steps):#the cartpole can run maximum 5000 timesteps

        #env.render()
        action = agent.act(state)

        next_state, reward, done , _ , _,  = env.step(action)

        totalReward += reward

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            break
    
    play_results.append(totalReward)
    print("episode: {}/{}, results: {}, score: {}, e: {:.2}".format(e, test_eps, len(play_results), totalReward, agent.epsilon))
    

list_str = str(play_results)
with open("playing.txt", "w") as file:
    file.write(list_str)

