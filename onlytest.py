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
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space

        self.memory = deque(maxlen=100_000) ##we play 1000 episodes per epoch but only remember some of the events which happen, state action and reward, and use them to train 
                                         ##the model. This prevents us to use all the episodes and not use close events which do not provide many new information
                                         ## Also by sampling by different scenarios we analyze more of them. We store these groups of episodes up to 2000, then we start to forget the oldest

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_floor = 0.01
        self.epsilon_max = 1.0

        self.gamma = 0.9
        self.alpha = 0.001
        self.delta_epsilon = 0.01

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()

        model.add(Dense(24 , input_dim = self.state_size , activation= 'relu'))
        model.add(Dense(24 , activation = 'relu'))
        model.add(Dense(self.action_size , activation='linear'))

        model.compile(loss = 'mse', optimizer = Adam(learning_rate=self.alpha)) #prova a runnare cross entropy e vedi la differnza

        return model 
    
    def remember(self,state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        
        if np.random.rand() <= self.epsilon:
            #print("random")
            return self.action_space.sample()
        
        act_values = self.model.predict(state, verbose=0)
        #print(act_values)  #uses the predict method of the sequential model for exploitation trying to predict the best action to take 
        return np.argmax(act_values[0])
    
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


agent = DQNagent(state_size, action_size, action_space)
agent.model = keras.models.load_model("./Cartpole_model4.keras") #per riprendere un vecchio training


results = []

for e in range(test_eps): #TESTING

    state , _= env.reset()
    state = np.reshape(state, [1, state_size])#just transposing row to column
    #state = np.transpose(state)

    for time in range(500):#the cartpole can run maximum 5000 timesteps

        #env.render()
        action = agent.act(state)

        next_state, reward, done , _ , _,  = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        #print('miao')

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, test_eps, time, agent.epsilon))
            results.append(time)
            break
        

list_str = str(results)
with open("learning.txt", "w") as file:
    file.write(list_str)