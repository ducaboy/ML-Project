import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

##import dependecies

import gymnasium as gym 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt


##define agent

class DQNagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000) ##we play 1000 episodes per epoch but only remember some of the events which happen, state action and reward, and use them to train 
                                         ##the model. This prevents us to use all the episodes and not use close events which do not provide many new information
                                         ## Also by sampling by different scenarios we analyze more of them. We store these groups of episodes up to 2000, then we start to forget the oldest

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_floor = 0.01
        self.delta_epsilon = 0.001

        self.alpha = 0.01

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
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)  #uses the predict method of the sequential model for exploitation trying to predict the best action to take 

        return np.argmax(act_values[0])
    
    def replay(self, batch_size): #TRAINING

        minibatch = random.sample(self.memory, batch_size) #we create a random sample of batch size of some of the memories we took before. 

        for state, action, reward, next_state, done in minibatch:
            target = reward                                #if we met a termination condition, so max steps(2000) or lost, we fetch the reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0])) #otherwise we make an estimation of the future reward, using the NN
            target_f = self.model.predict(state)            
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0 )
 
    #def load(self, name):
     #   self.model.load_weights(name)

    #def save(self, name):
     #   self.model.save_weights(name)


def training():

    done = False
    results = []

    for e in range(learn_eps): #training

        state , _= env.reset()
        #print(state)
        state = np.reshape(state, [1, state_size])#just transposing row to column
        #state = np.transpose(state)

        for time in range(500):#the cartpole can run maximum 5000 timesteps

            #env.render() no render to make it fast

            action = agent.act(state)

            next_state, reward, done , _ , _,  = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, learn_eps, time, agent.epsilon))
                results.append(time)
                break

        if len(agent.memory) > batch_size:
        
            agent.replay(batch_size)

        if agent.epsilon > agent.epsilon_floor:
            #self.epsilon *= self.epsilon_decay
            agent.epsilon = agent.epsilon_floor + (agent.epsilon_max - agent.epsilon_floor) * np.exp(agent.delta_epsilon* e)

        if e % 50 == 0:
            agent.model.save('./Cartpole_model.keras')
            print('#agent updated#')
        
    print(results)

def testing():

    results = []
   
    for e in range(test_eps): #TESTING

        state , _= env.reset()
        state = np.reshape(state, [1, state_size])#just transposing row to column
        #state = np.transpose(state)

        for time in range(500):#the cartpole can run maximum 5000 timesteps

            env.render()

            agent.epsilon = 0.0

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
        
    print(results)
    return(results)




env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = int(env.action_space.n)
batch_size = 32
learn_eps = 300
test_eps = 30

#output_dir = './Cartpole_model.keras'

agent = DQNagent(state_size, action_size)

'''if not os.path.isfile('./Scrivania/ML/Cartpole_model.keras'):
    agent.model.save('./Scrivania/ML/Cartpole_model.keras')
    print("######### MODEL CREATED #############")
else:
    agent.model = keras.models.load_model("./Scrivania/ML/Cartpole_model.keras")
    print("######### MODEL LOADED ##############")'''

mod = str(input("Inserire modalità: "))

#### TRAIN THE MODEL ####
if mod == '1':
    #agent.model = keras.models.load_model("./Desktop/ML/Cartpole_model.keras") #per riprendere un vecchio training
    results = training()
    points = list(range(learn_eps))
    plt.plot(points , results)
    plt.title("rewards with alpha=0.01, gamma=0.99, deltaEpsilon=0.001")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()
    agent.model.save('./Cartpole_model.keras')
    print("######### MODEL CREATED #############")
#### TEST MODEL ####
else:
    agent.model = keras.models.load_model("./Cartpole_model.keras")
    print("######### MODEL LOADED ##############")
    testing()