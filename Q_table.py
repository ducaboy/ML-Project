import os
import gymnasium as gym 
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1") 

def discretization():

    discretized_space = [] #voglio discretizzare lo spazio
    
    upperBound = [env.observation_space.high[0], 4 , env.observation_space.high[2], 4] #here i take the maximum and minimum values of the 4 states (i do it manually for velocity
    lowerBound = [env.observation_space.low[0], -4, env.observation_space.low[2], -4]  #since it goes to infinity) 

    for i in range(env.observation_space.shape[0]):

        intervals = np.linspace(upperBound[i], lowerBound[i], n_intervals) #given the bounds and the number of intervals i want i discretize the domanin into that intervals, I do it
                                                                            #inside  a for loop since i need to do it for each of the 4 states
        discretized_space.append(intervals)

    return discretized_space    #this function returns the discretized domain

def state_discretization(state, discretized_space): #this other function maps the state returned by the environment to the closest point in the discretized domain
    
    discretized_state= []

    for i in range(env.observation_space.shape[0]):
         
        if (state[i] >= max(discretized_space[i])):
            state[i] = max(discretized_space[i]) - 0.000001 #if the state is larger then the max value of the discretized state a key error is returned giving the length of the discretized
                                                            #space as the value of that state ex: (33, 50 , 12 ,4) this gives an error as the qtable entry 50 does not exist
        if (state[i] <= min(discretized_space[i])):
            state[i] = min(discretized_space[i]) + 0.000001     
        bin_index = np.digitize(state[i], discretized_space[i] ) # to do it it just uses this numpy function
        #discretized_state.append(discretized_space[i][bin_index]) #this function returns the index, so if I want the state I need to take the corresponding point in the discretized space
        discretized_state.append(bin_index) #anyway apparently i don't need that, it's more useful to get the index s.t i can use it in the q table so i do it

    return tuple(discretized_state)

def choose_action(d_state, Qtab ,epsilon):
    
    if np.random.uniform(0,1) <= epsilon:
            return random.randrange(env.action_space.n) #forse serve il .n??
    else:
         return np.argmax(Qtab[d_state]) #il max tra le due entrate della q table, definiscila e poi le aggiungi 
    
def Q_learning():

    Qtab = {} 
    epsilon = 1
    reward_list = []
    eps_list = []

    for i in range(n_intervals):                               #create a table for each possible entry of space velocity angle and angular velocity
         for j in range(n_intervals):
              for k in range(n_intervals):
                   for l in range(n_intervals):
                        Qtab.update({(i,j,k,l) : [0,0]})

    discretized_space = discretization() 
    
    for e in range(n_episodes):
    
        state, _ = env.reset()
        done = False                                          #reset the environemnt and get the starting space                            
        d_state = state_discretization(state, discretized_space)
        print(e , epsilon)
        episode_R = 0

        for step in range(max_steps):

            if done:
                break    
        
            #env.render()
            action = choose_action(d_state , Qtab , epsilon)                                    #we choose an action according to our policy and make the environment make a step 
            next_state, reward, done,_ , _ = env.step(action)           #fetch the results

            episode_R += reward
            
            next_d_state = state_discretization(next_state , discretized_space) #discretize the new state
            #print(next_d_state)

            if action == 0:
                
                Qtab[d_state][0] = Qtab[d_state][0] + alpha*(reward + gamma * np.max(Qtab[next_d_state])- Qtab[d_state][0]) #based on the action we took we update the table entry

            else:
                
                Qtab[d_state][1] = Qtab[d_state][1] + alpha*(reward + gamma * np.max(Qtab[next_d_state])- Qtab[d_state][1])
            
            #print(Qtab[next_d_state])
                
            d_state = next_d_state #we update the state, ww do it for OUR computations, the state is saved in the env which will do the next step on its own

        if epsilon > epsilon_floor:
            #epsilon -= epsilon_decay
            epsilon = epsilon_floor + (epsilon_max - epsilon_floor)*np.exp(-epsilon_delta*e)
        
        reward_list.append(episode_R)
        eps_list.append(epsilon)

    return Qtab , reward_list , eps_list

def testing(policy):

    episode_R = []
    #env.render()

    for i in range(n_runs):
        print(i)
        points = 0
        done = False
        state, _ = env.reset()
        discretized_space = discretization()                            
        d_state = state_discretization(state, discretized_space)
        for j in range(max_steps):
            if done:
                break
            if policy == 'random':
                action = random_a()
            if policy == 'greedy':
                action = greedy_a(d_state)
            next_state, reward, done, truncated, info = env.step(action)
            next_dstate = state_discretization(next_state , discretized_space)
            d_state = next_dstate
            points += reward

        episode_R.append(points)
        
    print(episode_R)
    return episode_R

def greedy_a(dstate):
    action = np.argmax(qtable[dstate])
    return action
def random_a():
    action = env.action_space.sample()
    return action
    

#epsilon = 1 
n_intervals = 50
n_episodes = 10000
n_runs = 200
max_steps = 500
epsilon_floor = 0.01
epsilon_max = 1

epsilon_delta = 0.0001
alpha = 0.5
gamma = 0.99

epsilon_decay = 0.000099
bucket = 100


qtable, reward_list , eps_lis = Q_learning()
# Example list

list_str = str(reward_list)
with open("learning.txt", "w") as file:
    file.write(list_str)

input("Press Enter to continue...")

total_reward = testing("greedy")

list_stri = str(total_reward)
with open("testing.txt", "w") as file:
    file.write(list_str)

total_reward = testing("random")