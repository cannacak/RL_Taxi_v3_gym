import gym
import numpy as np
import random
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3").env

#Q table

#env.observation_space.n ==> 500
#env.action_space.n ==> 6

q_table = np.zeros([env.observation_space.n,env.action_space.n])

#Hyper paramters

alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Plotting Metrix

reward_list = []
dropouts_list = list()

episode_number = 2000

for i in range(1,episode_number):
    
    #initializing the environment
    state = env.reset()
    
    reward_count = 0
    dropouts = 0
    
    while True:
        
        #exploit vs explore to find an action (greedy)
        #%10 = explore, %90 = exploit
        
        if random.uniform(0, 1) < epsilon:
            
            action = env.action_space.sample() #returns random number 0 to 6
            
        else:
            
            action = np.argmax(q_table[state])
            
        
        #action prcoess and take the reward / take an observation
        
        next_state, reward, done, info = env.step(action)
        
        #Q-Learning function
        
        old_value = q_table[state,action]#old_value
        
        next_max = np.max(q_table[next_state])#next_max
        
        next_value = (1-alpha) * old_value + alpha*(reward + gamma * next_max)
        
        #Updating the Q-table 
        
        q_table[state,action] = next_value
        
        #update state
        state = next_state
        #find wrong dropouts
        
        if reward == -10:
            
            dropouts += 1
            
        
        if done:
            
            break
        
        reward_count += reward
    if i % 10 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print("Epsiode: {}, reward {}, wrong dropout: {}".format(i,reward_count, dropouts))
        
#%%

fig,axs = plt.subplots(1,2) # two plotting in one axis

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropouts_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

plt.show()