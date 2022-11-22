#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# defines the reward matrix
r = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]]).astype("float64")
q = np.zeros_like(r)


def update_q(state, next_state, action, alpha, gamma):
    r_sa = r[state, action]
    q_sa = q[state, action]
    new_q = q_sa + alpha * (r_sa + gamma * max(q[next_state, :]) - q_sa)
    q[state, action] = new_q
    # rescale to between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]


def show_path():
    # show all the paths
    for i in range(len(q)):
        current_state = i
        path = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_state = np.argmax(q[current_state])
            current_state = next_state
            path += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        path = path[:-4]
        print("Optimal Path for starting state %i" % i)
        print(path)
        print("")


# hyperparameters
gamma = 0.7  # vary this between 0 and 1 i.e., (0,1)
alpha = 0.3  # vary this between 0 and 1 inclusive, i.e., [0,1]
n_episodes = 56000  # try different values, e.g., 10, 500, 10000 and so forth
epsilon = 0.05  # you can experiment with this as well

n_states = 6
n_actions = 6

random_state = np.random.RandomState(10)  # you may try without seed value and other seed values

for e in range(int(n_episodes)):
    states = list(range(n_states))
    random_state.shuffle(states)
    current_state = states[0]
    goal = False
    if e % int(n_episodes / 10.) == 0 and e > 0:
        pass
    while not goal:
        # epsilon greedy
        valid_moves = r[current_state] >= 0
        if random_state.rand() < epsilon:
            actions = np.array(list(range(n_actions)))
            actions = actions[valid_moves == True]
            if type(actions) is int:
                actions = [actions]
            random_state.shuffle(actions)
            action = actions[0]
            next_state = action
        else:
            if np.sum(q[current_state]) > 0:
                action = np.argmax(q[current_state])
            else:
                # Don't allow invalid moves at the start
                # Just take a random move
                actions = np.array(list(range(n_actions)))
                actions = actions[valid_moves == True]
                random_state.shuffle(actions)
                action = actions[0]
            next_state = action
        reward = update_q(current_state, next_state, action,
                          alpha=alpha, gamma=gamma)
        # Goal due to  rescaling
        if reward > 1:
            goal = True
        current_state = next_state
print("Q Table")
print(q)
show_path()


# In[ ]:




