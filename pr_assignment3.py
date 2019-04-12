# import numpy as np
# import pandas as pd
#
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon = e_greedy
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
#
#     def choose_action(self, observation):
#         self.check_state_exist(observation)
#         # action selection
#         if np.random.uniform() < self.epsilon:
#             # choose best action
#             state_action = self.q_table.loc[observation, :]
#             # some actions may have the same value, randomly choose on in these actions
#             action = np.random.choice(state_action[state_action == np.max(state_action)].index)
#         else:
#             # choose random action
#             action = np.random.choice(self.actions)
#         return action
#
#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         q_predict = self.q_table.loc[s, a]
#         if s_ != 'terminal':
#             q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
#         else:
#             q_target = r  # next state is terminal
#         self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
#
#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # append new state to q table
#             self.q_table = self.q_table.append(
#                 pd.Series(
#                     [0]*len(self.actions),
#                     index=self.q_table.columns,
#                     name=state,
#                 )
#             )

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UPRIGHT = 4
ACTION_UPLEFT = 5
ACTION_DOWNLEFT = 6
ACTION_DOWNRIGHT = 7

# probability for exploration
EPSILON = 0.01

# Sarsa step size
learning_rate = 0.5
dis_rate = 0.8
# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UPRIGHT, ACTION_UPLEFT, ACTION_DOWNLEFT, ACTION_DOWNRIGHT]

def step(state, action):
    i, j = state
    print
    print "i:",i
    print "j:",j
    print "1111",state
    print "action:",action
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]

    elif action == ACTION_UPLEFT:
        return [max(i - 1 - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_UPRIGHT:
        return [max(i - 1 - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWNLEFT:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
    elif action == ACTION_DOWNRIGHT:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False

# play for an episode
def episode(q_value):
    # track the total time steps in this episode
    time = 0
    print "q_value:", q_value
    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        print "values_:",values_
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            learning_rate * (REWARD + dis_rate * (q_value[next_state[0], next_state[1], next_action]) -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time

def figure_6_3():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 8))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('/Users/yuhaomao/Downloads/rte/figure_6_3.png')
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('GO')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('UP')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('DW')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('LE')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('RT')
            elif bestAction == ACTION_UPLEFT:
                optimal_policy[-1].append('UL')
            elif bestAction == ACTION_UPRIGHT:
                optimal_policy[-1].append('UR')
            elif bestAction == ACTION_DOWNLEFT:
                optimal_policy[-1].append('DL')
            elif bestAction == ACTION_DOWNRIGHT:
                optimal_policy[-1].append('DR')
    print "Optimal policy is:"
    for row in optimal_policy:
        print row
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND])),'\n'

if __name__ == '__main__':
    figure_6_3()

