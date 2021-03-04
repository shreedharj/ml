import numpy as np
import operator
import matplotlib.pyplot as plt
import pickle


class GridWorld:
    # Initialise starting data
    def __init__(self):
        # Set information about the grid world
        self.height = 15
        self.width = 15
        self.grid = np.zeros((self.height, self.width)) - 1

        # Set random start location for the agent
        self.start_location = (0, 0)

        # Set locations for the bomb and the gold
        self.gold_location = (14, 14)
        # self.trap_location1 = (3, 8)
        # self.trap_location2 = (7, 10)
        # self.trap_location3 = (5, 13)
        # self.trap_location4 = (5, 4)
        # self.trap_location5 = (12, 6)

        self.terminal_states = [self.gold_location]

        # self.terminal_states = [self.gold_location, self.trap_location1, self.trap_location2]
        # self.terminal_states = [self.gold_location, self.trap_location1, self.trap_location2, self.trap_location3,
        #                         self.trap_location4, self.trap_location5]

        # Set grid rewards for special cells
        self.grid[self.gold_location[0], self.gold_location[1]] = 10
        # self.grid[self.trap_location1[0], self.trap_location1[1]] = -100
        # self.grid[self.trap_location2[0], self.trap_location2[1]] = -100
        # self.grid[self.trap_location3[0], self.trap_location3[1]] = -100
        # self.grid[self.trap_location4[0], self.trap_location4[1]] = -100
        # self.grid[self.trap_location5[0], self.trap_location5[1]] = -100

        # Set available actions
        self.actions = ['U', 'D', 'L', 'R']

    def get_available_actions(self):  # returns possible actions
        return self.actions

    def agent_on_map(self):  # Prints start location of the agent on the grid
        grid = np.zeros((self.height, self.width))
        grid[self.start_location[0], self.start_location[1]] = (0, 0)
        return grid

    def get_reward(self, new_location):  # Returns the reward for an input position
        return self.grid[new_location[0], new_location[1]]

    def make_step(self, action):  # Agent moves in specified direction. At border, agent stays and collects (-) reward
        # Store previous location
        last_location = self.start_location

        if action == 'U':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.start_location = (self.start_location[0] - 1, self.start_location[1])
                reward = self.get_reward(self.start_location)

        elif action == 'D':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.start_location = (self.start_location[0] + 1, self.start_location[1])
                reward = self.get_reward(self.start_location)

        elif action == 'L':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.start_location = (self.start_location[0], self.start_location[1] - 1)
                reward = self.get_reward(self.start_location)

        elif action == 'R':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.start_location = (self.start_location[0], self.start_location[1] + 1)
                reward = self.get_reward(self.start_location)

        return reward

    def check_state(self):  # checks terminal states
        if self.start_location in self.terminal_states:
            return 'TERMINAL'  # this is where the program should end?


class RandomAgent():
    # Choose a random action
    def choose_action(self, available_actions):  # Returns a random choice of the available actions
        return np.random.choice(available_actions)


class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.05, alpha=0.9, gamma=0.9):
        self.environment = environment
        self.q_table = dict()  # Store all Q-values in dictionary of dictionaries
        pickling_on = open("self.q_table.pickle", "wb")
        pickle.dump(self.q_table, pickling_on)
        pickling_on.close()

        self.directional_q_table = dict()
        for x in range(environment.height):  # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(environment.width):
                self.q_table[(x, y)] = {'U': 0, 'D': 0, 'L': 0, 'R': 0}
                # Populate sub-dictionary with zero values for possible moves
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, available_actions):
        # returns optimal action. If multiple optional actions, one is chosen randomly.
        if np.random.uniform(0, 1) < self.epsilon:  # epsilon greedy strategy
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.start_location]
            maxvalue = max(q_values_of_state.values())  # picks max Q value of out the 4
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxvalue])
        return action

    def learn(self, old_state, reward, new_state, action):  # updates Q-values using Q-Learning
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        start_q_value = self.q_table[old_state][action]
        #  bellman equation
        self.q_table[old_state][action] = (1 - self.alpha) * start_q_value + self.alpha * (
                reward + self.gamma * max_q_value_in_new_state)

    def get_max(self, key):
        dict_data = self.q_table[(key[0], key[1])]
        max_val = dict_data['U']
        max_key = "U"
        for k in dict_data:
            v = dict_data[k]
            if v > max_val:
                max_val = v
                max_key = k
        return max_val, max_key

    def print_q_tab(self):
        for x in range(environment.width):  # printing the Q table keys
            for y in range(environment.height):
                v, k = self.get_max((x, y))
                # print("X {}, Y {} = {}".format(x, y, (v, k)))
                print("{}".format(k), end=" ")
            print()
        print()
        for x in range(environment.width):  # printing the Q values
            for y in range(environment.height):
                v, k = self.get_max((x, y))
                print("{}".format(round(v, 1)), end=" ")
            print('\n')


def play(environment, agent, episodes=10000, max_steps_per_episode=1000, learn=True):  # change to 100000
    #  Runs iterations and updates Q-values if desired.
    reward_per_episode = []  # Initialise performance log
    for trial in range(episodes):  # Run episodes
        cumulative_reward = 0  # Initialise values of each game
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:  # Run until max steps or until game is finished
            old_state = environment.start_location
            action = agent.choose_action(environment.actions)
            reward = environment.make_step(action)
            new_state = environment.start_location

            if learn == True:  # Update Q-values if learning is specified
                agent.learn(old_state, reward, new_state, action)

            cumulative_reward += reward  # cumulative reward increases over time
            # print(cumulative_reward)  # prints the cumulative reward for each episode ***

            if environment.check_state() == 'TERMINAL':  # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward)  # Append reward for start trial to performance log

    return reward_per_episode  # Return performance log


env = GridWorld()  # env is environment
agent = RandomAgent()

environment = GridWorld()
agentQ = Q_Agent(environment)
reward_per_episode = play(environment, agentQ, episodes=10000, learn=True)  # learn true allows learning

# plotting the learning curve
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Performance')
plt.plot(reward_per_episode)

agentQ.print_q_tab()
plt.show()
