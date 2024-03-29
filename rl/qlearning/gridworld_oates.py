import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import colors
import pickle
import sys
import time


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
        self.end_location = (14, 14)
        # self.trap_location1 = (3, 8)
        # self.trap_location2 = (7, 10)

        self.terminal_states = [self.end_location]
        # self.terminal_states = [self.end_location, self.trap_location1, self.trap_location2]

        # Set grid rewards for special cells
        self.grid[self.end_location[0], self.end_location[1]] = 10
        # self.grid[self.trap_location1[0], self.trap_location1[1]] = -100
        # self.grid[self.trap_location2[0], self.trap_location2[1]] = -100


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

    # function to change/overwrite the start location
    def set_start_location(self, start_state):
        self.start_location = start_state

    # function to return the end location
    def get_start_location(self):
        return self.start_location

    # function to change/overwrite the end location
    def set_end_location(self, end_state):
        self.end_location = end_state

    # function to return the end location
    def get_end_location(self):
        return self.end_location


class RandomAgent():
    # Choose a random action
    def choose_action(self, available_actions):  # Returns a random choice of the available actions
        return np.random.choice(available_actions)


class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.05, alpha=0.9, gamma=0.9):
        self.environment = environment
        self.q_table = dict()  # Store all Q-values in dictionary of dictionaries
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
        self.q_table[old_state][action] = start_q_value + self.alpha * \
                                          (reward + self.gamma * max_q_value_in_new_state - start_q_value)

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
        for x in range(self.environment.width):  # printing the Q table keys
            for y in range(self.environment.height):
                v, k = self.get_max((x, y))
                # print("X {}, Y {} = {}".format(x, y, (v, k)))
                print("{}".format(k), end=" ")
            print()
        print()
        for x in range(self.environment.width):  # printing the Q values
            for y in range(self.environment.height):
                v, k = self.get_max((x, y))
                print("{}".format(round(v, 1)), end=" ")

            print('\n')

    def get_epsilon(self):
        return self.epsilon

    #   function to change/overwrite the epsilon value
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_gamma(self):
        return self.gamma

    #   function to change/overwrite the gamma value
    def set_gamma(self, gamma):
        self.gamma = gamma

    def get_alpha(self):
        return self.alpha

    #   function to change/overwrite the alpha value
    def set_alpha(self, alpha):
        self.alpha = alpha


def play(environment, agent, episodes=10000, max_steps_per_episode=5000, learn=True, visualize=False, d3_visualize=False):
    #  Runs iterations and updates Q-values if desired.
    reward_per_episode = []  # Initialise performance log
    print("End location at start of play: ", environment.get_end_location())

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
            step += 1
            # print(cumulative_reward)  # prints the cumulative reward for each episode ***

            # print("Old State {} Action {}".format(old_state, action))
            if environment.check_state() == 'TERMINAL':  # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        # Visualize
        if visualize:
            q_tab_data = get_qtable_data(agent, agent.q_table)
            print_color_grid(q_tab_data, pause=False)

        # 3D Visualize
        if d3_visualize:
            if trial % 5000 == 0:
                q_tab_data = get_qtable_data(agent, agent.q_table)
                print_3D_plot(agent, q_tab_data, False)

        reward_per_episode.append(cumulative_reward)  # Append reward for start trial to performance log

    return reward_per_episode  # Return performance log


def get_qtable_data(qagent, qtable):
    # data = np.array((15, 15))
    # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array
    w, h = 15, 15
    data = [[0.0 for x in range(w)] for y in range(h)]
    for key, val in qtable.items():
        # print("Q Table Key: {} Data {}".format(key, val))
        x = key[0]
        y = key[1]
        data[x][y] = qagent.get_max(key)[0]
    return data


def apply_factor_qtable_data(qagent, qtable, factor):
    # data = np.array((15, 15))
    # https://stackoverflow.com/questions/6667201/how-to-define-a-two-dimensional-array
    w, h = 15, 15
    data = [[0.0 for x in range(w)] for y in range(h)]
    for key, val in qtable.items():
        val = [val] * factor


def print_color_grid(data, pause=True):
    # https://www.pythonpool.com/matplotlib-pcolormesh/
    fig, ax = plt.subplots()
    heatmap = plt.imshow(data, origin="upper", cmap="gray")
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+100+100")
    plt.draw()
    plt.pause(0.4)
    if pause:
        input("Press any key to continue...")
    plt.close()


def print_learning_curve(agentQ, reward_per_episode):
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Performance (Epsilon: {} | Gamma: {} | Alpha: {})'
              .format(agentQ.get_epsilon(), agentQ.get_gamma(), agentQ.get_alpha()))
    plt.plot(reward_per_episode)
    agentQ.print_q_tab()
    plt.show()


def print_3D_plot(agentQ, q_tab_data, pause=True):
    fig = plt.figure()
    wf = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(15), range(15))
    z = np.array(q_tab_data)
    wf.plot_wireframe(x, y, z, rstride=1, cstride=1, color='navy')
    wf.set_title('Performance (Epsilon: {} | Gamma: {} | Alpha: {})'
              .format(agentQ.get_epsilon(), agentQ.get_gamma(), agentQ.get_alpha()))
    # fig.savefig('plot.png')
    plt.show()


def main():
    environment1 = GridWorld()
    print("End location at start: ", environment1.get_end_location())
    print("Start location at start: ", environment1.get_start_location())

    agentQ = Q_Agent(environment1)

    if 'load' in sys.argv:
        agentQ = pickle.load(open('agent.pk', 'rb'))
        # environment1.set_end_location((12, 10))
        agentQ.set_gamma(0.7)

    agentQ.set_epsilon(0.05)
    agentQ.set_alpha(0.9)

    environment1.set_start_location((0, 0))
    print("End location before calling play: ", environment1.get_end_location())

    start_time = round(time.time() * 1000)
    # learn true allows learning and visualize true allows runtime color grid to update
    reward_per_episode = play(environment=environment1, agent=agentQ, episodes=10000, max_steps_per_episode=5000,
                              learn=True, visualize=False, d3_visualize=True)
    end_time = round(time.time() * 1000)

    # printing the color grid in the very end
    q_tab_data = get_qtable_data(agentQ, agentQ.q_table)
    # print_color_grid(q_tab_data, pause=True)

    # plotting the learning curve
    print_learning_curve(agentQ, reward_per_episode)


    # print_3D_plot(agentQ, q_tab_data)

    if 'save' in sys.argv:
        apply_factor_qtable_data(agentQ, agentQ.q_table, 1)  # this is manipulating QTab to influence performance
        pickle.dump(agentQ, open('agent.pk', 'wb'))

    print('Cumulative reward per episodes: %.2f' % (sum(reward_per_episode[:10000])/10000))
    print('Total run time %.2f ms' % (end_time - start_time), 'or %.2f mins' % ((end_time - start_time)/60000))
    print("End location: ", environment1.get_end_location())

    print(sys.argv)


if __name__ == "__main__":
    main()
