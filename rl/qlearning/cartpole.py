import pickle
import gym
import numpy as np
import math
import sys
from collections import deque
import time
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt


class CartPole():
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=10000, n_win_ticks=195, min_alpha=0.1, min_epsilon=0.1,
                 gamma=0.9, ada_divisor=25, max_env_steps=None, monitor=False):
        self.buckets = buckets  # down-scaling feature space to discrete range
        self.n_episodes = n_episodes  # training episodes
        self.n_win_ticks = n_win_ticks  # average ticks over 100 episodes required for win
        self.min_alpha = min_alpha  # learning rate
        self.min_epsilon = min_epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.ada_divisor = ada_divisor  # only for development purposes

        self.env = gym.make('CartPole-v0')
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        if monitor: self.env = gym.wrappers.Monitor(self.env, 'tmp/cartpole-1', force=True)  # record results for upload

        # initialising Q-table
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    # Discretizing input space to make Q-table and to reduce dimensionality
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Choosing action based on epsilon-greedy policy
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    # Updating Q-value of state-action pair based on the update equation
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    # Adaptive learning of Exploration Rate
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    # Adaptive learning of Learning Rate
    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_gamma(self):
        return self.gamma

    def set_gamma(self, gamma):
        self.gamma = gamma

    def run(self, render=False):
        # Initialize variables to track rewards
        timestamp_list = []
        steps_list = []

        for e in range(self.n_episodes):
            # As states are continuous, discretize them into buckets
            current_state = self.discretize(self.env.reset())

            # Get adaptive learning alpha and epsilon decayed over time
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            start_time = round(time.time(), 5)

            while not done:
                # Render environment
                if render:
                    self.env.render()

                # Choose action according to greedy policy and take it
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)

                if done:
                    end_time = round(time.time(), 5)
                    time_elapsed = end_time - start_time
                    timestamp_list.append(time_elapsed)
                    reward = -1
                else:
                    reward = 0
                new_state = self.discretize(obs)

                # Update Q-Table
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            steps_list.append(i)

        return timestamp_list, steps_list


def print_learning_curve(timestamp_list):
    plt.xlabel('Episodes')
    plt.ylabel('Time in Seconds')
    plt.plot(timestamp_list)
    plt.show()


def print_steps_curve(steps_list):
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.plot(steps_list)
    plt.show()


def main():
    # Make an instance of CartPole class
    polecart = CartPole(min_epsilon=0.001, n_episodes=4000)
    if 'load' in sys.argv:
        polecart = pickle.load(open('polecart.pk', 'rb'))
        polecart.set_gamma(0.7)

    render = 'render' in sys.argv
    timestamp_list, steps_list = polecart.run(render=render)
    y = savgol_filter(steps_list, 101, 3)
    print_steps_curve(y)
    # print(timestamp_list)

    if 'save' in sys.argv:
        pickle.dump(polecart, open('polecart.pk', 'wb'))


if __name__ == "__main__":
    main()
