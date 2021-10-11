import pickle
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


class MountainCar():
    # Define Q-learning function
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.8, min_eps=0, episodes=10000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.episodes = episodes
        self.env = gym.make('MountainCar-v0')

        # Determine size of discretized state space
        num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        # Initialize Q table
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    def run(self):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []

        # Calculate episodic reduction in epsilon
        reduction = (self.epsilon - self.min_eps) / self.episodes
        # Run Q learning algorithm

        for i in range(self.episodes):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            state = env.reset()

            # Discretize state
            state_adj = (state - env.observation_space.low) * np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            while not done:
                # Render environment for the episodes
                # if i <= 100 or i >= (episodes - 500):
                if False and i >= (self.episodes - 20):
                    env.render()

                # Determine next action - epsilon greedy strategy
                if np.random.random() < 1 - self.epsilon:
                    action = np.argmax(self.Q[state_adj[0], state_adj[1]])
                else:
                    action = np.random.randint(0, env.action_space.n)

                # Get next state and reward
                state2, reward, done, info = env.step(action)

                # if done:
                #     reward = 1
                # else:
                #     reward = 0

                # Discretize state2
                state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
                state2_adj = np.round(state2_adj, 0).astype(int)

                # Allow for terminal states
                if done and state2[0] >= 0.5:
                    self.Q[state_adj[0], state_adj[1], action] = reward

                # Adjust Q value for current state
                else:
                    self.Q[state_adj[0], state_adj[1], action] += self.alpha * (reward + self.gamma *
                    np.max(self.Q[state2_adj[0], state2_adj[1]]) - self.Q[state_adj[0], state_adj[1], action])

                # Update variables
                tot_reward += reward
                state_adj = state2_adj

            # Decay epsilon
            if self.epsilon > self.min_eps:
                self.epsilon -= reduction

            # Track rewards
            reward_list.append(tot_reward)

            if (i + 1) % 100 == 0:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

        env.close()

        return ave_reward_list

    def get_gamma(self):
        return self.gamma

    def set_gamma(self, gamma):
        self.gamma = gamma


def main():
    cart = MountainCar(env=env)
    if 'load' in sys.argv:
        cart = pickle.load(open('mtncar.pk', 'rb'))
        cart.set_gamma(0.99)

    rewards = cart.run()

    if 'save' in sys.argv:
        pickle.dump(cart, open('mtncar.pk', 'wb'))

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('mtncar2rewards.jpg')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
