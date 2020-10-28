import numpy as np

from rl.qlearning.qagent import QAgent


def main():
    print("Q Learning Algorithm!")

    # Initialize parameters
    gamma = 0.9  # Discount factor
    alpha = 0.1  # Learning rate
    Q = np.array(np.zeros([9, 9]))

    # Define the states
    location_to_state = {
        'L1': 0,
        'L2': 1,
        'L3': 2,
        'L4': 3,
        'L5': 4,
        'L6': 5,
        'L7': 6,
        'L8': 7,
        'L9': 8
    }

    # Maps indices to locations
    state_to_location = dict((state, location) for location, state in location_to_state.items())

    # Define the actions
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # this reward matrix is for an open grid w/ no obstacles.
    rewards = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0]])

    qagent = QAgent(alpha, gamma, location_to_state, actions, rewards, state_to_location, Q)

    qagent.training('L1', 'L9', 1000)


if __name__ == "__main__":
    main()
