import numpy as np


class QAgent():

    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location, Q):

        self.gamma = gamma
        self.alpha = alpha
        self.location_to_state = location_to_state
        # self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        self.Q = Q

    # Training the robot in the environment
    def training(self, start_location, end_location, iterations):

        rewards_new = np.copy(self.rewards)

        ending_state = self.location_to_state[end_location]
        rewards_new[ending_state, ending_state] = 999

        for i in range(iterations):
            current_state = np.random.randint(0, 9)
            playable_actions = []

            for j in range(9):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)

            next_state = np.random.choice(playable_actions)
            TD = rewards_new[current_state, next_state] + \
                 self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] - \
                 self.Q[current_state, next_state]

            # The Q values are being added to the previous one here. I'm not sure if that what is supposed to happen.
            self.Q[current_state, next_state] = self.alpha * TD

        # print("Q Table: {}".format(self.Q))
        route = [start_location]
        next_location = start_location

        # Get the route
        self.get_optimal_route(start_location, end_location, next_location, route, self.Q)

    # Get the optimal route
    def get_optimal_route(self, start_location, end_location, next_location, route, Q):

        while next_location != end_location:
            print("Start location: {}".format(start_location))
            starting_state = self.location_to_state[start_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = self.state_to_location[next_state]
            print("Next location: {}".format(next_location))
            route.append(next_location)
            start_location = next_location
            print()

        print("This is the optimal route:")
        print(route)


