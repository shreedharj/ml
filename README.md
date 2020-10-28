## Q-Learning Algorithm in Reinforcement Learning
Traversing a 3x3 grid. This code is referred from this tutorial: 
https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/


## Here are some of my observations:
* The action array is not used in this code at all because the actions are implicitly coded in the reward matrix itself. 
  * I can change the reward matrix according to my environment. Depending on if there are any obstacles or not.
  * The reward matrix is designed in a way that the agent is persuaded to follow the path to the final state only in 4 actions: north, south, east, west.
  * So, the Q-value matrix or the reward matrix isn't in the form of [(NxN)^2 x 4]. (NxN)^2 being the states by 4 actions matrix.
* For the 3x3 grid, it takes at least 100-200 iterations for the code to output the optimal path from the start state to the end state.
  * If the 15x15 grid is implemented using this code, it should take significantly more iteration! 
  * During the training stage, the algorithm picks a random start state on the grid and iterates through all the states to find their Q values. 
  * It doesn't necessarily mean all the states have an equal chance of being explored in those <100 iterations because the placement is random.
* To traverse the 15x15 grid using this code I plan to:
  * Not manually create the reward matrix because that will be a pain.
  * Write a code that follows the pattern and creates a reward table for me automatically.
  * This way I can create a reward table even for a 100x100 grid or bigger.

## I have the following questions:
* I tried changing the gamma and alpha values but there wasn't any significant difference in the output. Why is that so?
* How do you decide the number of iterations needed to complete the learning?

