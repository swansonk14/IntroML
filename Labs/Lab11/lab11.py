from collections import deque
import random

from keras.layers import Dense
from keras.models import Sequential
import numpy as np

#-------------------------------------------------------------------------------
# Part 1.1 - Random Agent
#-------------------------------------------------------------------------------

class RandomAgent:
    """An agent which randomly selects actions to perform."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """Given the current state of the game, selects an action to perform.

        Arguments:
            state(object): A gym observation object containing the current
                           state of the game.

        Returns:
            An int indicating the action to perform.
        """

        raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 1.2 - Engineered Agent
#-------------------------------------------------------------------------------

class EngineeredAgent:
    """An agent which is hand-engineered to select actions to perform."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """Given the current state of the game, selects an action to perform.

        Arguments:
            state(object): A gym observation object containing the current
                           state of the game.

        Returns:
            An int indicating the action to perform.
        """

        cart_position, cart_velocity, pole_angle, pole_velocity = state

        raise NotImplementedError

#-------------------------------------------------------------------------------
# Part 1.3 - Deep Q-Network (DQN) Agent
#-------------------------------------------------------------------------------

class DQNAgent:
    """An agent which uses a deep neural network to select actions to perform."""

    def __init__(self,
                 state_size,
                 action_size,
                 memory_max,
                 gamma,
                 epsilon_init,
                 epsilon_decay,
                 epsilon_min,
                 hidden_size,
                 num_layers,
                 batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_max)
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        """Builds the deep neural network which will learn to predict Q-values.

        Layers:
            - Dense, self.hidden_size neurons, relu activation (x self.num_layers)
            - Dense, self.action_size neurons, linear activation

        Compile:
            - mse loss
            - adam optimizer

        Returns:
            A keras Sequential model containing a deep neural network.
        """

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        for _ in range(self.num_layers - 1):
            model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model

    def remember(self, state, action, reward, next_state, game_over):
        """Add a the result of performing an action to the memory.

        Arguments:
            state(object): A gym observation object containing the current
                           state of the game.
            action(int): An int indicating the action performed at the
                         current state.
            reward(float): The reward gained by performing the action at
                           the current state.
            next_state(object): A gym observation object containing the
                                state reached after performing the action
                                in the current state.
            game_over(bool): A boolean indicating whether the game has ended
                             upon reaching next_state.
        """

        self.memory.append((state, action, reward, next_state, game_over))

    def predict(self, state):
        """Uses the neural network model to predict the Q-values for each action.

        Arguments:
            state(object): A gym observation object containing the current
                           state of the game.

        Returns:
            An array of floats indicating the predicted Q-values for taking
            each action from the current state.
        """

        return self.model.predict(np.array([state]))[0]

    def act(self, state):
        """Given the current state of the game, selects an action to perform.

        Arguments:
            state(object): A gym observation object containing the current
                           state of the game.

        Returns:
            An int indicating the action to perform.
        """

        # Explore with probability self.epsilon
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        # If not exploring, use best state predicted by model
        else:
            action_values = self.predict(state)
            action = np.argmax(action_values)

        return action

    def replay(self):
        """Train the neural network model by replaying previous game experiences."""

        # Get minibatch
        batch_size = min(len(self.memory), self.batch_size)
        minibatch = random.sample(self.memory, batch_size)

        # Initialize inputs and targets arrays
        inputs = np.zeros((len(minibatch), self.state_size))
        targets = np.zeros((len(minibatch), self.action_size))

        # Build array of inputs and targets
        for i, (state, action, reward, next_state, game_over) in enumerate(minibatch):
            # Set input
            inputs[i] = state

            # Compute reward
            if game_over:
                target = reward
            else:
                next_action_values = self.predict(next_state)
                target = reward + self.gamma * np.amax(next_action_values)

            # Set target to be predicted values
            targets[i] = self.predict(state)

            # Set target for action to be true target
            targets[i, action] = target

        # Train model
        self.model.train_on_batch(inputs, targets)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes, render_frequency):
        """Train the DQN agent to play the game.

        Arguments:
            env(object): A gym game environments.
            episodes(int): The number of game episodes to play
                           through and learn from.
            render_frequency(int): The number of episodes between
                                   each visual rendering of the
                                   agent playing the game.
        """

        for episode in range(episodes):
            state = env.reset()
            score = 0

            while True:
                # Render agent every 50 episodes
                if episode % render_frequency == 0:
                    env.render()

                # Use DQN to predict action
                action = self.act(state)

                # Perform action to get next state
                next_state, reward, game_over, _ = env.step(action)

                # Compute reward
                reward = reward if not game_over else -10

                # Remember the episode
                self.remember(state, action, reward, next_state, game_over)
                
                # Set state to next state
                state = next_state

                # Update score
                score += 1

                # End episode if game_over
                if game_over:
                    print("episode: {}/{}, score: {}, epsilon: {:.2}"
                          .format(episode, episodes, score, self.epsilon))
                    break

            # Train neural network model
            self.replay()
