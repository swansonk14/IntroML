import gym
import numpy as np

import lab11
import utils

#-------------------------------------------------------------------------------
# Part 1 - Reinforcement Learning
#-------------------------------------------------------------------------------

# Environment parameters
game = 'CartPole-v1'
episodes = 2000
render_frequency = 100

# DQN agent parameters
memory_max = 2000
gamma = 0.95
epsilon_init = 1.0
epsilon_decay = 1 - (5 / episodes)
epsilon_min = 0.01
hidden_size = 50
num_layers = 3
batch_size = 32

# Initialize game environment
env = gym.make(game)
state_size = np.prod(env.observation_space.shape)
action_size = env.action_space.n

#-------------------------------------------------------------------------------
# Part 1.1 - Random Agent
#-------------------------------------------------------------------------------

# # Initialize random agent
# agent = lab11.RandomAgent(state_size, action_size)

# # Render agent playing game
# utils.render_agent(env, agent, gif_path='random.gif')

#-------------------------------------------------------------------------------
# Part 1.2 - Engineered Agent
#-------------------------------------------------------------------------------

# # Initialize the engineered agent
# agent = lab11.EngineeredAgent(state_size, action_size)

# # Render agent playing game
# utils.render_agent(env, agent, gif_path='engineered.gif')

#-------------------------------------------------------------------------------
# Part 1.3 - Deep Q-Network (DQN) Agent
#-------------------------------------------------------------------------------

# # Initialize DQN agent
# agent = lab11.DQNAgent(state_size,
#                        action_size,
#                        memory_max,
#                        gamma,
#                        epsilon_init,
#                        epsilon_decay,
#                        epsilon_min,
#                        hidden_size,
#                        num_layers,
#                        batch_size)

# # Train DQN agent
# agent.train(env, episodes, render_frequency)

# # Render final agent playing game
# utils.render_agent(env, agent, gif_path='dqn.gif')
