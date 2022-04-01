import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gym
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import time

from game_risk import Game
from players.guesser import *
from players.codemaster import *
from utils.import_string_to_class import import_string_to_class

# STATES
# Nb of red cards remaining = range(0, 9, 1) -> 9 possibilities
# Nb of blue cards remaining = range(0, 8, 1) -> 8 possibilities
# Nb of grey cards remaining = range(0, 10, 1) -> 10 possibilities
# Mean nb of words guessed / turn = range(0, 4, 0.2) -> 20 possibilities ??
# Ratio of good guesses = range(0, 1, 0.01) -> 100 possibilities

# ACTIONS
# risk threshold = range(0, 1, 0.1) -> 10 possibilities

# Path to models
codemaster_model = 'players.codemaster_glove_rl.AICodemaster'
guesser_model = 'players.guesser_w2v.AIGuesser'
w2v = 'players/GoogleNews-vectors-negative300.bin'
wordnet = None
glove = 'players/glove.6B.300d.txt'

# Global parameters
GAMMA = 0.99
BATCH_SIZE = 4
UPDATE_TARGET_EVERY = 32
EPSILON_START = 1.0
DECREASE_EPSILON = 200
EPSILON_MIN = 0.05
N_EPISODES = 200
LEARNING_RATE = 0.1
BUFFER_CAPACITY = 10000
EVAL_EVERY = 5
REWARD_THRESHOLD = 1000


# Define our network
class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


# create network and target network
hidden_size = 128
obs_size = 4
n_actions = 10

q_net = Net(obs_size, hidden_size, n_actions)
target_net = Net(obs_size, hidden_size, n_actions)

# objective and optimizer
objective = nn.MSELoss()
optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)

# create instance of replay buffer
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

def get_q(states):
    """
    Compute Q function for a list of states
    """
    with torch.no_grad():
        states_v = torch.FloatTensor([states])
        output = q_net.forward(states_v).detach().numpy()  # shape (1, len(states), n_actions)
    return output[0, :, :]  # shape (len(states), n_actions)

def choose_action(state, epsilon):
    """
    Return action according to an epsilon-greedy exploration policy
    """
    q_values = get_q([state])
    
    if random.random() < epsilon:
        return random.randint(0, n_actions-1) # env.action_space.sample()
    else:
        return np.argmax(q_values[0])

def eval_dqn(n_sim=5):
    """
    Monte Carlo evaluation of DQN agent.

    Repeat n_sim times:
        * Run the DQN policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    pass

    # env_copy = deepcopy(env)
    # episode_rewards = np.zeros(n_sim)
    
    # for i in range(n_sim):
    #     obs, done, rewards_sum = env_copy.reset(), False, 0

    #     while not done:
    #         action = choose_action(obs, epsilon=0)
    #         obs, reward, done, _ = env_copy.step(action)
    #         rewards_sum += reward

    #     episode_rewards[i] = rewards_sum
    
    # return episode_rewards

def update(state, action, reward, next_state, done, BATCH_SIZE, GAMMA):
    """
    ** TO BE COMPLETED **
    """
    
    # add data to replay buffer
    if done:
        next_state = None
    replay_buffer.push(state, action, reward, next_state)
    
    if len(replay_buffer) < BATCH_SIZE:
        return np.inf
    
    # get batch
    transitions = replay_buffer.sample(BATCH_SIZE)
    
    # Compute loss - TO BE IMPLEMENTED!
    states = torch.FloatTensor([transition[0] for transition in transitions])
    actions = torch.LongTensor([[transition[1]] for transition in transitions])
    
    values = q_net.forward(states).gather(1, actions)

    rewards = torch.FloatTensor([[transition[2]] for transition in transitions])
    next_states = torch.FloatTensor([np.zeros(obs_size) if transition[3] is None else transition[3] for transition in transitions])
    none_filter = torch.FloatTensor([[0.0] if transition[3] is None else [1.0] for transition in transitions])

    targets = torch.max(target_net.forward(next_states), 1, keepdim=True).values
    targets = rewards + GAMMA * targets * none_filter
    
    loss = objective(values, targets)
     
    # Optimize the model - UNCOMMENT!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.detach().numpy()

def train():
    print('Beginning training')
    print('Loading models...')

    codemaster = import_string_to_class(codemaster_model)
    guesser = import_string_to_class(guesser_model)
    g_kwargs = {}
    cm_kwargs = {}
    if wordnet is not None:
        brown_ic = Game.load_wordnet(wordnet)
        g_kwargs["brown_ic"] = brown_ic
        cm_kwargs["brown_ic"] = brown_ic
    if glove is not None:
        glove_vectors = Game.load_glove_vecs(glove)
        g_kwargs["glove_vecs"] = glove_vectors
        cm_kwargs["glove_vecs"] = glove_vectors
    if w2v is not None:
        w2v_vectors = Game.load_w2v(w2v)
        g_kwargs["word_vectors"] = w2v_vectors
        cm_kwargs["word_vectors"] = w2v_vectors

    print('Models imported')

    nb_guesses = 0
    nb_good_guesses = 0

    state = [8, 7, 9, 0]
    epsilon = EPSILON_START
    ep = 0
    total_time = 0

    game = Game(codemaster,
            guesser,
            seed='time',
            do_print=False,
            do_log=False,
            game_name='train_risk_rl',
            cm_kwargs=cm_kwargs,
            g_kwargs=g_kwargs,
            nb_guesses=nb_guesses,
            nb_good_guesses=nb_good_guesses,
            display_board=False)

    while ep < N_EPISODES:
        print(f"Episode {ep}")
        action = choose_action(state, epsilon)

        # take action
        next_state, reward, done = game.step(action)
        loss = update(state, action, reward, next_state, done)

        # update state
        state = next_state

        # end episode if done
        if done:
            nb_guesses, nb_good_guesses = game.get_guesses()
            game = Game(codemaster,
                guesser,
                seed='time',
                do_print=False,
                do_log=False,
                game_name='train_risk_rl',
                cm_kwargs=cm_kwargs,
                g_kwargs=g_kwargs,
                nb_guesses=nb_guesses,
                nb_good_guesses=nb_good_guesses,
                display_board=False)
            ratio = np.round(nb_good_guesses / nb_guesses, 2)
            state = [8, 7, 9, ratio]
            ep   += 1

            # update target network
            if ep % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # decrease epsilon
            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * \
                            np.exp(-1. * ep / DECREASE_EPSILON )    

        total_time += 1


def train_risk(GAMMA = 0.99,BATCH_SIZE = 4,UPDATE_TARGET_EVERY = 32,EPSILON_START = 1.0,DECREASE_EPSILON = 200,EPSILON_MIN = 0.05,N_EPISODES = 200,LEARNING_RATE = 0.1,BUFFER_CAPACITY = 10000,EVAL_EVERY = 5,REWARD_THRESHOLD = 1000):
    train()