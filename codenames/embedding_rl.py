# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import scipy.spatial.distance
import gensim.models.keyedvectors as word2vec
from game import Game
import matplotlib.pyplot as plt

# Discount factor
GAMMA = 0.99

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 1000 # 10000

# Initial value of epsilon
EPSILON_START = 1.0
# Parameter to decrease epsilon
DECREASE_EPSILON = 5000
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes
N_EPISODES = 10000

# Learning rate
LEARNING_RATE = 0.5

EVAL_EVERY = 50
REWARD_THRESHOLD = 100

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

class Trainer():

    def __init__(self, master_vectors, guesser_vectors):

        self.master_vectors = master_vectors
        self.guesser_vectors = guesser_vectors

        # create instance of replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

        #create training pool
        total_wordlist = []
        with open('players/reduced_cm_wordlist.txt') as infile:
            for line in infile:
                total_wordlist.append(line.rstrip())

        self.clues = np.array(total_wordlist)

        board = []
        with open('reduced_game_wordpool.txt') as infile:
            for line in infile:
                board.append(line.rstrip().lower())

        self.board = np.array(board)

        print(len(self.board))
        print(len(self.clues))

        # create network
        hidden_size = 128
        master_size = len(self.master_vectors["word"])
        guesser_size = len(self.guesser_vectors["word"])
        n_actions = len(self.clues)

        self.q_net_codemaster = Net(master_size, hidden_size, n_actions)
        if torch.cuda.is_available(): 
            self.q_net_codemaster.cuda()

        """ q_net_guesser = Net(guesser_size, hidden_size, n_actions)
        if torch.cuda.is_available(): 
            q_net_guesser.cuda()

        # objective and optimizer
        objective_guesser = nn.MSELoss()
        optimizer_guesser = optim.Adam(params=q_net_guesser.parameters(), lr=LEARNING_RATE) """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())

    def get_q(self, states):
        """
        Compute Q function for a list of states
        """
        with torch.no_grad():
            states_v = torch.FloatTensor([states])
            states_v = states_v.to(self.device)
            output = self.q_net_codemaster.forward(states_v).detach().cpu().numpy()  # shape (1, len(states), n_actions)
        return output[0, :, :]  # shape (len(states), n_actions)

    def choose_clue(self, state, epsilon, avoid=[]):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if np.random.uniform()<epsilon:
            index = np.random.randint(0, len(self.clues))
            return index, 0
        else:
            q=self.get_q([state])[0]
            for i in avoid:
                q[i] = -np.inf
            return q.argmax(), q.max()

    def clue_to_word(self, clue_index):
        return self.clues[clue_index]

    def check_clue(self, bad_words, index, proba):
        for word in bad_words:
            q = self.get_q([self.master_vectors[word]])[0]
            if q[index] > proba:
                return False
        return True

    def choose_word(self, board, word_vectors, clue):
        w2v = []

        for word in board:
            try:
                w2v.append((scipy.spatial.distance.cosine(word_vectors[clue], word_vectors[word.lower()]), word))
            except KeyError:
                print(">>> error")
                continue

        w2v = list(sorted(w2v))
        return w2v[1]

    def eval_dqn(self, n_sim=5, test=False):
        """
        Monte Carlo evaluation of DQN agent.

        Repeat n_sim times:
            * Run the DQN policy until the environment reaches a terminal state (= one episode)
            * Compute the sum of rewards in this episode
            * Store the sum of rewards in the episode_rewards array.
        """
        episode_rewards = np.zeros(n_sim)
        eval_board = random.choices(self.board, k=25)

        for sim in range(n_sim):
            word_to_guess = eval_board[np.random.randint(0, len(eval_board))]
            #print(">>> To guess : ", word_to_guess)
            state=self.master_vectors[word_to_guess]
            action=self.choose_clue(state, 0.0)[0]
            #print(">>> Clue : ", clues[action])
            _, chosen_word = self.choose_word(eval_board, self.guesser_vectors, self.clues[action])
            #print(">>> Chosen : ", chosen_word)
            next_state = None
            reward = 1 if chosen_word == word_to_guess else 0
            episode_rewards[sim]+=reward
            state=next_state
            if test :
                print(">>> To guess : ", word_to_guess)
                print(">>> Clue : ", self.clues[action])
                print(">>> Chosen : ", chosen_word)
                
        return episode_rewards

    def update(self, state, action, reward, next_state, done, optimizer, objective):
        """
        ** TO BE COMPLETED **
        """

        # add data to replay buffer
        if done:
            next_state = None
        self.replay_buffer.push(state, action, reward, next_state)
        
        if len(self.replay_buffer) < BATCH_SIZE:
            return np.inf
        
        # get batch
        transitions = self.replay_buffer.sample(BATCH_SIZE)
        
        states=torch.FloatTensor([transitions[ii][0] for ii in range(BATCH_SIZE)])
        #print(f'STATE : {states}')
        actions=torch.LongTensor([transitions[ii][1] for ii in range(BATCH_SIZE)]).view(-1,1)
        #print(f'ACTIONS : {actions}')
        rewards=torch.FloatTensor([transitions[ii][2] for ii in range(BATCH_SIZE)]).view(-1,1)
        #print(f'REWARDS : {rewards}')
        next_states=torch.FloatTensor([transitions[ii][3] for ii in range(BATCH_SIZE) if transitions[ii][3] is not None])
        #print(f'NEXT_STATES : {next_states}')
        mask=torch.BoolTensor([transitions[ii][3] is not None for ii in range(BATCH_SIZE)])
        #print(f'MASK : {mask}')
        
        #Q(s_i, a_i)
        values=self.q_net_codemaster(states)
        values=torch.gather(values, dim=1, index=actions)
        
        # max_a Q(s_{i+1}, a)
        values_next_states=torch.zeros(BATCH_SIZE)
        values_next_states=values_next_states.to(self.device)
        values_next_states[mask]=0
        values_next_states=values_next_states.view(-1,1)
        
        #targets y_i
        targets=rewards+GAMMA*values_next_states
        
        #print(f'>>> TARGETS : {targets}')
        #print(f'>>> VALUES : {values}')
        
        loss = objective(values, targets)
        
        # Optimize the model - UNCOMMENT!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss=loss.cpu()
        return loss.detach().numpy()

    def train(self):
        epsilon = EPSILON_START
        ep = 0
        total_time = 0
        losses = []
        # objective and optimizer
        objective = nn.MSELoss()
        optimizer = optim.SGD(params=self.q_net_codemaster.parameters(), lr=LEARNING_RATE)
        while ep < N_EPISODES:
            rewards = [0]
            word_to_guess = self.board[np.random.randint(0, len(self.board))]
            state = self.master_vectors[word_to_guess]
            action, proba = self.choose_clue(state, epsilon)
            chosen_distance, chosen_word = self.choose_word(self.board, self.guesser_vectors, self.clues[action])
            #print(f'Proba = {proba}')

            # take action and update replay buffer and networks
            next_state = None
            reward = 1 if chosen_word == word_to_guess else 0
            done = True
            loss = self.update(state, action, reward, next_state, done, optimizer, objective)
            if loss < np.inf :
                losses.append(loss)

            # update state
            #state = next_state

            # end episode if done
            if done:
                ep += 1
                if ( (ep+1)% EVAL_EVERY == 0):
                    rewards = self.eval_dqn()
                    print("episode =", ep+1, ", rewards = ", rewards)
                    if np.mean(rewards) >= REWARD_THRESHOLD:
                        break

                # decrease epsilon
                epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * \
                                np.exp(-1. * ep / DECREASE_EPSILON )    

            total_time += 1
        rewards = self.eval_dqn(20, test=True)
        print("")
        print("mean reward after training = ", np.mean(rewards))
        plt.plot(losses)
        plt.show()


"""     # Evaluate the final policy
    rewards = eval_dqn(20)
    print("")
    print("mean reward after training = ", np.mean(rewards))
    rewards = eval_dqn(20, training=True)
    print("")
    print("mean reward with training = ", np.mean(rewards)) """