import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

def get_q(states):
    """
    Compute Q function for a list of states
    """
    with torch.no_grad():
        states_v = torch.FloatTensor([states])
        output = q_net.forward(states_v).detach().numpy()  # shape (1, len(states), n_actions)
    return output[0, :, :]  # shape (len(states), n_actions)


# create network and target network
hidden_size = 128
obs_size = 4
n_actions = 10

q_net = Net(obs_size, hidden_size, n_actions)
q_net.load_state_dict(torch.load('models/risk_2p_600ep_norm'))

# Plot optimal phase diagram
V_opt = [[np.max(get_q([[i/9, j/9, 0.5, 0.5]])[0]) for i in range(9)] for j in range(9)]
X = np.linspace(0, 9, 9)
Y = np.linspace(0, 9, 9)

best_actions = [[np.argmax(get_q([[i/9, j/9, 0.5, 0.5]])[0])/10 for i in range(9)] for j in range(9)]

#plt.pcolor(X, Y, V_opt, shading='auto')
plt.pcolor(X, Y, np.transpose(best_actions), shading='auto')
plt.xlabel('Remaining Red Cards')
plt.ylabel('Remaining Blue Cards')
plt.title("Codenames risk level\nBest actions")
plt.colorbar()
#plt.savefig('mountain_car_phase_diagram.jpg') 
plt.show()