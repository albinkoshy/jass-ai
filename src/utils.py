""" Implements some utility functions. """

import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

from envs.deck import Deck

# Used to transform hand indices to hand
GLOBAL_DECK = Deck()
GLOBAL_DECK.order_deck()
ORDERED_CARDS = np.array(GLOBAL_DECK.cards)


def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    os.environ["PYTHONHASHSEED"] = str(seed)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory:
    """
    Implementation of the memory class.
    """

    def __init__(self, max_capacity: int, min_capacity: int = 200, device: torch.device = torch.device("cpu")):
        self.memory = deque([], maxlen=max_capacity)
        self.device = device
        self.min_capacity = min_capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int, split_transitions: bool = False):
        minibatch = random.sample(self.memory, batch_size)
        if split_transitions:
            # Get individual elements of the namedtuple(s) as lists
            states, actions, rewards, next_states, dones = [], [], [], [], []

            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

            minibatch = [states, actions, rewards, next_states, dones]

            # for i in range(len(minibatch)):
            #     minibatch[i] = torch.tensor(minibatch[i], dtype=torch.float32, device=self.device)
        return minibatch

    def __len__(self) -> int:
        return len(self.memory)

    def reset(self):
        self.memory.clear()

    def start_optimizing(self) -> bool:
        # Training starts when memory collected enough data.
        return self.__len__() >= self.min_capacity


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        torch.nn.init.constant_(m.bias, 0)


class Q_Net(nn.Module):
    """
    Fully connected neural network. This class implements a neural network with a variable number of hidden layers and hidden units.
    """

    def __init__(self, state_size, action_size, layer_size, hidden_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, layer_size))  # Input layer
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(layer_size, layer_size))  # Hidden layers
        self.output_layer = nn.Linear(layer_size, action_size)  # Output layer

        self.activation = nn.functional.relu
        self.apply(weights_init_)

    def forward(self, s: torch.Tensor):

        for layer in self.layers:
            s = self.activation(layer(s))
        return self.output_layer(s)
