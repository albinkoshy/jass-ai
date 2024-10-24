""" Implements some utility functions. """

import os
import random

import numpy as np
import torch

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
