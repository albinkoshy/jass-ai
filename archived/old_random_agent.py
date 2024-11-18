import random

import numpy as np

import utils
from agents.agent_interface import IAgent
from envs.card import Card, Suit


class Random_Agent(IAgent):
    """Random agent, only to test environment - agent implementation"""

    def __init__(self, player_id, team_id):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id

        self.hand_card_indices = None
        self.is_starting_trick = None
        self.playing_suit = None

    def act(self, state) -> int:
        self._interpret_state(state)

        if self.is_starting_trick:
            card_idx = random.choice(self.hand_card_indices)
            self.hand_card_indices.remove(card_idx)
            return card_idx

        # Choose randomly from valid options
        valid_hand_card_indices = self._get_valid_hand_card_indices()

        card_idx = random.choice(valid_hand_card_indices)
        self.hand_card_indices.remove(card_idx)
        return card_idx

    def remember(self, state, action, reward, next_state, done):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        self.hand_card_indices = None
        self.is_starting_trick = None
        self.playing_suit = None

    def load_model(self, loadpath: str):
        pass

    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass

    def _interpret_state(self, state):
        card_distribution = state[0]
        leading_player_id = state[1] # Player who played the first card in the trick
        play_style = state[2]
        
        self.hand_card_indices = np.where(card_distribution[0, 0, :] == 1)[0].tolist()
        if leading_player_id == 0:
            self.is_starting_trick = True
            self.playing_suit = None
        else:
            leading_card_idx = np.where(card_distribution[1, leading_player_id, :] == 1)[0]
            assert len(leading_card_idx) == 1
            leading_card_idx = leading_card_idx[0]
            self.is_starting_trick = False
            self.playing_suit = utils.ORDERED_CARDS[leading_card_idx].suit

    def _get_valid_hand_card_indices(self) -> list:
        if self.playing_suit == Suit.ROSE:  # (0, 8)
            valid_hand_card_indices = [idx for idx in self.hand_card_indices if 0 <= idx <= 8]
        elif self.playing_suit == Suit.SCHILTE:  # (9, 17)
            valid_hand_card_indices = [idx for idx in self.hand_card_indices if 9 <= idx <= 17]
        elif self.playing_suit == Suit.EICHEL:  # (18, 26)
            valid_hand_card_indices = [idx for idx in self.hand_card_indices if 18 <= idx <= 26]
        elif self.playing_suit == Suit.SCHELLE:  # (27, 35)
            valid_hand_card_indices = [idx for idx in self.hand_card_indices if 27 <= idx <= 35]
        else:
            raise Exception("Unknown Suit")

        return valid_hand_card_indices if valid_hand_card_indices else self.hand_card_indices
