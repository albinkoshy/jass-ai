import copy
import random
from agents.agent_interface import IAgent
from envs.card import Suit


class Random_Agent(IAgent):

    def __init__(self, player_id, team_id):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id

        self.hand = None
        self.is_starting_trick = None
        self.playing_suit = None
        self.game_type = None

    def act(self, state) -> int:
        state = copy.deepcopy(state)
        self._interpret_state(state)

        if self.is_starting_trick:
            card = random.choice(self.hand)
            self.hand.remove(card)
            card_idx = card.index
            return card_idx

        # Choose randomly from valid options
        valid_hand = self._get_valid_hand()

        card = random.choice(valid_hand)
        self.hand.remove(card)
        card_idx = card.index
        return card_idx

    def remember(self, state, action, reward, next_state, done):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        self.hand = None
        self.is_starting_trick = None
        self.playing_suit = None
        self.game_type = None

    def load_model(self, loadpath: str):
        pass

    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass

    def _interpret_state(self, state):
        
        self.hand = state["hands"][f"P{self.player_id}"]
        leading_player_id = state["leading_player_id"]
        self.game_type = state["game_type"]
        
        if leading_player_id == self.player_id:
            self.is_starting_trick = True
            self.playing_suit = None
        else:
            self.is_starting_trick = False
            self.playing_suit = state["trick"].playing_suit

    def _get_valid_hand(self) -> list:
        if self.game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, self.game_type)
        else:
            trump_suit = None
        valid_hand = [card for card in self.hand if card.suit == self.playing_suit or card.suit == trump_suit]
 
        if trump_suit and self.playing_suit != trump_suit and all(card.suit == trump_suit for card in valid_hand):
            return self.hand
        return valid_hand if valid_hand else self.hand

    def choose_game_type(self, state, is_geschoben: bool = False) -> str:
        # Choose game type randomly
        if is_geschoben:
            game_type = random.choice(["TOP_DOWN", "BOTTOM_UP", "ROSE", "SCHILTE", "EICHEL", "SCHELLE"])
            return game_type
        
        game_type = random.choice(["TOP_DOWN", "BOTTOM_UP", "ROSE", "SCHILTE", "EICHEL", "SCHELLE", "SCHIEBEN"])
        return game_type
    