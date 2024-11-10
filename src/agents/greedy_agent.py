import copy
from agents.agent_interface import IAgent


class Greedy_Agent(IAgent):

    def __init__(self, player_id, team_id):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id

        self.hand = None
        self.is_starting_trick = None
        self.playing_suit = None

    def act(self, state) -> int:
        state = copy.deepcopy(state)
        self._interpret_state(state)

        if self.is_starting_trick:
            # Play lowest card
            card = self._get_lowest_card()
            self.hand.remove(card)
            card_idx = card.index
            return card_idx

        valid_hand = self._get_valid_hand()
        # If one card can take the trick, play this card
        for card in sorted(valid_hand, key=lambda c: c.rank.value, reverse=True):
            if state["trick"].can_take_trick(card):
                self.hand.remove(card)
                card_idx = card.index
                return card_idx
        
        # Otherwise play lowest valid card
        card = self._get_lowest_card()
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

    def load_model(self, loadpath: str):
        pass

    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass

    def _interpret_state(self, state):
        
        self.hand = state["hands"][f"P{self.player_id}"]
        leading_player_id = state["leading_player_id"]
        # play_style = state[2]
        
        if leading_player_id == self.player_id:
            self.is_starting_trick = True
            self.playing_suit = None
        else:
            self.is_starting_trick = False
            self.playing_suit = state["trick"].playing_suit

    def _get_valid_hand(self) -> list:
        valid_hand = [card for card in self.hand if card.get_suit() == self.playing_suit]
        return valid_hand if valid_hand else self.hand

    def _get_lowest_card(self):
        lowest_card = min(self.hand, key=lambda c: c.rank.value)
        return lowest_card