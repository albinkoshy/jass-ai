import copy
import  inquirer
from agents.agent_interface import IAgent
from envs.card import Suit


class Human_Agent(IAgent):

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
            card_idx = self._choose_card(self.hand)
            return card_idx

        valid_hand = self._get_valid_hand()
        card_idx = self._choose_card(valid_hand)
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

    def _choose_card(self, hand) -> int:
        questions = [
            inquirer.List(
                "card",
                message=f"Which card does P{self.player_id} want to play?",
                choices=[str(card) for card in hand],
            )
        ]
        answers = inquirer.prompt(questions)
        for c in hand:
            if str(c) == answers["card"]:
                card = c
                break
        
        self.hand.remove(card)
        card_idx = card.index
        return card_idx
    
    def choose_game_type(self, state, is_geschoben: bool = False) -> str:
        choices = ["TOP_DOWN", "BOTTOM_UP", "ROSE", "SCHILTE", "EICHEL", "SCHELLE"]
        if is_geschoben:
            choices.append("SCHIEBEN")
        questions = [
            inquirer.List(
                "game_type",
                message=f"Which game type does P{self.player_id} want to choose?",
                choices=choices,
            )
        ]
        answers = inquirer.prompt(questions)
        game_type = answers["game_type"]
        assert game_type in choices and type(game_type) == str
        return game_type
    