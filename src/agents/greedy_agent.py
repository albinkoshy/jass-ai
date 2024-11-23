import copy
import random
from agents.agent_interface import IAgent
from envs.card import Suit


class Greedy_Agent(IAgent):

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

        if self.game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, self.game_type)
        else:
            trump_suit = None
            
        if self.is_starting_trick:
            # Play highest card if leading trick
            return self._play_highest_card(hand=self.hand)

        valid_hand = [card for card in self.hand if card.suit == self.playing_suit]
        if valid_hand:
            # Play highest card if it can take the trick
            highest_card = max(valid_hand, key=lambda c: c.rank.value)
            if state["trick"].can_take_trick(highest_card):
                return self._play_highest_card(hand=valid_hand)
            else:
                return self._play_lowest_card(hand=valid_hand)
        else:
            # Play highest trump card if in hand
            trump_cards = [card for card in self.hand if card.suit == trump_suit]
            if trump_cards:
                return self._play_highest_card(hand=trump_cards)
            else: 
                # Play lowest card if no trump cards
                return self._play_lowest_card(hand=self.hand)

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
    
    def _play_highest_card(self, hand) -> int:
        card = max(hand, key=lambda c: c.rank.value)
        self.hand.remove(card)
        card_idx = card.index
        return card_idx
    
    def _play_lowest_card(self, hand) -> int:
        card = min(hand, key=lambda c: c.rank.value)
        self.hand.remove(card)
        card_idx = card.index
        return card_idx
    
    def choose_game_type(self, state, is_geschoben: bool = False) -> str:
        
        # Allow some randomness for choosing "SCHIEBEN"
        if not is_geschoben and random.random() < 0.14:
            return "SCHIEBEN"
        
        state = copy.deepcopy(state)
        hand = state["hands"][f"P{self.player_id}"]
        
        # Choose top-down if many high cards
        high_cards = [card for card in hand if card.rank.value > 10]
        if len(high_cards) > 5:
            return "TOP_DOWN"
        
        # Choose bottom-up if many low cards
        low_cards = [card for card in hand if card.rank.value <= 9]
        if len(low_cards) > 5:
            return "BOTTOM_UP"
        
        # Choose trump for which it has the most cards
        suits = [card.suit for card in hand]
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        max_count = max(suit_counts.values())
        max_suits = [suit for suit, count in suit_counts.items() if count == max_count]
        
        trump = random.choice(max_suits)

        if trump == Suit.ROSE:
            game_type = "ROSE"
        elif trump == Suit.SCHILTE:
            game_type = "SCHILTE"
        elif trump == Suit.EICHEL:
            game_type = "EICHEL"
        elif trump == Suit.SCHELLE:
            game_type = "SCHELLE"
        else:
            raise ValueError("Suit not recognized")

        return game_type