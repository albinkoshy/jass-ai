from abc import ABC, abstractmethod

from envs.card import Card, Rank
from envs.trick import Trick


class IPlayer(ABC):
    """Interface for various kinds of Players. Players cannot be trained"""

    def __init__(self, player_id: int, team_id: int):
        self.player_id = player_id
        self.team_id = team_id
        self.hand = []
        self.won_tricks = []

    def __repr__(self) -> str:
        return f"P{self.player_id}'s hand: {self.hand}"

    def receive_cards(self, cards: list[Card]):
        self.hand.extend(cards)

        # Sort received hand
        self.sort_hand()

    def sort_hand(self):
        self.hand.sort(key=lambda card: (card.suit.value, card.rank.value))

    def append_won_trick(self, trick: Trick):
        won_trick = list(trick.trick.values())
        self.won_tricks.extend(won_trick)

    def count_points(self) -> int:
        # top-down
        top_down_scoring = {
            Rank.SECHS: 0,
            Rank.SIEBEN: 0,
            Rank.ACHT: 8,
            Rank.NEUN: 0,
            Rank.ZEHN: 10,
            Rank.UNDER: 2,
            Rank.OBER: 3,
            Rank.KOENIG: 4,
            Rank.ASS: 11,
        }

        total_points = 0
        for card in self.won_tricks:
            total_points += top_down_scoring[card.rank]

        # Reset won_tricks
        self.won_tricks = []
        return total_points

    @abstractmethod
    def play_card(self, trick) -> Card:
        """
        Player decides for a card to play.

        Returns:
            Card: The Card the player wants to play
        """
        pass
