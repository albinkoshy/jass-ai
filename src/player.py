from abc import ABC, abstractmethod

from card import Card


class Player(ABC):
    """Interface for various kinds of Players"""

    def __init__(self, player_id, team_id):
        self.player_id = player_id
        self.team_id = team_id
        self.hand = []

    def __repr__(self) -> str:
        return f"P{self.player_id}'s hand: {self.hand}"

    def receive_cards(self, cards):
        self.hand.extend(cards)

        # Sort received hand
        self.sort_hand()

    def sort_hand(self):
        self.hand.sort(key=lambda card: (card.suit.value, card.rank.value))

    @abstractmethod
    def play_card(self, trick, state) -> Card:
        """
        Player decides for a card to play.

        Returns:
            Card: The Card the player wants to play
        """
        pass
