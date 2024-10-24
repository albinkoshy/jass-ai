import random

from envs.card import Card, Suit
from envs.players.player_interface import IPlayer


class Random_Player(IPlayer):

    def __init__(self, player_id, team_id):
        super().__init__(player_id, team_id)

    def play_card(self, trick) -> Card:
        """
        Player decides for a card to play.

        Returns:
            Card: The Card the player wants to play
        """
        # Starting player
        if trick.is_starting():
            # Play random card from deck
            card = random.choice(self.hand)
            self.hand.remove(card)
            return card
        # not starting
        # Choose randomly from valid options
        valid_hand = self.get_valid_hand(trick.playing_suit)
        card = random.choice(valid_hand)
        self.hand.remove(card)
        return card

    def get_valid_hand(self, playing_suit: Suit) -> list:
        """
        Given self.hand and playing_suit, return set of valid cards to play

        Args:
            playing_suit (Suit): Suit being played

        Returns:
            list: list of valid Cards to play
        """
        valid_hand = [card for card in self.hand if card.get_suit() == playing_suit]
        return valid_hand if valid_hand else self.hand
