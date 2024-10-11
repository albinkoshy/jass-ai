from card import Card, Suit
from player import Player


class Greedy_Player(Player):
    """Greedy Player"""

    def __init__(self, player_id, team_id):
        super().__init__(player_id, team_id)

    def play_card(self, trick, state) -> Card:
        """
        Player decides for a card to play.

        Returns:
            Card: The Card the player wants to play
        """
        # Starting player
        if trick.is_starting():
            # Play lowest card
            card = self.play_lowest_card(hand=self.hand)
            return card

        # not starting
        valid_hand = self.get_valid_hand(trick.playing_suit)
        # If one card can take the trick, play this card
        for card in sorted(valid_hand, key=lambda c: c.rank.value, reverse=True):
            if trick.can_take_trick(card):
                self.hand.remove(card)
                return card

        # Otherwise play lowest valid card
        card = self.play_lowest_card(hand=valid_hand)
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

    def play_lowest_card(self, hand: list[Card]) -> Card:
        lowest_card = min(hand, key=lambda c: c.rank.value)
        self.hand.remove(lowest_card)
        return lowest_card
