from card import Card
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
        pass
