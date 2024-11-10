from envs.card import Card
from envs.players.player_interface import IPlayer

# import inquirer # for input selection from user


class Human_Player(IPlayer):

    def __init__(self, player_id, team_id):
        super().__init__(player_id, team_id)

    def play_card(self, trick) -> Card:
        """
        Player decides for a card to play.

        Returns:
            Card: The Card the player wants to play
        """
        pass
