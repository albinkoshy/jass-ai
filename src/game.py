import random

from card import Card, Rank, Suit
from player import Player
from trick import Trick


class Game:
    """
    Defines game logic, turn system, trump selection and scoring.
    """

    def __init__(self, players: list[Player], starting_player_id: int):
        self.players = players  # Ordered by player_id ascending
        self.starting_player_id = starting_player_id
        self.leading_player_id = starting_player_id  # Starting player starts first trick
        """ 
        self.trump_suit = None
        self.top_down = False
        self.bottom_up = False 
        """

    def play_round(self) -> list[int]:
        # TODO Starting player decides which game variant to play (trump_suit, top_down or bottom_up, or schieben)
        # For now, always play bottom_up

        # TODO State of the game (e.g (N_CARDS, N_Players))
        state = None

        # Play 9 tricks
        for trick_idx in range(9):
            self.play_trick(trick_idx, state)

        # Count points
        points_per_player = []
        for p in self.players:
            points_per_player.append(p.count_points())
        # Last trick winner gets 5 additional points
        points_per_player[self.leading_player_id] += 5
        points_per_team = [points_per_player[0] + points_per_player[2], points_per_player[1] + points_per_player[3]]
        assert sum(points_per_player) == 157

        return points_per_team

    def play_trick(self, trick_idx: int, state):
        """
        Given leading_player_id, play one trick.
        """
        # Before the trick is played, print all players hand
        print(self.players[0])
        print(self.players[1])
        print(self.players[2])
        print(self.players[3])

        trick = Trick(self.leading_player_id)
        for i in range(4):
            current_turn = (self.leading_player_id + i) % 4
            current_player = self.players[current_turn]

            card = current_player.play_card(trick, state)
            trick.trick[f"P{current_turn}"] = card
            if i == 0:
                trick.set_suit(card=card)
                print(
                    f"Trick Nr. {trick_idx+1}. Lead Player is: P{current_turn}. Playing suit is {repr(trick.get_suit())}"
                )
            print(f"P{current_player.player_id} played {card}")

        trick_winner = trick.determine_trick_winner()
        print(f"Trick won by: {trick_winner}")
        self.leading_player_id = int(trick_winner[1])
        self.players[self.leading_player_id].append_won_trick(trick)

    def select_trump(self):
        """First player selects trump suit"""
        pass
