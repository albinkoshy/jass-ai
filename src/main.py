from deck import Deck
from game import Game
from players.greedy_player import Greedy_Player
from players.human_player import Human_Player
from players.random_player import Random_Player


def play_game():
    """
    Main game loop
    """

    # Player table
    #   P2
    # P3  P1
    #   P0
    # Create Players
    p0, p2 = Random_Player(player_id=0, team_id=0), Random_Player(player_id=2, team_id=0)
    p1, p3 = Random_Player(player_id=1, team_id=1), Random_Player(player_id=3, team_id=1)
    players = [p0, p1, p2, p3]

    starting_player_id = 0
    # Play 12 rounds of Jass (Turniermodus) (Or till team reaches a specific number of points?)
    for round_idx in range(4):  # TODO: change range to 4
        print(f"STARTING ROUND {round_idx + 1}")
        print(80 * "#")
        # Create card deck and shuffle
        deck = Deck()
        deck.shuffle_deck()

        # Distribute cards to players
        p0.receive_cards(deck.pop_cards(n_cards=9))
        p1.receive_cards(deck.pop_cards(n_cards=9))
        p2.receive_cards(deck.pop_cards(n_cards=9))
        p3.receive_cards(deck.pop_cards(n_cards=9))

        # Create game instance and play one round of Jass
        game = Game(players=players, starting_player_id=starting_player_id)
        game.play_round()

        starting_player_id = (starting_player_id + 1) % 4
        print()

    # Print team points
    # TODO


if __name__ == "__main__":
    play_game()
