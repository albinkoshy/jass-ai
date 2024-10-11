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
    p1, p3 = Greedy_Player(player_id=1, team_id=1), Greedy_Player(player_id=3, team_id=1)
    players = [p0, p1, p2, p3]

    team0_points = 0
    team1_points = 0

    starting_player_id = 0
    # Play 12 rounds of Jass (Turniermodus) (Or till team reaches a specific number of points?)
    for round_idx in range(4):  # TODO: change range to 12
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

        points_per_team = game.play_round()
        team0_points += points_per_team[0]
        team1_points += points_per_team[1]
        print()
        print(f"Current point distribution -- Team 0: {team0_points}, Team 1: {team1_points}")

        starting_player_id = (starting_player_id + 1) % 4
        print()

    # Print team points
    print("GAME FINISHED!")
    print(f"Total Points -- Team 0: {team0_points}, Team 1: {team1_points}")
    if team0_points > team1_points:
        print("TEAM 0 WON!")
    elif team0_points < team1_points:
        print("TEAM 1 WON!")
    else:
        print("TIE!")


if __name__ == "__main__":
    play_game()
