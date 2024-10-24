from agents.random_agent import Random_Agent
from envs.env import JassEnv


def play_game():

    # Player table
    #   P2
    # P3  P1
    #   P0
    players = {"P1": "random", "P2": "greedy", "P3": "random"}

    agent = Random_Agent(player_id=0, team_id=0)

    team0_points = 0
    team1_points = 0

    starting_player_id = 0
    # Play 12 rounds of Jass (Turniermodus) (Or till team reaches a specific number of points?)
    for round_idx in range(4):
        print(f"STARTING ROUND {round_idx + 1}")
        print(80 * "#")

        # Create game instance and play one round of Jass
        env = JassEnv(starting_player_id=starting_player_id, players=players)
        state = env.reset()
        agent.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            state = next_state

        team0_points += env.team0_points
        team1_points += env.team1_points

        print(f"Current point distribution -- Team 0: {team0_points}, Team 1: {team1_points}\n")
        starting_player_id = (starting_player_id + 1) % 4

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
