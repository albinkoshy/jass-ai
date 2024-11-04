import copy
import numpy as np
from agents.random_agent import Random_Agent
from agents.dqn_agent import DQN_Agent
from envs._env import JassEnv


def play_game():

    # Player table
    #   P2
    # P3  P1
    #   P0
    players = {"P1": "random", "P2": "greedy", "P3": "random"}

    team0_points = 0
    team1_points = 0

    agent = DQN_Agent(player_id=0, team_id=0)
    # agent = Random_Agent(player_id=0, team_id=0)
        
    starting_player_id = 0
    # Play 12 rounds of Jass (Turniermodus) (Or till team reaches a specific number of points?)
    for round_idx in range(12*100):
        print(f"STARTING ROUND {round_idx + 1}")
        print(80 * "#")

        # Create game instance and play one round of Jass
        env = JassEnv(starting_player_id=starting_player_id, players=players)
        state = env.reset()
        #agent.reset()
        done = False

        trick_count = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            if np.sum(next_state[0][0, :, :]) == 0:
                assert done
            
            agent.remember(state, action, reward, next_state, done)
            agent.optimize_model()
            
            state = copy.deepcopy(next_state)
            trick_count += 1

        team0_points += env.team0_points
        team1_points += env.team1_points

        # print(f"Current point distribution -- Team 0: {team0_points}, Team 1: {team1_points}\n")
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
