import copy
from agents.random_agent import Random_Agent
from agents.greedy_agent import Greedy_Agent
from agents.human_agent import Human_Agent
from agents.dqn_agent import DQN_Agent
from envs.jassenv import JassEnv

N_EPISODES = 1000
PRINT_ENV = True

""" Test JassEnv and different agents """
    
# Player table
#   P2
# P3  P1
#   P0

# Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
players = [Human_Agent(player_id=0, team_id=0),
            Greedy_Agent(player_id=1, team_id=1),
            Greedy_Agent(player_id=2, team_id=0),
            Greedy_Agent(player_id=3, team_id=1)]

# Initialize the environment
env = JassEnv(players=players, print_globals=PRINT_ENV)
starting_player_id = 0

rewards_list_per_player = []
reward_list_per_team = []

for episode in range(1, N_EPISODES+1):

    state = env.reset(starting_player_id=starting_player_id)
    
    is_geschoben = False
    game_type = players[starting_player_id].choose_game_type(state=state)
    if game_type ==  "SCHIEBEN":
        is_geschoben = True
        team_mate_id = (starting_player_id + 2) % 4
        game_type = players[team_mate_id].choose_game_type(state=state, is_geschoben=True)
    
    env.set_game_type(game_type, is_geschoben)

    current_turn = env.get_current_turn()
    done = False

    # Keep track of state, action pair for each player
    state_action_pairs = {
        "P0": {"state": None, "action": None},
        "P1": {"state": None, "action": None},
        "P2": {"state": None, "action": None},
        "P3": {"state": None, "action": None}
    }

    while not done:
        
        if state_action_pairs[f"P{current_turn}"]["state"] is not None:
            state_ = copy.deepcopy(state_action_pairs[f"P{current_turn}"]["state"]) # The state before the current player played
            action = state_action_pairs[f"P{current_turn}"]["action"] # The action the current player played before
            reward = 0 # Reward for the current player
            next_state = copy.deepcopy(state) # The state after the current player played
            done = False # Done is only True at the end of the game
            players[current_turn].remember(state_, action, reward, next_state, done)
        
        action = players[current_turn].act(state) # The agent is NOT ALLOWED TO CHANGE THE STATE, do deep copy
        
        state_action_pairs[f"P{current_turn}"]["state"] = copy.deepcopy(state)
        state_action_pairs[f"P{current_turn}"]["action"] = action
        
        new_state, rewards, done = env.step(action) # The environment changes the state

        current_turn = env.get_current_turn()

        state = copy.deepcopy(new_state)
        
    # Remember the transition for all players
    for player in players:
        state_ = copy.deepcopy(state_action_pairs[f"P{player.player_id}"]["state"])
        action = state_action_pairs[f"P{player.player_id}"]["action"]
        reward = rewards[player.player_id]
        next_state = copy.deepcopy(state)
        done = True
        player.remember(state_, action, reward, next_state, done)
        
    starting_player_id = (starting_player_id + 1) % 4
    
    rewards_list_per_player.append(env.rewards_per_player)
    reward_list_per_team.append(env.rewards_per_team)

import numpy as np
print()
print(f"Number of episodes: {len(rewards_list_per_player)}")
print(f"Average points per player: {np.average(rewards_list_per_player, axis=0)}")
print(f"Average points per team: {np.average(reward_list_per_team, axis=0)}")