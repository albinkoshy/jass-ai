import copy
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm

from agents.random_agent import Random_Agent
from agents.greedy_agent import Greedy_Agent
from agents.dqn_agent import DQN_Agent
from agents.double_dqn_agent import Double_DQN_Agent
from envs.jassenv import JassEnv
import utils

utils.seed_everything(random.randint(1, 999999), deterministic=False)

PRINT_ENV = False

def evaluate_agent(args):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    N_EPISODES = args.n_episodes
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    agent = Double_DQN_Agent(player_id=0, 
                             team_id=0,
                             deterministic=True,
                             hide_opponents_hands=True,
                             hidden_sizes=args.hidden_sizes,
                             activation="relu",
                             device=device) # Deterministic evaluation
    agent.load_model(args.model_path)
    
    players = [agent,
               Greedy_Agent(player_id=1, team_id=1),
               Greedy_Agent(player_id=2, team_id=0),
               Greedy_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players, print_globals=PRINT_ENV)
    starting_player_id = 0
    rewards_list_per_player = []
    rewards_list_per_team = []
    won_games_per_player = [0, 0, 0, 0]
    won_games_per_team = [0, 0]
    
    print(f"Evaluating agent {args.model_path}...")
    
    for episode in tqdm(range(1, N_EPISODES+1), desc="Evaluation of agent"):
        
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
        
        while not done:
            action = players[current_turn].act(state)
            new_state, rewards, done = env.step(action)
            current_turn = env.get_current_turn()
            state = copy.deepcopy(new_state)
        
        starting_player_id = (starting_player_id + 1) % 4
        
        rewards_list_per_player.append(env.rewards_per_player)
        rewards_list_per_team.append(env.rewards_per_team)
        
        winner_player = np.argmax(env.rewards_per_player)
        won_games_per_player[winner_player] += 1
        
        winner_team = np.argmax(env.rewards_per_team)
        won_games_per_team[winner_team] += 1

    print()
    print(f"Number of episodes: {len(rewards_list_per_player)}")
    print(f"Average points per player: {np.average(rewards_list_per_player, axis=0)}")
    print(f"Average points per team: {np.average(rewards_list_per_team, axis=0)}")
    print(f"Number of won games: {won_games_per_team}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate agent')
    # Add arguments
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True, 
                        help='path to the model to evaluate')
    
    parser.add_argument('--n_episodes', 
                        type=int, 
                        default=1000, 
                        help='number of episodes to evaluate the agent')
    
    parser.add_argument('--hidden_sizes', 
                        type=lambda s: [int(item) for item in s.split(',')], 
                        required=True,
                        help='hidden sizes of the neural network, input comma separated without spaces (e.g. "256,256,256")')
    
    args = parser.parse_args()
    
    evaluate_agent(args)