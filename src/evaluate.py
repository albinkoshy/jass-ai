import copy
import os
import random
import argparse
from collections import deque
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from agents.random_agent import Random_Agent
from agents.greedy_agent import Greedy_Agent
from agents.dqn_agent import DQN_Agent
from envs.jassenv import JassEnv
import utils

utils.seed_everything(random.randint(1, 999999), deterministic=False)

PRINT_ENV = False

def evaluate_agent(args):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    N_EPISODES = args.n_episodes
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    dqn_agent = DQN_Agent(player_id=0, team_id=0, deterministic=True, device=device)  # Deterministic evaluation
    dqn_agent.load_model(args.model_path)
    
    players = [dqn_agent,
               Greedy_Agent(player_id=1, team_id=1),
               Greedy_Agent(player_id=2, team_id=0),
               Greedy_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players, print_globals=PRINT_ENV)
    starting_player_id = 0
    rewards_list = []
    won_games = [0, 0, 0, 0]
    
    print("Evaluating agent...")
    
    for episode in tqdm(range(N_EPISODES), desc="Evaluation of agent"):
        
        state = env.reset(starting_player_id=starting_player_id)
        current_turn = env.get_current_turn()
        done = False
        
        while not done:
            action = players[current_turn].act(state)
            new_state, rewards, done = env.step(action)
            current_turn = env.get_current_turn()
            state = copy.deepcopy(new_state)
        
        starting_player_id = (starting_player_id + 1) % 4
        
        rewards_list.append(env.rewards)
        
        winner = np.argmax(env.rewards)
        won_games[winner] += 1

    print()
    print(f"Number of episodes: {len(rewards_list)}")
    print(f"Average points: {np.average(rewards_list, axis=0)}")
    print(f"Number of won games: {won_games}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate agent')
    # Add arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model to evaluate')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes to evaluate the agent')
    args = parser.parse_args()
    
    evaluate_agent(args)