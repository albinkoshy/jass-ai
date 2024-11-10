import copy
import os
import random
import argparse
from collections import deque
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from agents.random_agent import Random_Agent
from agents.greedy_agent import Greedy_Agent
from agents.dqn_agent import DQN_Agent
from envs.jassenv import JassEnv
import utils

utils.seed_everything(random.randint(1, 999999), deterministic=False)

PRINT_ENV = False

def train_agent(args):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Player table
    #   P2
    # P3  P1
    #   P0
    
    N_EPISODES = args.n_episodes
    
    percentage_above_epsilon_min: float = 0.8
    epsilon_min: float = 0.01
    epsilon_decay: float = epsilon_min ** (1 / (percentage_above_epsilon_min * N_EPISODES))
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    dqn_agent = DQN_Agent(player_id=0, team_id=0, epsilon_decay=epsilon_decay, gamma=1.0, tau=args.tau, lr=args.lr, device=device)
    
    players = [dqn_agent,
               Greedy_Agent(player_id=1, team_id=1),
               Greedy_Agent(player_id=2, team_id=0),
               Greedy_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players, print_globals=PRINT_ENV)
    starting_player_id = 0
    
    
    writer = SummaryWriter(os.path.join(args.log_dir, "tensorboard"))
    
    print("Training agent...")
    # Summary of used hyperparameters
    print("Hyperparameters:")
    print(f"    Tau: {args.tau}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Log directory: {args.log_dir}")
    
    # Training loop
    for episode in tqdm(range(N_EPISODES), desc="Training Episodes"):
        
        state = env.reset(starting_player_id=starting_player_id)
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
                reward = 0 # Reward for the current player (Reward is only given at the end of the trick)
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
        
        # Train the agent after each episode
        for player in players:
            loss = player.optimize_model()
            
            if episode % (N_EPISODES // 10000) == 0:
                if loss is not None:
                    writer.add_scalar(f"Loss/P{player.player_id}", loss, episode)
                    writer.add_scalar(f"Epsilon/P{player.player_id}", player.epsilon, episode)
        
        if episode % (N_EPISODES // 10) == 0:
            # Save the model
            for player in players:
                # Save to log_dir
                directory = os.path.join(args.log_dir, f"models/P{player.player_id}_{player.__class__.__name__}")
                player.save_model(name=f"dqn_agent_{episode}_{player.player_id}", directory=directory)
                
        starting_player_id = (starting_player_id + 1) % 4
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train agent')
    # Add arguments
    parser.add_argument('--n_episodes', type=int, default=10000, help='Number of episodes to train the agent')
    parser.add_argument('--tau', type=float, default=0.001, help='Soft update parameter for target network')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the adam optimizer')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save the logs')
    args = parser.parse_args()
    
    train_agent(args)
