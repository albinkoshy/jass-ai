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
    LOG_EVERY = 100 if N_EPISODES > 1000 else 10
    SAVE_MODEL_EVERY = 20000 if N_EPISODES > 100000 else 2000
    
    percentage_above_epsilon_min: float = 0.5
    epsilon_min: float = 0.01
    epsilon_decay: float = epsilon_min ** (1 / (percentage_above_epsilon_min * N_EPISODES))
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    dqn_agent = DQN_Agent(player_id=0, 
                          team_id=0, 
                          hidden_sizes=args.hidden_sizes,
                          epsilon_decay=epsilon_decay, 
                          gamma=args.gamma, 
                          tau=args.tau, 
                          lr=args.lr, 
                          device=device)
    
    players = [dqn_agent,
               Greedy_Agent(player_id=1, team_id=1),
               Greedy_Agent(player_id=2, team_id=0),
               Greedy_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players, print_globals=PRINT_ENV)
    starting_player_id = 0
    
    rewards_list = [deque([], maxlen=1000), deque([], maxlen=1000), deque([], maxlen=1000), deque([], maxlen=1000)]
    writer = SummaryWriter(os.path.join(args.log_dir, "tensorboard"))
    
    print("Training agent...")
    # Summary of used hyperparameters
    print("Hyperparameters:")
    print(f"    Number of episodes: {N_EPISODES}")
    print(f"    Hidden sizes: {args.hidden_sizes}")
    print(f"    Gamma: {args.gamma}")
    print(f"    Tau: {args.tau}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Log directory: {args.log_dir}")
    
    # Training loop
    for episode in tqdm(range(1, N_EPISODES+1), desc="Training Episodes"):
        
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
                reward = rewards[current_turn] # Reward for the current player
                next_state = copy.deepcopy(state) # The state after the current player played
                done = False # Done is only True at the end of the game
                players[current_turn].remember(state_, action, reward, next_state, done)

            action = players[current_turn].act(state) # The agent is NOT ALLOWED TO CHANGE THE STATE, do deep copy
            
            state_action_pairs[f"P{current_turn}"]["state"] = copy.deepcopy(state)
            state_action_pairs[f"P{current_turn}"]["action"] = action
            
            new_state, rewards, done = env.step(action) # The environment changes the state
            #print(f"Immediate rewards: {rewards}. Total rewards: {env.rewards}")
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
            # Keep track of rewards for each player after each episode
            rewards_list[player.player_id].append(env.rewards[player.player_id]/157)
            
            if episode % LOG_EVERY == 0:
                if loss is not None:
                    avg_reward = sum(rewards_list[player.player_id])/len(rewards_list[player.player_id])
                    writer.add_scalar(f"AVG_Reward/P{player.player_id}", avg_reward, episode)
                    writer.add_scalar(f"Loss/P{player.player_id}", loss, episode)
                    writer.add_scalar(f"Epsilon/P{player.player_id}", player.epsilon, episode)
        
        if episode % SAVE_MODEL_EVERY == 0:
            # Save the model
            for player in players:
                # Save to log_dir
                directory = os.path.join(args.log_dir, f"models/P{player.player_id}_{player.__class__.__name__}")
                player.save_model(name=f"dqn_agent_{episode}.pt", directory=directory)
                
        starting_player_id = (starting_player_id + 1) % 4
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train agent')
    
    # Add arguments
    parser.add_argument('--n_episodes', 
                        type=int, 
                        default=10000, 
                        help='number of episodes to train the agent')
    
    parser.add_argument('--hidden_sizes', 
                        type=lambda s: [int(item) for item in s.split(',')], 
                        default="256,256,256", 
                        help='hidden sizes of the neural network, input comma separated without spaces (e.g. "256,256,256")')
    
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discount factor for the reward')
    
    parser.add_argument('--tau',
                        type=float,
                        default=0.00005,
                        help='soft update parameter for target network')
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.00005,
                        help='learning rate for the adam optimizer')
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs', 
                        help='directory to save the logs')
    
    args = parser.parse_args()
    train_agent(args)
