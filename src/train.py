import copy
import os
import csv
import random
import argparse
from collections import deque
import torch
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from agents.random_agent import Random_Agent
from agents.greedy_agent import Greedy_Agent
from agents.dqn_agent import DQN_Agent
from agents.double_dqn_agent import Double_DQN_Agent
from envs.jassenv import JassEnv
import utils

PRINT_ENV = False

def train_agent(args):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # seed = random.randint(1, 999999)
    seed = args.seed
    utils.seed_everything(seed, deterministic=True)
    # Player table
    #   P2
    # P3  P1
    #   P0
    
    N_EPISODES = args.n_episodes
    LOG_EVERY = 100 if N_EPISODES > 1000 else 10
    SAVE_MODEL_EVERY = 20000 if N_EPISODES > 100000 else 2000
    
    percentage_above_epsilon_min: float = 0.6
    epsilon_min: float = 0.01
    epsilon_decay: float = epsilon_min ** (1 / (percentage_above_epsilon_min * N_EPISODES))
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    if args.agent == "dqn":
        AGENT_TYPE = args.agent
        agent = DQN_Agent(player_id=0, 
                            team_id=0,
                            hide_opponents_hands=args.hide_opponents_hands,
                            hidden_sizes=args.hidden_sizes,
                            activation=args.activation,
                            batch_size=args.batch_size,
                            epsilon_decay=epsilon_decay, 
                            gamma=args.gamma, 
                            tau=args.tau, 
                            lr=args.lr,
                            replay_memory_capacity=args.replay_buffer_size,
                            loss=args.loss,
                            device=device)
    elif args.agent == "double_dqn":
        AGENT_TYPE = args.agent
        agent = Double_DQN_Agent(player_id=0, 
                            team_id=0,
                            hide_opponents_hands=args.hide_opponents_hands,
                            hidden_sizes=args.hidden_sizes,
                            activation=args.activation,
                            batch_size=args.batch_size,
                            epsilon_decay=epsilon_decay, 
                            gamma=args.gamma, 
                            tau=args.tau, 
                            lr=args.lr,
                            replay_memory_capacity=args.replay_buffer_size,
                            loss=args.loss,
                            device=device)
    else:
        raise ValueError("Invalid agent type")
    
    players = [agent,
               Greedy_Agent(player_id=1, team_id=1),
               Greedy_Agent(player_id=2, team_id=0),
               Greedy_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players, print_globals=PRINT_ENV)
    starting_player_id = 0
    
    rewards_list = [deque([], maxlen=1000), deque([], maxlen=1000), deque([], maxlen=1000), deque([], maxlen=1000)]
    writer = SummaryWriter(os.path.join(args.log_dir, "tensorboard"))
    csv_file = open(os.path.join(args.log_dir, "log.csv"), mode='w')
    writer_csv = csv.DictWriter(csv_file, fieldnames=['player', 'episode', 'avg_reward', 'loss', 'epsilon'])
    writer_csv.writeheader()
    
    print("Training agent...")
    # Summary of used hyperparameters
    print("Hyperparameters:")
    print(f"    Number of episodes: {N_EPISODES}")
    print(f"    Hide opponents hands: {args.hide_opponents_hands}")
    print(f"    Hidden sizes: {args.hidden_sizes}")
    print(f"    Activation function: {args.activation}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Gamma: {args.gamma}")
    print(f"    Tau: {args.tau}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Replay buffer size: {args.replay_buffer_size}")
    print(f"    Loss function: {args.loss}")
    print(f"    Log directory: {args.log_dir}")
    print(f"    Seed: {args.seed}")
    
    # Training loop
    for episode in tqdm(range(1, N_EPISODES+1), desc="Training Episodes"):
        
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
                reward = rewards[current_turn] # Reward for the current player
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
            # Keep track of rewards for each player after each episode
            # rewards_list[player.player_id].append(env.rewards_per_player[player.player_id]/157) # Per player
            rewards_list[player.player_id].append(env.rewards_per_team[player.player_id % 2]/157) # Per team
            
            if episode % LOG_EVERY == 0:
                if loss is not None:
                    avg_reward = sum(rewards_list[player.player_id])/len(rewards_list[player.player_id])
                    writer.add_scalar(f"AVG_Reward/P{player.player_id}", avg_reward, episode)
                    writer.add_scalar(f"Loss/P{player.player_id}", loss, episode)
                    writer.add_scalar(f"Epsilon/P{player.player_id}", player.epsilon, episode)
                    
                    writer_csv.writerow({'player': f'P{player.player_id}', 'episode': episode, 'avg_reward': avg_reward, 'loss': loss, 'epsilon': player.epsilon})
        
        if episode % SAVE_MODEL_EVERY == 0:
            # Save the model
            for player in players:
                # Save to log_dir
                directory = os.path.join(args.log_dir, f"models/P{player.player_id}_{player.__class__.__name__}")
                hidden_sizes_hyphen = "-".join(map(str, args.hidden_sizes))
                player.save_model(name=f"{AGENT_TYPE}_{hidden_sizes_hyphen}_{episode}.pt", directory=directory)
                
        starting_player_id = (starting_player_id + 1) % 4
    
    csv_file.close()
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train agent against greedy players')
    
    # Add arguments
    parser.add_argument('--agent',
                        type=str,
                        choices=["dqn", "double_dqn"],
                        required=True,
                        help='type of agent to train')
    
    parser.add_argument('--n_episodes', 
                        type=int, 
                        default=10000, 
                        help='number of episodes to train the agent')
    
    parser.add_argument('--hide_opponents_hands',
                        action='store_true',
                        help='whether to hide opponents hands from the agent')
    
    parser.add_argument('--hidden_sizes', 
                        type=lambda s: [int(item) for item in s.split(',')], 
                        default="256,256,256", 
                        help='hidden sizes of the neural network, input comma separated without spaces (e.g. "256,256,256")')
    
    parser.add_argument('--activation',
                        type=str,
                        choices=["relu", "tanh", "sigmoid"],
                        default="relu",
                        help='activation function to use, either "relu", "tanh" or "sigmoid"')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=512, 
                        help='batch size for training the neural network')
    
    parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help='discount factor for the reward')
    
    parser.add_argument('--tau',
                        type=float,
                        default=0.00005,
                        help='soft update parameter for target network')
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.00005,
                        help='learning rate for the adam optimizer')
    
    parser.add_argument('--replay_buffer_size', 
                        type=int, 
                        default=50000, 
                        help='size of the replay buffer')
    
    parser.add_argument('--loss',
                        type=str,
                        choices=["smooth_l1", "mse"],
                        default="smooth_l1",
                        help='loss function to use, either "mse" or "smooth_l1"')
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs', 
                        help='directory to save the logs')
    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='seed for random number generators')
    
    args = parser.parse_args()
    train_agent(args)
