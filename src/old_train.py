import copy
import os
import numpy as np
import argparse
# ML libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from archived.old_random_agent import Random_Agent
from agents.old_dqn_agent import DQN_Agent
from archived._env import JassEnv
import utils

utils.seed_everything(99, deterministic=False)

NUM_EPISODES = 1000000

def train_agent(args):

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Player table
    #   P2
    # P3  P1
    #   P0

    agent = DQN_Agent(player_id=0, team_id=0, device=device)
    players = {"P1": "greedy", "P2": "greedy", "P3": "greedy"}
    starting_player_id = 0

    writer = SummaryWriter()
    for i in range(NUM_EPISODES):
        env = JassEnv(starting_player_id=starting_player_id, players=players)
        state = env.reset()
        done = False
        
        total_reward = 0
        while not done:
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
                
            state = copy.deepcopy(next_state)
        
        starting_player_id = (starting_player_id + 1) % 4

        if agent.memory.start_optimizing():
            loss = agent.optimize_model()
            writer.add_scalar("Loss", loss, i)
            print(f"Loss: {loss:.4f}")
            writer.add_scalar("Epsilon", agent.epsilon, i)

        writer.add_scalar("Total Reward per Round", total_reward, i)
        print(f"Total reward: {total_reward:.4f}")
        
        if i % 50000 == 0:
            directory = "./src/agents/models"
            if not os.path.isdir(directory):
                os.mkdir(directory)
            torch.save(agent.network.state_dict(), f"./src/agents/models/dqn_agent_{i}.pt")
            
    writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Jass Training")

    parser.add_argument("--out-dir", "-o", type=str, required=True, help="Where to save the output of the training.")
    parser.add_argument('--episodes', '-e', type=int, default=100000, help='How many episodes to train the model for.')

    parser.add_argument(
        '--evaluate-every',
        type=int,
        default=1000,
        help='After how many episodes to evaluate the model'
    )

    cli_args = parser.parse_args()
    train_agent(cli_args)
