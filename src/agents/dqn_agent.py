import copy
import os
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from agents.agent_interface import IAgent
from envs.card import Card, Suit
from utils import Q_Net, ReplayMemory, soft_update


class DQN_Agent(IAgent):

    def __init__(
        self,
        player_id,
        team_id,
        state_size: int = 443,
        action_size: int = 36,
        hidden_size: int = 256,
        hidden_layers: int = 3,
        batch_size: int = 128,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999995,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id
        
        self.hand_card_indices = None
        self.is_starting_trick = None
        self.playing_suit = None

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        # Exploration rate (epsilon-greedy)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay

        self.device = device

        # Replay Memory
        self.memory = ReplayMemory(max_capacity=50000, min_capacity=200, device=self.device)

        # Parameters
        self.gamma = 1  # Discount rate (to weigh each trick equally, since there are only 9 tricks in one round)
        self.tau = 1e-3  # Soft update param
        self.lr = 0.01  # Optimizer learning rate

        self.num_optimizations = 0

        self.network = Q_Net(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net = Q_Net(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def act(self, state) -> int:
        self._interpret_state(state)
        card_distribution = state[0]
        leading_player_id = state[1]  # Player who played the first card in the trick
        play_style = state[2]
        
        flattened_state = np.concatenate([np.ravel(card_distribution), np.eye(4)[leading_player_id], np.eye(7)[play_style]])

        # Epsilon-greedy policy
        if random.random() > self.epsilon:
            # Exploitation
            state = torch.tensor(flattened_state, dtype=torch.float, device=self.device).reshape(1, -1)
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(state)
            self.network.train()
            
            # Mask invalid actions to -inf
            q_values = self._mask_invalid_actions(q_values)
            action = torch.argmax(q_values).item()
            return action
        else:
            # Exploration
            action = self._random_valid_action()
        return action

    def remember(self, state: list, action: int, reward: int, next_state: list, done: bool):
        state = copy.deepcopy(state)
        next_state = copy.deepcopy(next_state)
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if not self.memory.start_optimizing():
            return None
        print(f"Optimizing model {self.num_optimizations}")
        self.network.train()
        self.target_net.train()

        # Get samples from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)
        states_masks, next_states_masks = self._get_masks_for_optimization(states, next_states, dones) # Masks: True for invalid actions, False for valid actions
        
        states, actions, rewards, next_states, dones = self._preprocess_batch(states, actions, rewards, next_states, dones)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).detach()  # Detach since no gradient calc needed
            next_q_values = torch.where(next_states_masks, -1e7, next_q_values).max(dim=1, keepdim=True)[0]  # Get max Q-Values for the next_states.
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        q_values = self.network(states)
        q_values = torch.where(states_masks, -1e7, q_values)
        q_values = q_values.gather(dim=1, index=actions)  # Get Q-Values for the actions

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.network, self.target_net, self.tau)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_optimizations += 1
        
        return loss.item()

    def reset(self):
        self.hand_card_indices = None
        self.is_starting_trick = None
        self.playing_suit = None
        self.memory.reset()
        self.num_optimizations = 0
        self.network = Q_Net(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net = Q_Net(self.state_size, self.action_size, self.hidden_size, self.hidden_layers).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

    def load_model(self, loadpath: str):
        pass

    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass

    def _interpret_state(self, state):
        card_distribution = state[0]
        leading_player_id = state[1]  # Player who played the first card in the trick
        play_style = state[2]
        
        self.hand_card_indices = np.where(card_distribution[0, 0, :] == 1)[0].tolist()
        if leading_player_id == 0:
            self.is_starting_trick = True
            self.playing_suit = None
        else:
            leading_card_idx = np.where(card_distribution[1, leading_player_id, :] == 1)[0].tolist()
            assert len(leading_card_idx) == 1
            leading_card_idx = leading_card_idx[0]
            self.is_starting_trick = False
            self.playing_suit = utils.ORDERED_CARDS[leading_card_idx].get_suit()

    def _get_valid_hand_card_indices(self, hand_card_indices, playing_suit) -> list:
        if playing_suit == Suit.ROSE:  # (0, 8)
            valid_hand_card_indices = [idx for idx in hand_card_indices if 0 <= idx <= 8]
        elif playing_suit == Suit.SCHILTE:  # (9, 17)
            valid_hand_card_indices = [idx for idx in hand_card_indices if 9 <= idx <= 17]
        elif playing_suit == Suit.EICHEL:  # (18, 26)
            valid_hand_card_indices = [idx for idx in hand_card_indices if 18 <= idx <= 26]
        elif playing_suit == Suit.SCHELLE:  # (27, 35)
            valid_hand_card_indices = [idx for idx in hand_card_indices if 27 <= idx <= 35]
        else:
            raise Exception("Unknown Suit")

        return valid_hand_card_indices if valid_hand_card_indices else hand_card_indices

    def _random_valid_action(self):
        if self.is_starting_trick:
            card_idx = random.choice(self.hand_card_indices)
            self.hand_card_indices.remove(card_idx)
            return card_idx

        # Choose randomly from valid options
        valid_hand_card_indices = self._get_valid_hand_card_indices(self.hand_card_indices, self.playing_suit)

        card_idx = random.choice(valid_hand_card_indices)
        self.hand_card_indices.remove(card_idx)
        return card_idx

    def _mask_invalid_actions(self, q_values):
        if self.is_starting_trick:
            # Mask q_values for card that is not in hand
            masked_q_values = torch.ones_like(q_values) * -1e7
            masked_q_values[:, self.hand_card_indices] = q_values[:, self.hand_card_indices]
            return masked_q_values

        valid_hand_card_indices = self._get_valid_hand_card_indices(self.hand_card_indices, self.playing_suit)
        masked_q_values = torch.ones_like(q_values) * -1e7
        masked_q_values[:, valid_hand_card_indices] = q_values[:, valid_hand_card_indices]
        return masked_q_values
    
    def _get_masks_for_optimization(self, states, next_states, dones) -> tuple[list, list]:
        states_masks = np.empty((self.batch_size, self.action_size), dtype=int)
        for i, state in enumerate(states):
            card_distribution = state[0]
            leading_player_id = state[1]
            play_style = state[2]
            
            hand_card_indices = np.where(card_distribution[0, 0, :] == 1)[0].tolist()
            if leading_player_id == 0:
                # Mask for card that is not in hand
                mask = np.ones((self.action_size,), dtype=int)
                mask[hand_card_indices] = 0
                states_masks[i] = mask
            else:
                leading_card_idx = np.where(card_distribution[1, leading_player_id, :] == 1)[0].tolist()
                assert len(leading_card_idx) == 1
                leading_card_idx = leading_card_idx[0]
                playing_suit = utils.ORDERED_CARDS[leading_card_idx].get_suit()
            
                valid_hand_card_indices = self._get_valid_hand_card_indices(hand_card_indices, playing_suit)
                mask = np.ones((self.action_size,), dtype=int)
                mask[valid_hand_card_indices] = 0
                states_masks[i] = mask
        
        next_states_masks = np.empty((self.batch_size, self.action_size), dtype=int)
        for j, state in enumerate(next_states):
            if dones[j]:
                # If game is done, mask all actions as invalid
                mask = np.ones((self.action_size,), dtype=int)
                next_states_masks[j] = mask
                continue
            
            card_distribution = state[0]
            leading_player_id = state[1]
            play_style = state[2]
            
            hand_card_indices = np.where(card_distribution[0, 0, :] == 1)[0].tolist()
            if leading_player_id == 0:
                # Mask for card that is not in hand
                mask = np.ones((self.action_size,), dtype=int)
                mask[hand_card_indices] = 0
                next_states_masks[j] = mask
            else:
                leading_card_idx = np.where(card_distribution[1, leading_player_id, :] == 1)[0].tolist()
                assert len(leading_card_idx) == 1
                leading_card_idx = leading_card_idx[0]
                playing_suit = utils.ORDERED_CARDS[leading_card_idx].get_suit()
            
                valid_hand_card_indices = self._get_valid_hand_card_indices(hand_card_indices, playing_suit)
                mask = np.ones((self.action_size,), dtype=int)
                mask[valid_hand_card_indices] = 0
                next_states_masks[j] = mask
                
        states_masks = torch.tensor(states_masks, dtype=torch.bool, device=self.device)
        next_states_masks = torch.tensor(next_states_masks, dtype=torch.bool, device=self.device)
        return states_masks, next_states_masks
    
    def _preprocess_batch(self, states, actions, rewards, next_states, dones) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened_states = np.empty((self.batch_size, self.state_size), dtype=np.float32)
        for i, state in enumerate(states):
            flattened_state = np.concatenate([np.ravel(state[0]), np.eye(4)[state[1]], np.eye(7)[state[2]]])
            flattened_states[i] = flattened_state
        
        states = torch.tensor(flattened_states, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        
        flattened_next_states = np.empty((self.batch_size, self.state_size), dtype=np.float32)
        for j, state in enumerate(next_states):
            flattened_next_state = np.concatenate([np.ravel(state[0]), np.eye(4)[state[1]], np.eye(7)[state[2]]])
            flattened_next_states[j] = flattened_next_state
            
        next_states = torch.tensor(flattened_next_states, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).reshape(self.batch_size, -1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        
        return states, actions, rewards, next_states, dones