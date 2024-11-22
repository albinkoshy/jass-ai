import copy
import os
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from utils import Q_Net, ReplayMemory, soft_update
from agents.agent_interface import IAgent
from envs.card import Suit


class Double_DQN_Agent(IAgent):
    
    def __init__(
        self,
        player_id,
        team_id,
        deterministic: bool = False,
        hidden_sizes: list[int] = [256, 256, 256],
        batch_size: int = 512,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99999,
        gamma: float = 1.0,
        tau: float = 5e-3,
        lr: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id
        self.deterministic = deterministic
        
        self.hand = None
        self.is_starting_trick = None
        self.playing_suit = None
        self.game_type = None

        self.state_size = 300
        self.action_size = 36
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size

        # Exploration rate (epsilon-greedy)
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay

        # Parameters
        self.gamma = gamma  # Discount rate (to weigh each trick equally set =1, since there are only 9 tricks in one round)
        self.tau = tau  # Soft update param
        self.lr = lr  # Optimizer learning rate
        
        self.device = device

        # Replay Memory
        self.memory = ReplayMemory(max_capacity=50000, min_capacity=self.batch_size, device=self.device)

        self.num_optimizations = 0

        self.network = Q_Net(self.state_size, self.action_size, self.hidden_sizes).eval().to(self.device)
        self.target_net = Q_Net(self.state_size, self.action_size, self.hidden_sizes).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

        #self.criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.SmoothL1Loss(reduction='sum', beta=0.01)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def act(self, state):
        state = copy.deepcopy(state)
        self._interpret_state(state)
        
        # Epsilon-greedy policy
        if random.random() > self.epsilon or self.deterministic:
            # Exploitation
            state_onehot_encoded = self._encode_state(state)
            state = torch.tensor(state_onehot_encoded, dtype=torch.float, device=self.device).reshape(1, -1)
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(state)
            
            # Mask invalid actions to -inf
            q_values = self._mask_invalid_actions(q_values)
            action = torch.argmax(q_values).item()
            return action
        else:
            # Exploration
            action = self._random_valid_action()
        return action

    def remember(self, state: dict, action: int, reward, next_state: dict, done):
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        if not self.memory.start_optimizing():
            return None
        
        self.network.train()
        self.target_net.eval()
        
        # Get samples from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, split_transitions=True)
        states_masks, next_states_masks = self._get_masks_for_optimization(states, next_states, dones) # Masks: True for invalid actions, False for valid actions

        states, actions, rewards, next_states, dones = self._preprocess_batch(states, actions, rewards, next_states, dones)
        
        q_values = self.network(states)
        q_values = torch.where(states_masks, -1e7, q_values)
        q_values = q_values.gather(dim=1, index=actions)  # Get Q-Values for the actions
        
        with torch.no_grad():
            max_actions = q_values.argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states)
            next_q_values = torch.where(next_states_masks, -1e7, next_q_values)
            next_q_values = next_q_values.gather(dim=1, index=max_actions)  # Double DQN: Decouple action selection and evaluation
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad() # Set all gradients (.grad) to zero to avoid accumulation
        loss.backward() # Compute gradients of the loss w.r.t. the parameters (stored in .grad)
        self.optimizer.step() # Perform gradient descent by adjusting the weights by its gradients stored in .grad

        # Update target network
        soft_update(self.network, self.target_net, self.tau)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_optimizations += 1
        
        return loss.item()
        
    def reset(self):
        self.hand = None
        self.is_starting_trick = None
        self.playing_suit = None
        self.game_type = None
        self.memory.reset()
        self.num_optimizations = 0
        self.network = Q_Net(self.state_size, self.action_size, self.hidden_sizes).eval().to(self.device)
        self.target_net = Q_Net(self.state_size, self.action_size, self.hidden_sizes).eval().to(self.device)
        self.target_net.load_state_dict(self.network.state_dict())

    def load_model(self, loadpath: str):
        self.network.load_state_dict(torch.load(loadpath, map_location=self.device, weights_only=True))
        self.network.eval()
        self.target_net.load_state_dict(torch.load(loadpath, map_location=self.device, weights_only=True))
        self.target_net.eval()

    def save_model(self, name: str, directory: str):
        if not os.path.isdir(directory):
            os.makedirs(directory)
            
        file_path = os.path.join(directory, name)
        torch.save(self.network.state_dict(), file_path)
    
    def _interpret_state(self, state):
        
        self.hand = state["hands"][f"P{self.player_id}"]
        leading_player_id = state["leading_player_id"]
        self.game_type = state["game_type"]
        
        if leading_player_id == self.player_id:
            self.is_starting_trick = True
            self.playing_suit = None
        else:
            self.is_starting_trick = False
            self.playing_suit = state["trick"].playing_suit

    def _encode_state(self, state) -> np.ndarray:
        
        hands_encoding = np.zeros((4, 36), dtype=int)
        player_id = self.player_id
        for i in range(4):
            for card in state["hands"][f"P{player_id}"]:
                hands_encoding[i, card.index] = 1
            player_id = (player_id + 1) % 4
            
        trick_encoding = np.zeros((4, 36), dtype=int)
        if state["trick"] is not None: # If game is not done (state["trick"] is None after game is done)
            for p, card in state["trick"].trick.items():
                if card is not None:
                    p_id = (int(p[1]) - self.player_id) % 4
                    trick_encoding[p_id, card.index] = 1
        
        leading_pid_encoding = np.zeros(4, dtype=int)
        if state["leading_player_id"] is not None: # If game is not done (state["leading_player_id"] is None after game is done)
            leading_pid = (state["leading_player_id"] - self.player_id) % 4
            leading_pid_encoding = np.eye(4, dtype=int)[leading_pid]
            
        if state["is_geschoben"] == True and state["game_type"] == "SCHIEBEN":
            raise ValueError("Game type SCHIEBEN cannot be set if is_geschoben is True")
        
        game_type_encoding = np.zeros(7, dtype=int)
        if state["game_type"] is not None: # If game type was set
            game_type_idx = utils.GAME_TYPES.index(state["game_type"]) # Either one of in utils.GAME_TYPES
            game_type_encoding = np.eye(7, dtype=int)[game_type_idx]
        
        is_geschoben_encoding = np.array([1 if state["is_geschoben"] else 0])
        
        hands_encoding = hands_encoding.flatten()
        trick_encoding = trick_encoding.flatten()
        
        state_encoding = np.concatenate([hands_encoding, trick_encoding, leading_pid_encoding, game_type_encoding, is_geschoben_encoding])
        
        return state_encoding
    
    def _mask_invalid_actions(self, q_values):
        if self.is_starting_trick:
            # Mask q_values for card that is not in hand
            masked_q_values = torch.ones_like(q_values) * -1e7
            hand_card_indices = [card.index for card in self.hand]
            masked_q_values[:, hand_card_indices] = q_values[:, hand_card_indices]
            return masked_q_values

        valid_hand = self._get_valid_hand(hand=self.hand, playing_suit=self.playing_suit, game_type=self.game_type)
        valid_hand_card_indices = [card.index for card in valid_hand]
        masked_q_values = torch.ones_like(q_values) * -1e7
        masked_q_values[:, valid_hand_card_indices] = q_values[:, valid_hand_card_indices]
        return masked_q_values
    
    def _get_valid_hand(self, hand, playing_suit, game_type) -> list:
        if game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, game_type)
        else:
            trump_suit = None
        
        valid_hand = [card for card in hand if card.suit == playing_suit or card.suit == trump_suit]
 
        if trump_suit and playing_suit != trump_suit and all(card.suit == trump_suit for card in valid_hand):
            return hand
        return valid_hand if valid_hand else hand
    
    def _random_valid_action(self) -> int:
        if self.is_starting_trick:
            card = random.choice(self.hand)
            self.hand.remove(card)
            card_idx = card.index
            return card_idx

        # Choose randomly from valid options
        valid_hand = self._get_valid_hand(hand=self.hand, playing_suit=self.playing_suit, game_type=self.game_type)

        card = random.choice(valid_hand)
        self.hand.remove(card)
        card_idx = card.index
        return card_idx
    
    def _get_masks_for_optimization(self, states, next_states, dones):
        states_masks = np.empty((self.batch_size, self.action_size), dtype=int)
        for i, state in enumerate(states):
            hand = state["hands"][f"P{self.player_id}"]
            leading_player_id = state["leading_player_id"]
            
            if leading_player_id == self.player_id:
                # Mask for card that is not in hand
                hand_card_indices = [card.index for card in hand]
                mask = np.ones((self.action_size,), dtype=int)
                mask[hand_card_indices] = 0
                states_masks[i] = mask
            else:
                # Mask for cards that are not in hand and do not follow the playing suit and are not trump
                playing_suit = state["trick"].playing_suit
                game_type = state["game_type"]
                valid_hand = self._get_valid_hand(hand=hand, playing_suit=playing_suit, game_type=game_type)
                valid_hand_card_indices = [card.index for card in valid_hand]
                mask = np.ones((self.action_size,), dtype=int)
                mask[valid_hand_card_indices] = 0
                states_masks[i] = mask
                
        next_states_masks = np.empty((self.batch_size, self.action_size), dtype=int)
        for j, next_state in enumerate(next_states):
            if dones[j]:
                # If game is done, mask all actions as invalid
                mask = np.ones((self.action_size,), dtype=int)
                next_states_masks[j] = mask
                continue
            
            hand = next_state["hands"][f"P{self.player_id}"]
            leading_player_id = next_state["leading_player_id"]
            
            if leading_player_id == self.player_id:
                # Mask for card that is not in hand
                hand_card_indices = [card.index for card in hand]
                mask = np.ones((self.action_size,), dtype=int)
                mask[hand_card_indices] = 0
                next_states_masks[j] = mask
            else:
                # Mask for cards that are not in hand and do not follow the playing suit and are not trump
                playing_suit = next_state["trick"].playing_suit
                game_type = state["game_type"]
                valid_hand = self._get_valid_hand(hand=hand, playing_suit=playing_suit, game_type=game_type)
                valid_hand_card_indices = [card.index for card in valid_hand]
                mask = np.ones((self.action_size,), dtype=int)
                mask[valid_hand_card_indices] = 0
                next_states_masks[j] = mask
          
        states_masks = torch.tensor(states_masks, dtype=torch.bool, device=self.device)
        next_states_masks = torch.tensor(next_states_masks, dtype=torch.bool, device=self.device)
        return states_masks, next_states_masks
    
    def _preprocess_batch(self, states, actions, rewards, next_states, dones):
        
        states_onehot_encoded = np.empty((self.batch_size, self.state_size))
        for i, state in enumerate(states):
            states_onehot_encoded[i] = self._encode_state(state)
        
        next_states_onehot_encoded = np.empty((self.batch_size, self.state_size))
        for j, next_state in enumerate(next_states):
            next_states_onehot_encoded[j] = self._encode_state(next_state)
            
        states = torch.tensor(states_onehot_encoded, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        next_states = torch.tensor(next_states_onehot_encoded, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).reshape(self.batch_size, -1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)
        dones = torch.tensor(dones, dtype=torch.int64, device=self.device).reshape(self.batch_size, -1)
        
        return states, actions, rewards, next_states, dones
    
    def choose_game_type(self, state, is_geschoben: str = False) -> str:
        
        assert state["game_type"] is None, "Game type should not have been set"
        state = copy.deepcopy(state)
        state["is_geschoben"] = is_geschoben
        
        n_game_types = 7 # All game types
        if is_geschoben:
            n_game_types = 6 # All game types except "SCHIEBEN"
        
        best_q_value = -1e7
        best_game_type = None
        self.network.eval()
        for i in range(n_game_types):
            game_type = utils.GAME_TYPES[i]
            state["game_type"] = game_type
            state_onehot_encoded = self._encode_state(state)
            state_tensor = torch.tensor(state_onehot_encoded, dtype=torch.float, device=self.device).reshape(1, -1)
            with torch.no_grad():
                q_values = self.network(state_tensor)
            max_q_value = torch.max(q_values).item()
            
            if max_q_value > best_q_value:
                best_q_value = max_q_value
                best_game_type = game_type
        
        return best_game_type
    