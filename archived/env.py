import gymnasium as gym
import numpy as np
from gymnasium import spaces

import utils
from envs.card import Card
from envs.deck import Deck
from envs.players.greedy_player import Greedy_Player
from envs.players.random_player import Random_Player
from envs.trick import Trick


class JassEnv(gym.Env):
    """
    The Game environment for RL training
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, starting_player_id: int, players: dict = {"P1": "greedy", "P2": "greedy", "P3": "greedy"}):

        assert set(players.keys()) == {"P1", "P2", "P3"}, "Players dictionary must contain exactly the keys 'P1', 'P2', 'P3'"
        assert all(player_type in ["greedy", "random"] for player_type in players.values()), "Unknown player values"

        self.agent_hand = []
        self.p1 = self._create_player(player_type=players["P1"], player_id=1, team_id=1)  # Opponent
        self.p2 = self._create_player(player_type=players["P2"], player_id=2, team_id=0)  # Partner
        self.p3 = self._create_player(player_type=players["P3"], player_id=3, team_id=1)  # Opponent
        self.players = [self.p1, self.p2, self.p3]

        self.starting_player_id = starting_player_id
        self.leading_player_id = starting_player_id  # Starting player starts first trick
        self.current_turn = starting_player_id

        self.team0_points = 0
        self.team1_points = 0

        self.state = None
        self.trick = None

        print(f"P0: agent\nP1: {players['P1']}\nP2: {players['P2']}\nP3: {players['P3']}")
        print(f"P{self.starting_player_id} is starting the round")

    def reset(self):

        self.team0_points = 0
        self.team1_points = 0

        self.state = None
        self.trick = None

        # Create card deck and shuffle
        deck = Deck()

        self.agent_hand = deck.pop_cards(n_cards=9)
        self.agent_hand.sort(key=lambda card: (card.suit.value, card.rank.value))

        self.p1.receive_cards(deck.pop_cards(n_cards=9))
        self.p2.receive_cards(deck.pop_cards(n_cards=9))
        self.p3.receive_cards(deck.pop_cards(n_cards=9))

        print(f"P0's hand: {self.agent_hand}")
        print(self.p1)
        print(self.p2)
        print(self.p3)

        # Create initial state based on card distribution
        self.state = self._get_inital_state()

        self.trick = Trick(self.starting_player_id)
        # If agent is starting, return state and let agent begin
        if self.starting_player_id == 0:
            return self.state

        # Play until it is the agent's turn
        self._play_init_trick()

        return self.state

    def step(self, action: int) -> tuple[np.ndarray, int, bool]:
        """
        Agents acts in the environment

        Args:
            action (int): Card number (from set {0, ..., 35}) the agent plays

        Returns:
            tuple[np.ndarray, int, bool]: Returns (next_state, reward, done)
        """

        # Update state using action
        self._update_state_after_play(action=action, player_id=0)
        agent_card = utils.ORDERED_CARDS[action]
        self.agent_hand.remove(agent_card)
        self.trick.trick[f"P{self.current_turn}"] = agent_card
        if self.leading_player_id == 0:
            assert self.current_turn == 0
            self.trick.set_suit(card=agent_card)
            print(f"Lead Player is: P{self.leading_player_id}. Playing suit is {repr(self.trick.playing_suit)}")
        print(f"P{self.current_turn} played {agent_card}")
        self.current_turn += 1

        # Continue Trick
        while any(value is None for value in self.trick.trick.values()):
            assert self.current_turn <= 3, "self.current_turn cannot be larger than 3"
            current_player = self.players[self.current_turn - 1]
            player_card = current_player.play_card(self.trick)
            self.trick.trick[f"P{self.current_turn}"] = player_card
            print(f"P{current_player.player_id} played {player_card}")

            self._update_state_after_play(action=player_card.index, player_id=self.current_turn)

            self.current_turn += 1

        trick_winner = self.trick.determine_trick_winner()
        print(f"Trick won by: {trick_winner}")
        self.leading_player_id = int(trick_winner[1])
        self.current_turn = self.leading_player_id

        self._update_state_after_trick_ends()

        reward = 0
        # If agent won trick: reward = trickpoints
        if self.leading_player_id == 0:
            reward = self.trick.get_trick_points()
            self.team0_points += reward
        # If partner won trick: reward = trickpoints
        elif self.leading_player_id == 2:
            reward = self.trick.get_trick_points()
            self.team0_points += reward
            self.players[self.leading_player_id - 1].append_won_trick(self.trick)
        else:
            self.team1_points += self.trick.get_trick_points()
            self.players[self.leading_player_id - 1].append_won_trick(self.trick)

        # Determine if game is done
        done = False
        if np.all(self.state[0, :, :] == 0):
            done = True

        # If not done, start next trick
        if not done:

            print(f"P0's hand: {self.agent_hand}")
            print(self.p1)
            print(self.p2)
            print(self.p3)

            self.trick = Trick(self.leading_player_id)

            # If agent is leading, return state and let agent begin
            if self.leading_player_id == 0:
                return self.state, reward, done

            # Play until it is the agent's turn
            self._play_init_trick()
        # If done, count +5 points for the last trick winner
        else:
            if self.leading_player_id == 0 or self.leading_player_id == 2:
                self.team0_points += 5
            else:
                self.team1_points += 5

        return self.state, reward, done

    def render(self):
        pass

    def close(self):
        pass

    def _create_player(self, player_type: str, player_id: int, team_id: int):
        if player_type == "greedy":
            return Greedy_Player(player_id=player_id, team_id=team_id)
        elif player_type == "random":
            return Random_Player(player_id=player_id, team_id=team_id)
        else:
            raise ValueError(f"Unknown player type: {player_type}")

    def _get_inital_state(self) -> np.ndarray:
        # Shape: (n_state_of_card, n_player, n_cards). n_state_of_card: {0: in hand, 1: in trick, 2: first card in trick, 3: card played}
        init_state = np.zeros((4, 4, 36), dtype=int)

        for card in self.agent_hand:
            init_state[0, 0, card.index] = 1

        for idx, player in enumerate([self.p1, self.p2, self.p3]):
            for card in player.hand:
                init_state[0, idx + 1, card.index] = 1

        # The sum of one-hot encoded array should be always the sum of all cards (36)
        assert int(np.sum(init_state)) == 36
        return init_state

    def _update_state_after_play(self, action: int, player_id: int):
        if self.leading_player_id == player_id:
            self.state[2, player_id, action] = 1
            self.state[0, player_id, action] = 0
        else:
            self.state[1, player_id, action] = 1
            self.state[0, player_id, action] = 0
        assert int(np.sum(self.state)) == 36

    def _update_state_after_trick_ends(self):
        self.state[3, :, :] = self.state[1, :, :] | self.state[3, :, :]
        self.state[1, :, :] = 0

        self.state[3, :, :] = self.state[2, :, :] | self.state[3, :, :]
        self.state[2, :, :] = 0
        assert int(np.sum(self.state)) == 36

    def _play_init_trick(self):
        for i in range(4):
            self.current_turn = (self.leading_player_id + i) % 4

            if self.current_turn == 0:
                break

            current_player = self.players[self.current_turn - 1]
            card = current_player.play_card(self.trick)
            self.trick.trick[f"P{self.current_turn}"] = card
            if i == 0:
                self.trick.set_suit(card=card)
                print(f"Lead Player is: P{self.leading_player_id}. Playing suit is {repr(self.trick.playing_suit)}")
            print(f"P{current_player.player_id} played {card}")

            self._update_state_after_play(action=card.index, player_id=self.current_turn)
        assert int(np.sum(self.state)) == 36

