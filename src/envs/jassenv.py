from envs.deck import Deck
from envs.trick import Trick
import utils

class JassEnv:
    
    def __init__(self, players, print_globals=False):
        self.players = players # Either learning/trained agents or fixed strategy players
        self.print_globals = print_globals
        
        self.starting_player_id = None
        self.leading_player_id = None
        self.current_turn = None
        
        self.hands = None
        self.trick = None
        
        self.n_tricks = None
        
        self.rewards = None
        
        self.reward_won_last_trick = None

    def reset(self, starting_player_id) -> dict:
        """
        Start a new round. Deal cards to players and return the initial state.
        
        Args:
            starting_player_id (int): Player who starts the first trick
            
        Returns:
            dict: Dict containing the state of the game
        """
        
        self.starting_player_id = starting_player_id # Player who starts the first trick
        self.leading_player_id = starting_player_id  # Player who won the last trick, starts the next trick
        self.current_turn = starting_player_id # Player who is currently playing a card
        
        self.hands = {f"P{player.player_id}": [] for player in self.players}
        self._deal_cards()
        
        self.trick = Trick(leading_player_id=self.leading_player_id)
        if self.print_globals:
            self._print_hands()
            print(f"P{self.starting_player_id} is starting the round")
        
        self.n_tricks = 0
        
        self.rewards = [0, 0, 0, 0] # To keep track of rewards (points won in tricks) for each player
        self.reward_won_last_trick = [0, 0, 0, 0] # To keep track of the reward won in the last trick
        return self._get_state()
    
    def get_current_turn(self):
        return self.current_turn

    def _print_hands(self):
        for player in self.hands:
            print(f"{player}'s hand: {self.hands[player]}")
            
    def _deal_cards(self):
        deck = Deck()
        self.hands["P0"] = deck.pop_cards(n_cards=9)
        self.hands["P1"] = deck.pop_cards(n_cards=9)
        self.hands["P2"] = deck.pop_cards(n_cards=9)
        self.hands["P3"] = deck.pop_cards(n_cards=9)
        
        # Sort all hands
        for player in self.hands:
            self.hands[player].sort(key=lambda card: (card.suit.value, card.rank.value))

    def step(self, action: int) -> tuple[dict, list, bool]:
        """
        Players act in the environment. Update the state of the game based on the action of the current player.

        Args:
            action (int): The action the current player takes, from the set {0, ..., 35}

        Returns:
            tuple[dict, list, bool]: Returns (next_state, rewards, done)
        """
        
        # Update state using action
        self._update_env_after_play(action=action, player_id=self.current_turn)
        
        # Check if all players have played a card
        if self.current_turn == self.leading_player_id:
            self.n_tricks += 1
            # Determine winner of the trick
            trick_winner = self.trick.determine_trick_winner()
            trick_winner_id = int(trick_winner[1])
            
            # Update leading player for next trick
            self.leading_player_id = trick_winner_id
            self.current_turn = trick_winner_id
            
            trick_points = self.trick.get_trick_points()
            self.rewards[trick_winner_id] += trick_points
            self.reward_won_last_trick = [0, 0, 0, 0]
            self.reward_won_last_trick[trick_winner_id] = trick_points
            
            if self.print_globals and self.n_tricks < 9:
                print(f"P{trick_winner_id} won {self.n_tricks}.trick")
                print(f"Points: {self.rewards}")
                print()
            
            if self.n_tricks == 9:
                assert self.hands == {f"P{player.player_id}": [] for player in self.players}, "All cards should have been played"
                # Round is over
                done = True
                
                # Trick winner gets 5 additional points
                self.rewards[trick_winner_id] += 5
                assert sum(self.rewards) == 157, "Total points should be 157"

                self.trick = None
                self.leading_player_id = None
                
                if self.print_globals:
                    print(f"P{trick_winner_id} won {self.n_tricks}.trick")
                    print(f"Final points distribution: {self.rewards}")
                
                return self._get_state(), [r / 44 for r in self.reward_won_last_trick], done
            
            # Start a new trick
            self.trick = Trick(leading_player_id=self.leading_player_id)
            
            if self.print_globals:
                self._print_hands()
                print(f"P{self.leading_player_id} is leading the round")
            
            return self._get_state(), [r / 44 for r in self.reward_won_last_trick], False
        else:
            # Continue playing the trick            
            return self._get_state(), [r / 44 for r in self.reward_won_last_trick], False

    def _update_env_after_play(self, action: int, player_id: int):
        # Remove card from player's hand
        card = utils.ORDERED_CARDS[action]
        assert card in self.hands[f"P{player_id}"], "Card not in player's hand"
        self.hands[f"P{player_id}"].remove(card)
        
        # Update trick
        assert self.trick.trick[f"P{player_id}"] is None, "Player has already played a card"
        self.trick.trick[f"P{player_id}"] = card
        if self.leading_player_id == player_id:
            # Leading player determines the suit of the trick
            self.trick.set_suit(card=card)
        
        if self.print_globals:
            if self.leading_player_id == player_id:
                print(f"Trick suit: {self.trick.playing_suit.__repr__()}")
            print(f"P{self.current_turn} played {utils.ORDERED_CARDS[action]}")
            
        # Update current turn
        self.current_turn = (self.current_turn + 1) % 4
    
    def _get_state(self):
        return {
            'hands': self.hands,
            'trick': self.trick,
            'leading_player_id': self.leading_player_id,
        }


import copy
from agents.random_agent import Random_Agent

if __name__ == "__main__":
    
    """ Test JassEnv """
    
    # Player table
    #   P2
    # P3  P1
    #   P0
    
    # Initialize players: Either learning/trained agents or fixed strategy players. To be passed to JassEnv
    players = [Random_Agent(player_id=0, team_id=0),
               Random_Agent(player_id=1, team_id=1),
               Random_Agent(player_id=2, team_id=0),
               Random_Agent(player_id=3, team_id=1)]
    
    # Initialize the environment
    env = JassEnv(players=players)
    starting_player_id = 0
    
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

        action = players[current_turn].act(state)
        new_state, rewards, done = env.step(action)
        
        state_action_pairs[f"P{current_turn}"]["state"] = copy.deepcopy(state)
        state_action_pairs[f"P{current_turn}"]["action"] = action

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
    