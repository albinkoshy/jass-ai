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
        self.game_type = None
        
        self.n_tricks = None
        
        self.rewards_per_player = None
        self.rewards_per_team = None
        self.reward_won_last_trick_per_player = None
        self.reward_won_last_trick_per_team = None

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
        
        self.game_type = None # Which game type is played for this round, one of these: ["TOP_DOWN", "BOTTOM_UP", "ROSE", "SCHILTE", "EICHEL", "SCHELLE", "SCHIEBEN"]
                              # If the starting player chooses 'SCHIEBEN', his team mate has to set the game type (all of above except 'SCHIEBEN')
        Trick.game_type = None
        
        self.n_tricks = 0
        
        self.rewards_per_player = [0, 0, 0, 0] # To keep track of rewards (points won in tricks) for each player
        self.rewards_per_team = [0, 0] # To keep track of rewards (points won in tricks) for each team
        self.reward_won_last_trick_per_player = [0, 0, 0, 0] # To keep track of the reward won in the last trick
        self.reward_won_last_trick_per_team = [0, 0] # To keep track of the reward won in the last trick by each team
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
            self.rewards_per_player[trick_winner_id] += trick_points
            self.rewards_per_team[trick_winner_id % 2] += trick_points
            
            self.reward_won_last_trick_per_player = [0, 0, 0, 0]
            self.reward_won_last_trick_per_player[trick_winner_id] = trick_points
            self.reward_won_last_trick_per_team = [0, 0]
            self.reward_won_last_trick_per_team[trick_winner_id % 2] = trick_points
            
            if self.print_globals and self.n_tricks < 9:
                print(f"P{trick_winner_id} won {self.n_tricks}.trick")
                print(f"Points per player: {self.rewards_per_player}")
                print(f"Points per team: {self.rewards_per_team}")
                print()
            
            if self.n_tricks == 9:
                assert self.hands == {f"P{player.player_id}": [] for player in self.players}, "All cards should have been played"
                # Round is over
                done = True
                
                # Trick winner gets 5 additional points
                self.rewards_per_player[trick_winner_id] += 5
                assert sum(self.rewards_per_player) == 157, "Total points should be 157"
                self.rewards_per_team[trick_winner_id % 2] += 5
                assert sum(self.rewards_per_team) == 157, "Total points should be 157"
                
                self.reward_won_last_trick_per_player[trick_winner_id] += 5
                self.reward_won_last_trick_per_team[trick_winner_id % 2] += 5

                self.trick = None
                self.leading_player_id = None
                
                if self.print_globals:
                    print(f"P{trick_winner_id} won {self.n_tricks}.trick")
                    print(f"Final points distribution per player: {self.rewards_per_player}")
                    print(f"Final points distribution per team: {self.rewards_per_team}")
                
                return self._get_state(), [r / 157 for r in (2*self.reward_won_last_trick_per_team)], done # Note: 2*my_list concatenates the list with itself (so the player sees the reward or its team)
            
            # Start a new trick
            self.trick = Trick(leading_player_id=self.leading_player_id)
            
            if self.print_globals:
                self._print_hands()
                print(f"P{self.leading_player_id} is leading the round")
            
            return self._get_state(), [r / 157 for r in (2*self.reward_won_last_trick_per_team)], False
        else:
            # Continue playing the trick            
            return self._get_state(), [r / 157 for r in (2*self.reward_won_last_trick_per_team)], False

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
            self.trick.playing_suit = card.suit
        
        if self.print_globals:
            if self.leading_player_id == player_id:
                print(f"Trick suit: {self.trick.playing_suit.__repr__()}")
            print(f"P{self.current_turn} played {utils.ORDERED_CARDS[action]}")
            
        # Update current turn
        self.current_turn = (self.current_turn + 1) % 4
    
    def set_game_type(self, game_type: str, is_geschoben: bool = False):
        assert game_type in ["TOP_DOWN", "BOTTOM_UP", "ROSE", "SCHILTE", "EICHEL", "SCHELLE"], "Invalid game type"
        self.game_type = game_type
        Trick.game_type = game_type
        
        if self.print_globals:
            if is_geschoben:
                team_mate_id = (self.starting_player_id + 2) % 4
                print(f"Geschoben. P{team_mate_id} chooses: {game_type}")
            else:
                print(f"P{self.starting_player_id} chooses: {game_type}")
    
    def _get_state(self):
        return {
            'hands': self.hands,
            'trick': self.trick,
            'leading_player_id': self.leading_player_id,
            'game_type': self.game_type
        }
