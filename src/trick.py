from card import Card, Suit


class Trick:

    def __init__(self, leading_player_id):
        self.trick = {"P0": None, "P1": None, "P2": None, "P3": None}
        self.leading_player_id = leading_player_id
        self.playing_suit = None

    def get_suit(self) -> Suit:
        return self.playing_suit

    def set_suit(self, card: Card):
        self.playing_suit = card.get_suit()

    def is_starting(self):
        return all(value is None for value in self.trick.values())

    def determine_trick_winner(self):
        # Check if all cards have been laid
        if all(value is not None for value in self.trick.values()):
            # TODO: Differentiate between playing variant (trump_suit, bottom_up, top_down)

            # top_down
            winner = None
            winner_card_value = None
            for player, card in self.trick.items():
                card_value = card.rank.value if card.get_suit() == self.playing_suit else 0
                if winner_card_value is None or card_value > winner_card_value:
                    winner_card_value = card_value
                    winner = player
            return winner
        else:
            raise Exception("Not all cards have been laid!")