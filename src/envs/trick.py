from envs.card import Card, Rank, Suit


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

    def can_take_trick(self, card: Card) -> bool:
        if self.playing_suit != card.get_suit():
            return False
        highest_card = self._get_highest_card()
        return card.get_rank().value > highest_card.get_rank().value

    def _get_highest_card(self) -> Card:
        same_suit_cards = [card for card in list(self.trick.values()) if card and card.get_suit() == self.playing_suit]
        return max(same_suit_cards, key=lambda c: c.rank.value)

    def determine_trick_winner(self) -> str:
        """
        Determine the winner of the trick

        Raises:
            Exception: Not all cards have been laid!

        Returns:
            str: Player who won the trick. Either "P0", "P1", "P2" or "P3"
        """
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

    def get_trick_points(self):
        # Check if all cards have been laid
        if all(value is not None for value in self.trick.values()):

            # top-down
            top_down_scoring = {
                Rank.SECHS: 0,
                Rank.SIEBEN: 0,
                Rank.ACHT: 8,
                Rank.NEUN: 0,
                Rank.ZEHN: 10,
                Rank.UNDER: 2,
                Rank.OBER: 3,
                Rank.KOENIG: 4,
                Rank.ASS: 11,
            }

            trick_points = 0
            for card in self.trick.values():
                trick_points += top_down_scoring[card.get_rank()]

            return trick_points
        else:
            raise Exception("Not all cards have been laid!")
