from envs.card import Card, Rank, Suit


TOP_DOWN_SCORING = {
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

BOTTOM_UP_SCORING = {
    Rank.SECHS: 11,
    Rank.SIEBEN: 0,
    Rank.ACHT: 8,
    Rank.NEUN: 0,
    Rank.ZEHN: 10,
    Rank.UNDER: 2,
    Rank.OBER: 3,
    Rank.KOENIG: 4,
    Rank.ASS: 0,
}

TRUMP_SCORING = {
    Rank.SECHS: 0,
    Rank.SIEBEN: 0,
    Rank.ACHT: 0,
    Rank.NEUN: 14,
    Rank.ZEHN: 10,
    Rank.UNDER: 20,
    Rank.OBER: 3,
    Rank.KOENIG: 4,
    Rank.ASS: 11,
}

NOT_TRUMP_SCORING = {
    Rank.SECHS: 0,
    Rank.SIEBEN: 0,
    Rank.ACHT: 0,
    Rank.NEUN: 0,
    Rank.ZEHN: 10,
    Rank.UNDER: 2,
    Rank.OBER: 3,
    Rank.KOENIG: 4,
    Rank.ASS: 11,
}

TRUMP_ORDERING = {
    Rank.SECHS: 16,
    Rank.SIEBEN: 17,
    Rank.ACHT: 18,
    Rank.ZEHN: 19,
    Rank.OBER: 20,
    Rank.KOENIG: 21,
    Rank.ASS: 22,
    Rank.NEUN: 23,
    Rank.UNDER: 24
}

class Trick:

    # Global settings
    game_type = None
    
    def __init__(self, leading_player_id):
        self.trick = {"P0": None, "P1": None, "P2": None, "P3": None}
        self.leading_player_id = leading_player_id
        self.playing_suit = None

    def can_take_trick(self, card: Card) -> bool:
        if Trick.game_type in ["TOP_DOWN", "BOTTOM_UP"]:
            if self.playing_suit != card.suit:
                return False
            same_suit_cards = [c for c in self.trick.values() if c and c.suit == self.playing_suit]
            if not same_suit_cards:
                return True
            best_card = max(same_suit_cards, key=lambda c: c.rank.value) if Trick.game_type == "TOP_DOWN" else min(same_suit_cards, key=lambda c: c.rank.value)
            return card.rank.value > best_card.rank.value if Trick.game_type == "TOP_DOWN" else card.rank.value < best_card.rank.value
        elif Trick.game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, Trick.game_type)
            
            if self.playing_suit != card.suit and trump_suit != card.suit:
                return False
            best_c_value = -1
            for c in self.trick.values():
                if c:
                    if c.suit == trump_suit:
                        c_value = TRUMP_ORDERING[c.rank]
                    elif c.suit == self.playing_suit:
                        c_value = c.rank.value
                    else:
                        c_value = -1
                    best_c_value = max(best_c_value, c_value)
            card_value = TRUMP_ORDERING[card.rank] if card.suit == trump_suit else card.rank.value
            return card_value > best_c_value
        else:
            raise ValueError("Invalid game type")

    def determine_trick_winner(self) -> str:
        """
        Determine the winner of the trick

        Returns:
            str: Player who won the trick. Either "P0", "P1", "P2" or "P3"
        """
        # Check if all cards have been laid
        assert all(value is not None for value in self.trick.values()), "Not all cards have been laid!"

        winner = None
        winner_card_value = None
        if Trick.game_type == "TOP_DOWN":
            for player, card in self.trick.items():
                card_value = card.rank.value if card.suit == self.playing_suit else 0
                if winner_card_value is None or card_value > winner_card_value:
                    winner_card_value = card_value
                    winner = player
            return winner
        elif Trick.game_type == "BOTTOM_UP":
            for player, card in self.trick.items():
                card_value = card.rank.value if card.suit == self.playing_suit else 0
                if winner_card_value is None or card_value < winner_card_value:
                    winner_card_value = card_value
                    winner = player
            return winner
        elif Trick.game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, Trick.game_type)
            for player, card in self.trick.items():
                
                if card.suit == trump_suit:
                    card_value = TRUMP_ORDERING[card.rank]
                elif card.suit == self.playing_suit:
                    card_value = card.rank.value
                else:
                    card_value = 0
                
                if winner_card_value is None or card_value > winner_card_value:
                    winner_card_value = card_value
                    winner = player
            return winner
        else:
            raise ValueError("Invalid game type")

    def get_trick_points(self):
        # Check if all cards have been laid
        assert all(value is not None for value in self.trick.values()), "Not all cards have been laid!"

        trick_points = 0
        if Trick.game_type == "TOP_DOWN":
            for card in self.trick.values():
                trick_points += TOP_DOWN_SCORING[card.rank]
        elif Trick.game_type == "BOTTOM_UP":
            for card in self.trick.values():
                trick_points += BOTTOM_UP_SCORING[card.rank]
        elif Trick.game_type in ["ROSE", "SCHILTE", "EICHEL", "SCHELLE"]:
            trump_suit = getattr(Suit, Trick.game_type)
            for card in self.trick.values():
                if card.suit == trump_suit:
                    trick_points += TRUMP_SCORING[card.rank]
                else:
                    trick_points += NOT_TRUMP_SCORING[card.rank]
        else:
            raise ValueError("Invalid game type")

        return trick_points
