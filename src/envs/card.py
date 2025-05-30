from enum import Enum


class Suit(Enum):
    ROSE = 0
    SCHILTE = 1
    EICHEL = 2
    SCHELLE = 3

    def __repr__(self) -> str:
        representation_suit = {
            "ROSE": "🌼",
            "SCHILTE": "🛡️ ",
            "EICHEL": "🌰",
            "SCHELLE": "🔔",
        }
        return f"{representation_suit[str(self.name)]}"


class Rank(Enum):
    SECHS = 6
    SIEBEN = 7
    ACHT = 8
    NEUN = 9
    ZEHN = 10
    UNDER = 11
    OBER = 12
    KOENIG = 13
    ASS = 14


class Card:
    def __init__(self, suit: Suit, rank: Rank, value=None):
        self.suit = suit
        self.rank = rank
        self.value = value

        # To identify the card using a single number from 0 to 35
        self.index = self.suit.value * 9 + (self.rank.value - 6)

    # Equality to compare two cards (useful when removing card from hand using .remove())
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __repr__(self) -> str:
        representation_suit = {
            "ROSE": "🌼",
            "SCHILTE": "🛡️ ",
            "EICHEL": "🌰",
            "SCHELLE": "🔔",
        }
        representation_rank = {
            "SECHS": "6",
            "SIEBEN": "7",
            "ACHT": "8",
            "NEUN": "9",
            "ZEHN": "10",
            "UNDER": "U",
            "OBER": "O",
            "KOENIG": "K",
            "ASS": "A",
        }
        return f"{representation_suit[str(self.suit.name)]}{representation_rank[str(self.rank.name)]}"
