from enum import Enum

class Suit(Enum):
    EICHEL = 0
    SCHILTEN = 1
    ROSEN = 2
    SCHELLEN = 3

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

class Card():
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    
    def __repr__(self) -> str:
        representation_suit = {
            "EICHEL": "🌰",
            "SCHILTEN": "🛡️",
            "ROSEN": "🌼",
            "SCHELLEN": "🔔"
        }
        representation_rank = {
            "SECHS" : "6",
            "SIEBEN" : "7",
            "ACHT" : "8",
            "NEUN" : "9",
            "ZEHN" : "10",
            "UNDER" : "U",
            "OBER" : "O",
            "KOENIG" : "K",
            "ASS" : "A"
        }
        return f"{representation_suit[str(self.suit.name)]}{representation_rank[str(self.rank.name)]}"