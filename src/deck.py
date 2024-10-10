import random

from card import Card, Rank, Suit


class Deck:
    """Full Deck of Cards"""

    def __init__(self):
        self.cards = [Card(s, r) for s in Suit for r in Rank]

        # Shuffle the card deck
        self.shuffle_deck()

    def shuffle_deck(self):
        random.shuffle(self.cards)

    def pop_cards(self, n_cards: int) -> list:
        popped_elements = []
        for _ in range(n_cards):
            if self.cards:  # Ensure the list isn't empty
                popped_elements.append(self.cards.pop())
            else:
                raise Exception("Deck is empty!")
        return popped_elements
