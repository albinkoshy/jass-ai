from card import Suit, Rank, Card
import random

""" Full Deck of Cards """
class Deck():
    
    def __init__(self):
        self.cards = [Card(s, r) for s in Suit for r in Rank]
        
        # Shuffle the card deck
        self.shuffle_deck()
        
    def shuffle_deck(self):
        self.cards = random.shuffle(self.cards)