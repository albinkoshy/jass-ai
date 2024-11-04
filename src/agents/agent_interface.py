from abc import ABC, abstractmethod

"""
Interface for various kinds of Agents
"""


class IAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def optimize_model(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def load_model(self, loadpath: str):
        pass

    @abstractmethod
    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass
