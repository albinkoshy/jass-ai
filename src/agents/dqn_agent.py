from agents.agent_interface import IAgent
from envs.card import Card, Suit


class DQN_Agent(IAgent):

    def __init__(self, player_id, team_id):
        super().__init__()
        self.player_id = player_id
        self.team_id = team_id

    def act(self, state):
        pass

    def load_model(self, loadpath: str):
        pass

    def save_model(self, name: str = "", directory: str = "./saved_models/"):
        pass

    def optimize_model(self):
        pass

    def reset(self):
        pass
