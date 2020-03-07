from players.Player import *

class RandomPlayer(Player):
    def play(self, state):
        legal_actions = state.legal_actions()
        return np.random.choice(legal_actions)