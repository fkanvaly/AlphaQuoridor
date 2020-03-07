from players.Player import *
from open_spiel.python.algorithms.minimax import *
from utils import * 

class GenPlayer(Player):
    def play(self, state):
        legal_actions = state.legal_actions()
        _ , action = alpha_beta_search(state.get_game(), state, self.evaluate, 2 ,self.idx)
        # _ , action = expectiminimax(state, 1, self.evaluate, self.idx)
        return action
    
    def evaluate(self, state):
        pass