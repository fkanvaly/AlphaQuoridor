from players.Player import *
from open_spiel.python.algorithms import mcts
from utils import * 
import math
import pyspiel
import numpy as np
from game import *

UCT_C = math.sqrt(2)

import networkx as nx
n=17; G=nx.grid_2d_graph(n,n); G.remove_nodes_from([(i,j) for i in range(1,n,2) for j in range(1,n,2)])
coord = lambda i: (i//17, i%17)

pawn_move = lambda i: (i//17)%2==0 and (i%17)%2==0

class RandomRolloutEvaluator(mcts.Evaluator):
    """
    A simple evaluator doing random rollouts.

    This evaluator returns the average outcome of playing random actions from the
    given state until the end of the game.  n_rollouts is the number of random
    outcomes to be considered.
    """
    
    terminal_state = {1: list(range(17*16, 17*16+17, 2)), 
                      0: list(range(0, 17, 2))}

    def __init__(self, n_rollouts=1, random_state=None):
        self.n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()

    def evaluate(self, state):
        """Returns evaluation on given state."""
        result = np.array([0,0])
        max_moves=100
        start_player = state.current_player()
        
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            n=0
            while not working_state.is_terminal() and n<max_moves:
                current_player = working_state.current_player()
                action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
                n += 1
                
            if state.current_player() < 0 :
                result[current_player]+=10

        return result / self.n_rollouts
    

    def prior(self, state):
        """Returns equal probability for all actions."""
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            legal_actions = state.legal_actions(state.current_player())
            return [(action, 1.0 / len(legal_actions)) for action in legal_actions]
        
       
        



class MCTS_Player(Player):
    def __init__(self, idx, max_simulations=500):
        super().__init__(idx)
        game = pyspiel.load_game("quoridor")
        self.evaluator = RandomRolloutEvaluator(2)
        self.bot = mcts.MCTSBot(game, UCT_C, max_simulations, self.evaluator)
        
    def play(self, state):
        """Monte carlo simulation to choose action"""
        return self.bot.step(state)
    
    def evaluate(self, state):
        """Returns evaluation on given state."""
        result = None
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            while not working_state.is_terminal():
                action = self._random_state.choice(working_state.legal_actions())
                working_state.apply_action(action)
        returns = np.array(working_state.returns())
        result = returns if result is None else result + returns

        return result / self.n_rollouts
    
    
