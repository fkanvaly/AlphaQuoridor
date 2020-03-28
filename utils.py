from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from time import time
import numpy as np


import networkx as nx

def shortest_path(state, source, target, Graph=None):
    if Graph==None:
        raise("No graph provided")       
    try:
        sp = nx.shortest_path_length(Graph, source, target)
        return sp//2
    except nx.NetworkXNoPath:
        return float("inf")

# def _alpha_beta_negamax(state, memory, best_action_hist, depth, alpha, beta, value_function, history={}):
#   """An alpha-beta algorithm.
#   Returns:
#     A tuple of the optimal value of the sub-game starting in state
#     (given alpha/beta) and the move that achieved it.

#   Raises:
#     NotImplementedError: If we reach the maximum depth. Given we have no value
#       function for a non-terminal node, we cannot break early.
#   """
  
#   alpha_init = alpha
  
#   if depth == 0 and value_function is None:
#     raise NotImplementedError(
#         "We assume we can walk the full depth of the tree. "
#         "Try increasing the maximum_depth or provide a value_function.")
    
#   if depth == 0 or state.is_terminal():
#     return value_function(state), None, None, None, history
  
#   # if I have already enconter the state
#   if state.history_str() in memory:
#     v, a, d, f, h = memory[state.history_str()]
#     if d>=depth:
#       if f == 0 : 
#         return v, a, d, f, h
#       elif f == -1 : alpha=max(alpha, v)
#       elif f == 1 : beta=min(beta, v)
      
#       if alpha>=beta: 
#         return v, a, d, f, h
  
#   best_action = -1
#   best_hist = {}
#   value = -float("inf")
  
#   old_best_action = []
#   if state.history_str() in best_action_hist:
#     old_best_action.append(best_action_hist[state.history_str()])
  
#   legal_actions = old_best_action + state.legal_actions() if len(old_best_action)>0 else state.legal_actions()
#   for action in legal_actions:
#     child_state = state.clone()
#     child_state.apply_action(action)
    
#     hist = history.copy()
#     hist[state.history_str()] = action
    
#     if child_state.is_terminal():
#       value = value_function(state) 
#       best_action = action
#       history[state.history_str()] = action
#       break
    
#     if child_state.history_str() in history:
#       continue
    
#     child_value, _, _, _, hist =  _alpha_beta_negamax(child_state, memory, best_action_hist, depth - 1, -alpha, -beta,
#                                                           value_function, hist.copy())
#     child_value = -child_value
    
#     if child_value > value:
#       value = child_value
#       best_action = action
#       best_hist = hist
      
#     alpha = max(alpha, value)
#     if alpha >= beta:
#       break
  
  
#   if value<=alpha_init:
#     flag = 1
#   elif value >= beta:
#     flag=-1
#   else : 
#     flag = 0
    
  
#   memory[state.history_str()]=(value, best_action, depth, flag, best_hist)
#   return value, best_action, depth, flag, best_hist



# def alpha_beta_negamax_search(state,
#                               memory,
#                               value_function,
#                               best_action_hist,
#                               depth=1
#                               ):
#   """
#   """
#   return _alpha_beta_negamax(
#               state.clone(),
#               memory,
#               best_action_hist,
#               depth=depth,
#               alpha=-float("inf"),
#               beta=float("inf"),
#               value_function=value_function,
#               )

# def iterative_alpha_beta_negamax(state, memory, value_function, max_time=2):
#   d=1
#   t0 = time()
#   best_action_hist = {}
#   while (time()-t0)<max_time and d<3:
#     _, action, _, _, best_action_hist = alpha_beta_negamax_search(state, memory, value_function, best_action_hist, d)
#     d += 1
#   return action
  

# if __name__ == "__main__":
#   from players.GenPlayer import *
#   game = pyspiel.load_game("quoridor")
#   state = game.new_initial_state()
#   agent = GenPlayer(0, [1]*9)
#   action = iterative_alpha_beta_negamax(state, 5, agent.evaluate)