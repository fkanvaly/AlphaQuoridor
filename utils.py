from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time

forbidden = {"u": list(range(0,17,2)), "d": list(range(17*16, 17*16+17, 2)),
             "l": list(range(0,17*17, 34)),"r": list(range(16, 17*16+17, 34))}

direction = {"r": 2, "l":-2, "u":-34, "d":34}

def valid(pos, move, fences):
    if (pos in forbidden[move]):
        return False
    
    if fences[ pos + direction[move]//2 ]:
        return False
    
    return True
    

def BFS(observation_tensor, startCoord, endCoords):
    import heapq
    
    pq = [(0, startCoord)]
    direction = {"r": 2, "l":-2, "u":-2*17, "d":2*17}
    fences = observation_tensor[289*2:289*3]
    
    while len(pq)>0:
        dist, pos = heapq.heappop(pq)
        for move in ("u", "d", "r", "l"):
            if valid(pos, move, fences):
                new_pos = pos + direction[move]
                if new_pos in endCoords:
                    return dist + 1
                else:
                    heapq.heappush( pq, ( dist+1, new_pos ) )



import pyspiel

def _alpha_beta(state, depth, alpha, beta, value_function,
                maximizing_player_id):
  """
  """
  if state.is_terminal():
    return state.player_return(maximizing_player_id), None

  if depth == 0 and value_function is None:
    raise NotImplementedError(
        "We assume we can walk the full depth of the tree. "
        "Try increasing the maximum_depth or provide a value_function.")
  if depth == 0:
    return value_function(state), None

  player = state.current_player()
  best_action = -1
  if player == maximizing_player_id:
    value = -float("inf")
    for action in state.legal_actions():
      child_state = state.clone()
      child_state.apply_action(action)
      child_value, _ = _alpha_beta(child_state, depth - 1, alpha, beta,
                                   value_function, maximizing_player_id)
      if child_value > value:
        value = child_value
        best_action = action
      alpha = max(alpha, value)
      if alpha >= beta:
        break  # beta cut-off
    return value, best_action
  else:
    value = float("inf")
    for action in state.legal_actions():
      child_state = state.clone()
      child_state.apply_action(action)
      child_value, _ = _alpha_beta(child_state, depth - 1, alpha, beta,
                                   value_function, maximizing_player_id)
      if child_value < value:
        value = child_value
        best_action = action
      beta = min(beta, value)
      if alpha >= beta:
        break  # alpha cut-off
    return value, best_action


def alpha_beta_search(game,
                      state=None,
                      value_function=None,
                      maximum_depth=30,
                      maximizing_player_id=None):
  """Solves deterministic, 2-players, perfect-information 0-sum game.

  For small games only! Please use keyword arguments for optional arguments.

  Arguments:
    game: The game to analyze, as returned by `load_game`.
    state: The state to run from, as returned by `game.new_initial_state()`.  If
      none is specified, then the initial state is assumed.
    value_function: An optional function mapping a Spiel `State` to a numerical
      value, to be used as the value of the maximizing player for a node when we
      reach `maximum_depth` and the node is not terminal.
    maximum_depth: The maximum depth to search over. When this depth is reached,
      an exception will be raised.
    maximizing_player_id: The id of the MAX player. The other player is assumed
      to be MIN. The default (None) will suppose the player at the root to be
      the MAX player.

  Returns:
    A tuple containing the value of the game for the maximizing player when
    both player play optimally, and the action that achieves this value.
  """
  game_info = game.get_type()

  if game.num_players() != 2:
    raise ValueError("Game must be a 2-player game")
  if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    raise ValueError("The game must be a Deterministic one, not {}".format(
        game.chance_mode))
  if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    raise ValueError(
        "The game must be a perfect information one, not {}".format(
            game.information))
  if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("The game must be turn-based, not {}".format(
        game.dynamics))
  if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
    raise ValueError("The game must be 0-sum, not {}".format(game.utility))

  if state is None:
    state = game.new_initial_state()
  if maximizing_player_id is None:
    maximizing_player_id = state.current_player()
  return _alpha_beta(
      state.clone(),
      maximum_depth,
      alpha=-float("inf"),
      beta=float("inf"),
      value_function=value_function,
      maximizing_player_id=maximizing_player_id)


class PV:
  def __init__(self, value, move, state):
    self.value = value
    self.state = state
    self.move = move

class Entry():
  def __init__(self, depth=None, flag=None, pv=None, action=None):
    self.depth = depth
    self.flag = flag
    self.pv = pv  
    self.action = action  
    self.state = None

def _alpha_beta_negamax(state, depth, alpha, beta, value_function, pv, history,  tmp, move=None):
  """An alpha-beta algorithm.
  Returns:
    A tuple of the optimal value of the sub-game starting in state
    (given alpha/beta) and the move that achieved it.

  Raises:
    NotImplementedError: If we reach the maximum depth. Given we have no value
      function for a non-terminal node, we cannot break early.
  """
  # oriAlpha = alpha
  # if hash(state) in tmp:
  #   entry = tmp[hash(state)]
  #   import ipdb; ipdb.set_trace()
  #   if entry.depth>=depth:
  #     if entry.flag == 0:
  #       return pv + [entry.pv], entry.action
  #     if entry.flag == 1:
  #       alpha = max(alpha, entry.pv.value)
  #     if entry.flag == -1:
  #       beta = max(beta, entry.pv.value)
      
  #     if alpha>=beta:
  #       return pv + [entry.pv], entry.action
        
  if depth == 0 and value_function is None:
    raise NotImplementedError(
        "We assume we can walk the full depth of the tree. "
        "Try increasing the maximum_depth or provide a value_function.")
  if depth == 0 or state.is_terminal():
    new_pv = PV(value_function(state), move, state.clone())
    return pv + [new_pv], None
  
  # history[hash(state)]=1
  
  best_action = -1
  value = -float("inf")
  for action in state.legal_actions():
    child_state = state.clone()
    child_state.apply_action(action)
    
    # if child_state.is_terminal():
      # break
    
    # if hash(child_state) in history:
    #   continue
    
    pv_child, _ = _alpha_beta_negamax(child_state, depth - 1, -alpha, -beta,
                                          value_function, pv.copy(), history.copy(), tmp, action)
    pv_child[-1].value = -pv_child[-1].value
    if pv_child[-1].value > value:
      value = pv_child[0].value
      best_action=action
    alpha = max(alpha, value)
    if alpha >= beta:
      break
    
  new_pv = PV(value_function(state), move, state.clone())
  new_entry = Entry(pv=new_pv, action=best_action)
  if new_pv.value <= oriAlpha:
    new_entry.flag = 1
  elif new_pv.value >= beta:
    new_entry.flag = -1
  else:
    new_entry.flag = 0
  new_entry.depth = depth
  new_entry.state = state.clone()
  tmp[hash(state)] = new_entry
  
  return pv + [new_pv], best_action



def alpha_beta_negamax_search(state,
                              value_function=None,
                              depth=1,
                              tmp = {},
                              pv=[]):
  """
  """
  return _alpha_beta_negamax(
              state.clone(),
              depth=depth,
              alpha=-float("inf"),
              beta=float("inf"),
              value_function=value_function,
              pv = pv,
              history={}, 
              tmp=tmp)

def iterative_alpha_beta_negamax(state, max_time, value_function=None):
  d=2
  t0 = time()
  tmp = {}
  pv = []
  # while (time()-t0)<max_time:
  _, action = alpha_beta_negamax_search(state,value_function, d, tmp)
  import ipdb; ipdb.set_trace()
    # d += 1
  
  return action
  

if __name__ == "__main__":
  from players.GenPlayer import *
  game = pyspiel.load_game("quoridor")
  state = game.new_initial_state()
  agent = GenPlayer(0, [1]*9)
  action = iterative_alpha_beta_negamax(state, 5, agent.evaluate)
  import ipdb; ipdb.set_trace()
  