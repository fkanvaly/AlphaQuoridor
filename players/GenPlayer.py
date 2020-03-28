
from open_spiel.python.algorithms.minimax import alpha_beta_search
from players.Player import *
from utils import * 

from threading import Thread

coord = lambda i: (i//17, i%17)

import networkx as nx
n=17; H=nx.grid_2d_graph(n,n); H.remove_nodes_from([(i,j) for i in range(1,n,2) for j in range(1,n,2)])
mapping = dict( ((u//n,u%n),u) for u in range(n*n) )
G = nx.relabel_nodes(H, mapping)

class GenPlayer(Player):
    def __init__(self, idx, weights):
        super().__init__(idx)
        self.weights = weights
        self.memory = {}
        
    def play(self, state):
        legal_actions = state.legal_actions()
        _ , action = alpha_beta_search(state.get_game(), state, self.evaluate, 2 , self.idx)
        return action
    
    def evaluate(self, state):
        i_p = state.current_player()
        i_o = (state.current_player() + 1) % 2
        observation = state.observation_tensor()
        pos_p = np.where(np.array(observation[289*i_p:289*(i_p+1)])==1.0)[0][0]
        pos_o = np.where(np.array(observation[289*i_o:289*(i_o+1)])==1.0)[0][0]
        
        #build the game in graph structure
        fences = observation[289*2:289*3]
        fences = list(np.where(np.array(fences)==1.0)[0])
        G_s = G.copy()
        G_s.remove_nodes_from(fences)
        
        #compute shortest path
        PD = shortest_path(state, pos_p, pos_o, G_s)
        SPP = min([shortest_path(state, pos_p, pos, G_s) for pos in self.terminal_state[i_p]])
        SPO = min([shortest_path(state, pos_o, pos, G_s) for pos in self.terminal_state[i_o]])
        if SPP == float("inf") or SPO == float("inf") : raise "No path to goal"
        SPP = (81-SPP)/81
        SPO = (81-SPO)/81
        GDD = SPO-SPP
        PD = (81-PD)/81
        
        MDP = abs(self.terminal_state[i_p][0]//34 - pos_p//34)
        MDO = abs(self.terminal_state[i_o][0]//34 - pos_o//34)
        MDP = (8-MDP)/8
        MDO = (8-MDP)/8
        
        GSP = (MDP<4)*1
        GSO = (MDO<4)*1
        
        NFP = observation[289*(3+i_p):289*(3+i_p+1)][0]/10
        
        f = np.array([ SPP, SPO, GDD, MDP, MDO, PD, GSP, GSO, NFP ])
        
        return np.sum(f * self.weights)
    
    
    def test(self, state):
        i_p = state.current_player()
        i_o = (state.current_player() + 1) % 2
        observation = state.observation_tensor()
        pos_p = np.where(np.array(observation[289*i_p:289*(i_p+1)])==1.0)[0][0]
        pos_o = np.where(np.array(observation[289*i_o:289*(i_o+1)])==1.0)[0][0]
        
        #build the game in graph structure
        fences = observation[289*2:289*3]
        fences = list(np.where(np.array(fences)==1.0)[0])
        fences = [coord(i) for i in fences]
        G_s = G.copy()
        G_s.remove_nodes_from(fences)
        
        #compute shortest path
        PD = shortest_path(state, pos_p, pos_o, G_s)
        SPP = min([shortest_path(state, pos_p, pos, G_s) for pos in self.terminal_state[i_p]])
        SPO = min([shortest_path(state, pos_o, pos, G_s) for pos in self.terminal_state[i_o]])
        if SPP == float("inf") or SPO == float("inf") : raise "No path to goal"
        
        GDD = SPO-SPP
        
        MDP = abs(self.terminal_state[i_p][0]//34 - pos_p//34)
        MDO = abs(self.terminal_state[i_o][0]//34 - pos_o//34)
        
        GSP = (MDP<4)*1
        GSO = (MDO<4)*1
        
        NFP = observation[289*(3+i_p):289*(3+i_p+1)][0]
        
        f = { "SPP":SPP,  "SPO":SPO,  "GDD":GDD, "PD":PD,  "MDP":MDP,  "MDO":MDO,  "NFP":NFP,  "GSP":GSP,  "GSO":GSO}
        
        return f