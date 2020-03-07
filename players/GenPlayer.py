from players.Player import *
from utils import * 

class GenPlayer(Player):
    def __init__(self, idx, weights):
        super().__init__(idx)
        self.weights = weights
        
    def play(self, state):
        legal_actions = state.legal_actions()
        # action = iterative_alpha_beta_negamax(state, 5, self.evaluate)
        _ , action = alpha_beta_search(state.get_game(), state, self.evaluate, 2 ,self.idx)
        return action
    
    def evaluate(self, state):
        print(state)
        i_p = state.current_player()
        i_o = (state.current_player() + 1) % 2
        observation = state.observation_tensor()
        pos_p = np.where(np.array(observation[289*i_p:289*(i_p+1)])==1.0)[0][0]
        pos_o = np.where(np.array(observation[289*i_o:289*(i_o+1)])==1.0)[0][0]

        SPP = BFS(observation, pos_p, self.terminal_state[i_p])
        SPO = BFS(observation, pos_o, self.terminal_state[i_o])
        SPP = (81-SPP)/81
        SPO = (81-SPO)/81
        
        GDD = (81 - abs(SPP-SPO))/81
        
        PD = BFS(observation, pos_p, [pos_o])
        PD = (81-PD)/81
        
        MDP = abs(self.terminal_state[i_p][0]//34 - pos_p//34)
        MDO = abs(self.terminal_state[i_o][0]//34 - pos_o//34)
        MDP = (8-MDP)/8
        MDO = (8-MDP)/8
        
        GSP = (MDP<4)*1
        GSO = (MDO<4)*1
        
        NFP = observation[289*(3+i_p):289*(3+i_p+1)][0]
        
        f = np.array([SPP, SPO, GDD, PD, MDP, MDO, NFP, GSP, GSO])
        
        return np.sum(f * self.weights)