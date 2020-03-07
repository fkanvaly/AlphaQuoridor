import numpy as np 

class Player:
    terminal_state = {1: list(range(17*16, 17*16+17, 2)), 
                      0: list(range(0, 17, 2))}
    def __init__(self, idx):
        self.idx = idx
        self.end_positions = self.terminal_state[idx]
        self.pos = None
        self.score = 0
        
    def play(self, state):
        pass
    
    