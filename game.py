import pyspiel
import numpy as np

class Game():
    
    
    icon = {0: "0", 1:"@"}
    def __init__(self, players):
        self.game = pyspiel.load_game("quoridor")
        self.state = self.game.new_initial_state()
        
        self.players = {}
        for player in players:
            self.players[ player.idx ] = player
    
    def start(self, display=True, max_moves=120):
        
        self.state = self.game.new_initial_state()
        
        if display : print(self.state)
        
        n = 0
        while not self.state.is_terminal() and n<max_moves:
            current_player = self.state.current_player()
            action = self.players[current_player].play(self.state)
            self.state.apply_action(action)
            n += 1
            
            if display : print(self.state)
        
        
        if self.state.current_player() < 0 :
            winner = current_player
        else:
            winner = -1 #draw

        if display : print("winner is : %s" % winner)
        
        return winner
            
            
    