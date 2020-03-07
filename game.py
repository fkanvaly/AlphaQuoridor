import pyspiel
import numpy as np

class Game():
    
    def __init__(self, players):
        self.game = pyspiel.load_game("quoridor")
        self.state = self.game.new_initial_state()
        
        self.players = {}
        for player in players:
            self.players[ player.idx ] = player
    
    def start(self, rounds, display=True, max_moves=100):
        winner_history = []
        for _ in range(rounds):
            
            if display :print("ROUND %s/%s: " % ( _, rounds))
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
            
            winner_history.append(winner)
        
        return winner_history
            
            
    