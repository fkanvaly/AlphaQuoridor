from game import *
from players.RandomPlayer import *
from players.GenPlayer import *
from players.MCTS_Player import *

if __name__ == "__main__":
    
    psi1_weigts = np.array([0.747,0.096,0, 0,-0.792,0,0,0,0.327])
    gen2_weigts = np.array([0,0.331,1.165,0, -0.300,0.903,0.522,-0.504,-0.265])
    diff_weigts = np.array([-1,1,0,0,0,0,0,0,0])
    players = [GenPlayer(0, psi1_weigts), GenPlayer(1, diff_weigts)]
    game = Game(players)
    winner = game.start()
    
    # players = {"60k": MCTS_Player(0,60000),
    #            "120k": MCTS_Player(0,120000),
    #            "psi1": GenPlayer(0, psi1_weigts),
    #            "ch1": GenPlayer(0, gen2_weigts),
    #            "diff": GenPlayer(0, diff_weigts),
    #            }
    
    # #60k
    # P = players["60k"] ; P.idx = 0
    # win = []
    # for name, player in players.items():
    #     if name != "60k":
    #         player.idx = 1
    #         game = Game([P,player])
    #         winner = game.start()
    #         if winner == 0:
    #             win.append(1)
    #         else :
    #             win.append(0)
                
    # win_rate_60k = np.mean(np.array(win))
    # import ipdb; ipdb.set_trace()