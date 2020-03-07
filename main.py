from game import *
from players.RandomPlayer import *
from players.GenPlayer import *

if __name__ == "__main__":
    players = [GenPlayer(0, [1]*9), GenPlayer(1, [1]*9)]
    game = Game(players)
    winner_hist = game.start(rounds=2)
    import ipdb; ipdb.set_trace()