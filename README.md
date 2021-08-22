# AlphaQuoridor

Quoridor is not so famous game and there is not a lot of research about it. It has a state-spacecomplexity similar to Chess with a higher game-tree complexity. We will code this game like AlphaGo Zero from Deepmind, which learns to play the game by self playing without human games data.

![](https://i.imgur.com/C8KoNNn.png)

Quoridor is a board game that involves both spatial intuition and strong logical analysis -- in brief, each of the players try to get their piece to the other side of the board while walls are being placed by each player. 

We did : 
- First, explore baseline search methods (e.g. alpha beta seach) to gain a stronger understanding of the deficiencies in that elementary approach.
- Then, use heuristics features combined with genetic algorithms to get the best weight of these features.
- After that, we test MCTS method to find how good it is compare to the previous one

It has to be done:
- Try to combine the deep learning approach with MCTS like AlphaGo Zero.
