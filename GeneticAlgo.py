from random import *
from pylab import*
from numpy.random import normal
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pdb

#imports pour algo genet
import getopt

from players.GenPlayer import *
from game import *

import pdb
import random
import pyspiel
import numpy as np
import os

from threading import Thread

import datetime
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/genetics_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


class Agent():
    """
    This class defines an agent
    This agent has two attributes
    id : this agent id
    weights : the list of the LEN_WEIGHTS weigths
    """

    def __init__(self,idt,weigths):
        self.id = idt
        self.weights = weigths
        self.cache = []

class GeneticAlgorithm:
    def __init__(self, pop_settings):
        self.settings = pop_settings
        self.run_logs = False
        self.gen_done = {}
        
    def random_agent(self, idx):
        """
        This function returns a random agent :
        random_agent.id = "random_agent"
        random_agent.weigths = a list of length LEN_WEIGHTS
        with random weigths
        """
        weights = np.array([ np.random.uniform(-1, 1) * (np.random.rand()< self.settings["HASFEAT"])*1 for _ in range(self.settings["LEN_WEIGHTS"])])
        return Agent(idx,weights)
        
    def logs(self):
        while self.run_logs :
            if not self.curr_gen in self.gen_done:
                for i in range(self.settings["LEN_WEIGHTS"]):
                    writer.add_scalar("%s/avg_weights_%i"%(self.settings["NAME"], self.curr_gen), self.avg_weights[i], i)
                    
                for i, agent in enumerate(self.current_pop):
                    for j in range(self.settings["LEN_WEIGHTS"]):
                        writer.add_scalar("genration_%i/chomosome%i/weights"%(self.curr_gen, i), agent.weights[j], j)

                self.gen_done[self.curr_gen] = True
                    
            for i, agent in enumerate(self.current_pop):
                win_rate = sum(agent.cache)/len(agent.cache) if len(agent.cache)!=0 else 0
                writer.add_scalar("genration_%i/chomosome%i/win_rate"%(self.curr_gen, i), win_rate, len(agent.cache))
    
    def start_evolution(self):
        """
        This function computes a genetic selection
        """
        list_pop = list()
        
        self.current_pop = [self.random_agent(i) for i in range(self.settings["POP_LEN"])]
        
        logs_thread = Thread(target=self.logs)
        list_pop.append(self.current_pop)
        
        try : os.makedirs("saves/%s"%self.settings["NAME"])
        except : print("folder already exist")
        
        for j in range(self.settings["GENETIC_STEPS"]):
            
            self.curr_gen = j
            np.save("saves/%s/%s.npy"%(self.settings["NAME"], j), np.array([a.weights for a in self.current_pop]))
            self.avg_weights = np.mean(np.array([a.weights for a in self.current_pop]), axis=0)
            
            if not self.run_logs: 
                self.run_logs = True
                logs_thread.start()
            
            clash_of_titans = [Thread(target=self.clash_group, args=(agent, self.current_pop[id0+1:])) for id0, agent in enumerate(self.current_pop)]
            for battle in clash_of_titans:
                battle.start()
            
            for battle in clash_of_titans:
                battle.join()
                        
            self.current_pop = self.selection(self.current_pop)
            list_pop.append(self.current_pop)
            
        return list_pop
    
    def clash_group (self, agent, group):
        for op in group:
            self.clash(agent,op)
            self.clash(op, agent) 
    
    def clash(self, agent1,agent2):
        """                         
        This function makes a round between two agents
        """
        name1 = "player1"
        w1 = agent1.weights
        name2 = "player2"
        w2 = agent2.weights
        players = [GenPlayer(0, w1), GenPlayer(1, w2)]
        game = Game(players)
        winner = game.start()
        
        if winner==0: #If player1 wins
            agent1.cache.append(1)
            agent2.cache.append(0)
        elif winner==1: #If player2 wins
            agent2.cache.append(1) 
            agent1.cache.append(0)
        else:
            agent2.cache.append(0) 
            agent2.cache.append(0)                                             
    
    def selection(self, list_agents):
        """
        This algorithm creates a new generation of agents, by selecting them 
        with a roulette_wheel selection
        """
        
        ranked_agents = sorted(list_agents, key=lambda x: sum(x.cache), reverse=True)
        
        C_p = [sum(play.cache) for play in ranked_agents]
        tot_score = sum(C_p)
        p = np.array(C_p)/tot_score if tot_score != 0 else [1/len(C_p)] * len(C_p)
        
        n_elit = self.settings["ELITISM"]*self.settings["POP_LEN"]
        elit = int(ranked_agents[:n_elit])
        
        new_agents = list()
        new_agents += elit
        for i in range(len(list_agents)-n_elit):
            agent1 = np.random.choice(list_agents, p=p)
            agent2 = np.random.choice(list_agents, p=p)
            
            agent3 = self.crossover(agent1,agent2, i)
            new_agents.append(agent3)
            
        return new_agents
        
    def crossover(self, agent1, agent2, i):
        """
        This function computes a crossover of a new child of agent1 and agent2
        """
        mat1 = agent1.weights
        mat2 = agent2.weights
        mat = [0]*self.settings['LEN_WEIGHTS']
        
        for j in range(self.settings['LEN_WEIGHTS']):
            mat[j] = mat1[j] if np.random.rand()>0.5 else mat2[j]
            if np.random.rand() < self.settings['MUTATION']:
                if mat[j] == 0: mat[j] = np.random.uniform(-1,1)
                else :
                    mat[j] = 0 if np.random.rand() < self.settings['LOSEFEAT'] else np.random.uniform(-1,1) * self.settings['MUTFEAT']
                    
        agent_child = Agent(i,mat)
        return agent_child
    
    
if __name__ == "__main__":
    pop_sett = {
                "NAME": "psy1",
                "LEN_WEIGHTS" : 9,
                "HASFEAT": 0.3,
                "MUTATION" : 0.99,
                "POP_LEN" : 10,
                "ELITISM" : 0.2,
                "GENETIC_STEPS" : 5,
                "LOSEFEAT" : 0.9,
                "MUTFEAT" : 0.1,
                "MOVECAP":120
                }
    
    GA = GeneticAlgorithm(pop_sett)
    GA.start_evolution()
