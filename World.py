import numpy as np
import copy
import cv2
import sys
import random
from Utils import Cell, Params
import matplotlib.pyplot as plt


global base_map
base_map = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
ret, base_map = cv2.threshold(base_map,150,1,cv2.THRESH_BINARY_INV)
###  Dilate ###
kernel = np.ones((5,5),np.uint8)
base_map = cv2.dilate(base_map,kernel,iterations = 1)
### Resize ###
sz = 100
base_map = cv2.resize(base_map, (sz, sz), interpolation = cv2.INTER_LINEAR)


class World():  ####base world
    
    def __init__(self, size, enable_global_reward=1):
        
        self.size = size
        self.grid = np.zeros((self.size,self.size))

        self.agentTable = []  ###store location and option
        
        ###### Attributes in true world #######
        self.victimCount = Params.params[0]
        
        print('vic count', self.victimCount)
        
        self.begin = [(50,5), (50,95), (5,50), (95,50)]  ### possible starting locations
        #######################################


        
        ########## attr. lists ########################
        self.attrList = {}
        self.attrList[Cell.unknown] = []
        self.attrList[Cell.station] = []
        self.attrList[Cell.victim_critical] = []
        self.attrList[Cell.victim_stable] = []
        self.attrList[Cell.debris] = []
        self.attrList[Cell.path_blockage] = []
        
        
        self.health_mat = np.zeros((self.size, self.size, 2))  ### this matrix keeps record of decaying health of victims (index 0) and the decay rate (index 1), 
                                                            ### although keeing a separate matrix increases memory requirement,
                                                            ### this is preferred due to direct addressing vs. item search in attrList which would have
                                                            ### more time complexity 
        ###############################################
        
        
        ################# make ground truth ################################   
        if 1:
            self.grid = copy.deepcopy(base_map)
            self.size = base_map.shape[0]            
            ### Secure border ###
            for i in range(self.size):
                self.grid[i][0] = 1
                self.grid[i][self.size-1] = 1
                    
            for j in range(self.size):
                self.grid[0][j] = 1
                for z in range(2):
                    self.grid[self.size-z-1][j] = 1
        #####################################################################        
        
        ################## add obstacles #########################
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] == 0:
                    self.grid[i,j] = Cell.vacant
                    self.attrList[Cell.unknown].append([i,j])          ###### every vacant cell in base world is unknown
                elif self.grid[i,j] == 1:
                    self.grid[i,j] = Cell.filled
                    #self.attrList[Cell.unknown].append([i,j])       ###### and so is filled cell
                else:
                    #print('??', i,j, self.grid[i,j])
                    pass

        
        ############# add blockage ################################
        for i in range(20, 25):
            for j in range(53, 56):
                self.grid[i,j] = Cell.path_blockage
                
        ##################### add victims ###########################
        vc = self.victimCount
        self.victims_under_debris = []
        under_debris = random.sample(range(vc), 2) ##### two random victims will be under debris
        for z in range(vc):
            i = np.random.randint(2, self.size-2)
            j = np.random.randint(2, self.size-2)
            while self.grid[i, j] != Cell.vacant:
                i = np.random.randint(2, self.size-2)
                j = np.random.randint(2, self.size-2)

            self.grid[i,j] = Cell.victim_critical
            if z in under_debris:
                self.victims_under_debris.append([i,j])
                self.grid[i,j] = Cell.debris

  
        

        #########################################################################################
    
        
        
        ######## backup ###################
        self.attrListB = copy.deepcopy(self.attrList)
        ########################################

        plt.matshow(self.grid)
        plt.show()
        
        self.reset()
        
    
    def reset(self):
              
        self.situation = np.ones((self.size,self.size))*Cell.unknown
               
        self.victims_found = 0
        self.victims_relocated = 0
        self.relocation_points = 0
        self.relocation_reward = 0
        self.searchsteps = 1e-15
        self.num_of_deaths = 0
        self.pre_discovery_health = 1
                
        self.trgtq = []
        self.scanlist = []
        
        self.attrList = copy.deepcopy(self.attrListB)
        
        self.health_mat = np.zeros((self.size, self.size, 2))
        self.health_of_carried_victims = {}
                    
        self.station = self.begin[np.random.choice(len(self.begin))]    ##### change to a set of different locations, chose randomly from the set
        self.grid[self.station] = Cell.station
        self.attrList[Cell.station].append(self.station)
        self.attrList[Cell.unknown].remove(list(self.station))  ###station is known
        self.situation[self.station] = self.grid[self.station]
        
        
        for i in range(-1,2):      #### also, let the station's surrounding be known!!!
            for j in range(-1,2):
                cel = (self.station[0] + i, self.station[1]+j)
                if list(cel) in self.attrList[Cell.unknown]: 
                    self.attrList[Cell.unknown].remove(list(cel))  ###station is known
                    self.situation[cel] = self.grid[cel]
        
        
        ##########list of permissible initiation cells for agents##################
        self.validcells = []
        for i in range(self.size):
            for j in range(self.size):
                if self.situation[i,j] == Cell.vacant:
                    self.validcells.append((i,j))
        
        #self.validcells = [(4,6)]
        ###############################################################

        self.fig_cntr = 0
        
                
        self.decay_rate = np.zeros(4)
        
        self.decay_rate[1] = 200
        #print(self.decay_rate[0])
        self.decay_rate[2] = self.decay_rate[1]*2
        self.decay_rate[3] = self.decay_rate[2]*5
        
        self.decay_rate[0] = self.decay_rate[1] ### decay denominator before discovery
        
        #print("DECAY RATE", self.decay_rate[0])
        
        self.unknown_area = len(self.attrList[Cell.unknown])
        
        
    
    def decayAmount(self, stage):
        
        return 1 / self.decay_rate[stage] ### decay rate; can be randomized
    

    def decayHealth(self):
        
        self.pre_discovery_health -= self.decayAmount(stage = 0)
        
        for v in self.attrList[Cell.victim_critical]:
            v = tuple(v)
            if self.health_mat[(v[0], v[1], 0)] >= 0:
                self.health_mat[(v[0], v[1], 0)] -= self.health_mat[(v[0], v[1], 1)]
            elif self.health_mat[(v[0], v[1], 0)] < 0 and self.health_mat[(v[0], v[1], 0)] > -np.inf:   
                self.health_mat[(v[0], v[1], 0)] = -np.inf    #### victim is dead!!!!!!!!!!!!!!!    
                self.num_of_deaths += 1

        
        for v in self.attrList[Cell.victim_stable]:
            v = tuple(v)
            if self.health_mat[(v[0], v[1], 0)] >= 0:
                self.health_mat[(v[0], v[1], 0)] -= self.health_mat[(v[0], v[1], 1)]
            elif self.health_mat[(v[0], v[1], 0)] < 0 and self.health_mat[(v[0], v[1], 0)] > -np.inf:   
                self.health_mat[(v[0], v[1], 0)] = -np.inf    #### victim is dead!!!!!!!!!!!!!!!    
                self.num_of_deaths += 1

  
        for agent_id in self.health_of_carried_victims:
            for i in range(len(self.health_of_carried_victims[agent_id])):
                if self.health_of_carried_victims[agent_id][i][0] >= 0:
                    self.health_of_carried_victims[agent_id][i][0] -= self.health_of_carried_victims[agent_id][i][1]
                    
                elif self.health_of_carried_victims[agent_id][i][0] < 0 and self.health_of_carried_victims[agent_id][i][0] > -np.inf:
                    self.num_of_deaths += 1
                    self.health_of_carried_victims[agent_id][i][0] = -np.inf
        


            
    def on_scan(self, scan_count, discovered_victims, which_agent):
 
        for cel in discovered_victims:
            self.victims_found += 1
            self.health_mat[(cel[0], cel[1], 0)] = self.pre_discovery_health
            self.health_mat[(cel[0], cel[1], 1)] = self.decayAmount(stage = 1)
        
        return scan_count
               
            
    def on_aid(self, cel, which_agent):
        
        ret = 0
        
        if self.health_mat[(cel[0], cel[1], 0)] >= 0: ### still alive
            self.health_mat[(cel[0], cel[1], 0)] += 1 #### bump back to good health
            
            ret = self.health_mat[(cel[0], cel[1], 0)]
            
        self.health_mat[(cel[0], cel[1], 1)] = self.decayAmount(stage=1)    
           
        
        self.situation[cel] = Cell.victim_stable
        self.attrList[Cell.victim_critical].remove(list(cel))
        self.attrList[Cell.victim_stable].append(list(cel))
        
        return ret
        
        


    def on_carry(self, cel, which_agent):
        
        ret = 0
        
        if self.health_mat[(cel[0], cel[1], 0)] >= 0: ### still alive
            self.health_mat[(cel[0], cel[1], 0)] += 1 #### bump back to good health

            ret = self.health_mat[(cel[0], cel[1], 0)]
            
        self.health_mat[(cel[0], cel[1], 1)] = self.decayAmount(stage=2)
        
                
        temp = copy.deepcopy(self.health_mat[(cel[0], cel[1])])
        if which_agent not in self.health_of_carried_victims:
            self.health_of_carried_victims[which_agent] = []

        self.health_of_carried_victims[which_agent].append(temp)
  
        self.health_mat[(cel[0], cel[1], 0)] = 0  ### clear cell      
        self.situation[cel] = Cell.vacant
        self.attrList[Cell.victim_stable].remove(list(cel))  ###remove victim from list since it is being relocated
  
        return ret

        
    def on_relocation(self, which_agent):
        
        self.victims_relocated += len(self.health_of_carried_victims[which_agent])

        temp = 0
        for h in self.health_of_carried_victims[which_agent]:
            if h[0] < 0:
                h[0] = 0
            self.relocation_points += h[0]
            temp += 1 / (abs(h[0] - 3) + 1)
            
            self.relocation_reward += 1 if h[0] else -20

        #print('points: ', temp, '; by agent: ', which_agent)
        
        self.health_of_carried_victims.pop(which_agent, None)
        
        return temp
    
    
    def on_clear_debris(self, cel):
        
        if cel in self.victims_under_debris:
            self.victims_found += 1
            self.health_mat[(cel[0], cel[1], 0)] = self.pre_discovery_health
            self.health_mat[(cel[0], cel[1], 1)] = self.decayAmount(stage = 1)
            self.situation[cel] = Cell.victim_critical 
            self.attrList[Cell.victim_critical].append(list(cel))
            self.victims_under_debris.remove(list(cel))
        else:
            self.situation[cel] = Cell.vacant
        
        self.attrList[Cell.debris].remove(list(cel))
        
        
    def on_clear_blockage(self, cel):
        
        self.situation[cel] = Cell.unknown
        self.attrList[Cell.unknown].append(list(cel))
        
        self.attrList[Cell.path_blockage].remove(list(cel))
        

        
    def shared_global_reward(self):
        
        R_search = 0
        R_reloc = 0
        
        R_search = (1/self.searchsteps) * self.search_finish()
        R_search *= 1e-15 ### scale to 1
        '''if R_search != 0:
            print('ok')
            sys.exit()'''
        
        R_reloc = self.relocation_reward * (1 / self.victimCount)  ##max positive reward at a single step can be equal to victim count
        
        self.relocation_reward = 0
        return R_search, R_reloc
        
    
    def search_finish(self):
        a = len(self.attrList[Cell.unknown]) / (self.size**2)
        if a<0.1:
            return 1       
        return 0
        
    def finish(self, timeout):
        
        if timeout:
            victims_unreached = self.victimCount - self.victims_found
            self.num_of_deaths += victims_unreached
            return 1
        
        a = self.search_finish()
        b = len(self.attrList[Cell.victim_critical])
        c = len(self.attrList[Cell.victim_stable])
        d = abs(self.victims_relocated - self.victims_found)   
        
        if a==1 and b==0 and c==0 and d==0:
            return 1
        
        return 0
    
        
           
    
             
                       