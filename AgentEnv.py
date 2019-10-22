import numpy as np
import copy
import sys
import time
from Utils import Cell, Params
            
                
#### state environ of each agent #########
class Agent_Environment:
    
    def __init__(self, wrld, args):
        self.w = wrld
        self.grid = wrld.grid
        self.size = wrld.size
        self.agent_id = -1
        self.agent_type = ''
        
        self.args = args
        
        self.rng = np.random.RandomState(1234)
        
        self.oset = []
                                                            
        self.fignum = 0

        
        self.reset()
                
            
            
    def reset(self):
        
        self.ri = 0
        
        self.med = 0
        self.victims_for_relocation = 0  ### if this agent relocates, how many victims it is carrying
        self.local_un = 0  ####local unknown cells : count        
        self.maxreloc = Params.params[2]
        self.scan_region = Params.params[3]  
        
                  
        self.currentcell = self.w.validcells[np.random.choice(len(self.w.validcells))]   
            
        self.startcell = self.currentcell
        



        self.state_centroids = []
        
        self.num_blocks = 100
        self.n = int(np.sqrt(self.num_blocks))
        
        #print('state resolution: ', self.n)

        ######## define state approximation kernels ###############
        if 1:

            kernel_size = int(self.size/self.n)
            self.kernel_size = kernel_size
            for i in range(self.n):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.state_centroids.append([a, b])

            
            
   
        
        self.option_space_kernels()




    def option_space_kernels(self):
        
        ###################### option space kernels ####################################
        self.option_centroids = [[] for x in range(5)]
        
        self.n1 = int(np.sqrt(self.num_blocks))
        self.n2 = int(np.sqrt(self.num_blocks))
        self.n3 = int(np.sqrt(self.num_blocks))
        

        if 1:

            kernel_size = int(self.size/self.n1)
            for i in range(self.n1):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n1):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.option_centroids[0].append([a, b])

            self.siteunknowns = [[] for x in range(len(self.option_centroids[0]))]
            
            for p in self.w.attrList[Cell.unknown]:
                C, ksite, _ =   self.find_option_membership_kernel(p,0)              
                self.siteunknowns[ksite].append(list(p))
                
            
            self.siteknowns = [[] for x in range(len(self.option_centroids[0]))]
            
            
        if 1:

            kernel_size = int(self.size/self.n2)
            for i in range(self.n2):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n2):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.option_centroids[1].append([a, b]) ### for aid
        
        
        if 1:

            kernel_size = int(self.size/self.n3)
            for i in range(self.n3):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n3):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.option_centroids[2].append([a, b])  ### for relocate
                    
                    
        if 1:

            kernel_size = int(self.size/self.n3)
            for i in range(self.n3):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n3):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.option_centroids[3].append([a, b])     ### for clear debris
                    
                    
        if 1:

            kernel_size = int(self.size/self.n3)
            for i in range(self.n3):
                a = i*kernel_size + int(kernel_size/2)
                for j in range(self.n3):
                    b = j*kernel_size + int(kernel_size/2)
                    
                    self.option_centroids[4].append([a, b])    ### for clear blockage

        

   
    def pseudo_reset(self):
       
       self.situupdate([5,5])
       
                
    def getstate(self):
        
        state = copy.deepcopy(self.situ)
                
        return state



    ##### put this in Utils #######
    def dist(self, p1, p2):
        xd = p1[0] - p2[0]
        yd = p1[1] - p2[1]
        d = xd**2 + yd**2
        return d
        
        
        
    def update(self, loc, agent_id, option, termit):  ### primitive step
   
        if termit:
            old_situ = copy.deepcopy(self.situ)      
            old_agent_sites = copy.deepcopy(self.agents_sites)  ## site number of agents
            old_unknownmap = copy.deepcopy(self.unknownmap)                
            old_criticalvictim_map = copy.deepcopy(self.criticalvictim_map)  ### count
            old_stablevictim_map = copy.deepcopy(self.stablevictim_map)    ### count
                                             

        self.situupdate(loc) ############# features will change now ###############################
        
        
        ISER_to = []
        
        #### precondition checks ########################################################################
        if termit:
            if self.agent_type == 'Search':
                for site in range(len(self.unknownmap)):
                    if self.unknownmap[site] > old_unknownmap[site]: ### new scannable region revealed
                        iser_to = self.get_agents_in_proximity_of(site, old_agent_sites)
                        if iser_to != []:
                            ISER_to += iser_to
                            
                
                if (old_situ[1] != 0) and (self.situ[1] == 0):
                    if old_situ[0] == self.situ[0]:
                        site = old_situ[0]
                        iser_to = self.get_agents_in_proximity_of(site, old_agent_sites)
                        if iser_to != []:
                            ISER_to += iser_to       
                            
                            
            if self.agent_type == 'Aid':
                for site in range(len(self.criticalvictim_map)):
                    if self.criticalvictim_map[site] > old_criticalvictim_map[site]: ### new critical victim revealed
                        iser_to = self.get_agents_in_proximity_of(site, old_agent_sites)
                        if iser_to != []:
                            ISER_to += iser_to
                            
                            
                            
            if self.agent_type == 'Relocate':
                for site in range(len(self.stablevictim_map)):
                    if self.stablevictim_map[site] > old_stablevictim_map[site]: ### new stable victim revealed
                        iser_to = self.get_agents_in_proximity_of(site, old_agent_sites)
                        if iser_to != []:
                            ISER_to += iser_to
        ####################################################################################################
        
        if ISER_to != []:
            ISER_to = list(set(ISER_to))  ### remove duplicates
            print(ISER_to)
            sys.exit()
            
        next_state = self.getstate()      #### new state

        return next_state, ISER_to


    def get_agents_in_proximity_of(self, site, agent_sites):
        
        iser_to = []
        n = self.n
        proximity_sites = [site, site-1 , site+1, site+n, site+n-1, site+n+1, site-n, site-n-1, site-n+1]
        for i in range(len(agent_sites)):
            if (agent_sites[i] in proximity_sites):
                iser_to.append(i)
                
        return iser_to
    
    def find_feature_membership_kernel(self, pt):
        
        kernel_size = self.w.size / np.sqrt(len(self.state_centroids))
        
        c0 = int(pt[0]/kernel_size)*kernel_size + int(kernel_size/2)
        c1 = int(pt[1]/kernel_size)*kernel_size + int(kernel_size/2)
            
        kernel_site = self.state_centroids.index([c0, c1])
            
        return [c0, c1], kernel_site, kernel_size
    
    
    
    def find_option_membership_kernel(self,pt, typ):
        
        kernel_size = self.w.size / np.sqrt(len(self.option_centroids[typ]))
        
        c0 = int(pt[0]/kernel_size)*kernel_size + int(kernel_size/2)
        c1 = int(pt[1]/kernel_size)*kernel_size + int(kernel_size/2)
            
        #kernel_site = int((c0+int(kernel_size/2)) / kernel_size) * kernel_size - 10 + int((c1+int(kernel_size/2)) / kernel_size) 
        kernel_site = self.option_centroids[typ].index([c0, c1])
            
        return [c0, c1], kernel_site, kernel_size
        
        

    def situupdate(self, Loc):
        
        start = time.time()
        
        self.situ = []

        self.currentcell = Loc 

        
        ##### LOCAL ATTRIBUTES ##########################################################################
        C, self._kernel_site, _ = self.find_feature_membership_kernel(self.w.agentTable[self.agent_id][:2]) 
        self.loc_centroid = C

        self.local_un = 0                  #### is local area unexplored               
        for x in range(-1,2):
                for y in range(-1,2):
                        cel = tuple([Loc[0]+x, Loc[1]+y])
                        if cel[0]>=0 and cel[0]<self.size and cel[1]>=0 and cel[1]<self.size:
                            if list(cel) in self.w.attrList[Cell.unknown]:
                                self.local_un = 1 
        
        

        self.situ.append(self._kernel_site / (self.n**2))  #### dummy padding for self
        self.situ.append(self.local_un) 
        self.situ.append(self.med) #### supply
        self.situ.append(self.victims_for_relocation / self.maxreloc) #### capacity
        ################################################################################################


        ##### GLOBAL ATTRIBUTE CALCULATIONS ###############
        if 1:  ##### features for unknown regions
            self.unknownmap = [0 for x in range(len(self.state_centroids))]  
            for p in self.w.attrList[Cell.unknown]:
                C, ksite, _ =   self.find_feature_membership_kernel(p)              
                self.unknownmap[ksite] += 1 / (self.kernel_size**2)   #### there are kernel_size**2 number of cells in one block..so the unknown value represents how many cells are unknown
         
            self.situ += self.unknownmap
            
           
        utopia_value = 0.0 ### everything is fine,there are no victims
        
        if 1: ##### features for critical victims            
            tempcount = [0 for x in range(len(self.state_centroids))] 
            tempminhealth = [utopia_value for x in range(len(self.state_centroids))]
            tempavghealth = [utopia_value for x in range(len(self.state_centroids))]

            for p in self.w.attrList[Cell.victim_critical]:
                
                C, ksite, _ = self.find_feature_membership_kernel(p)

                tempcount[ksite] += 1
                
                hlth = self.w.health_mat[p[0], p[1], 0] / 3.0 ## health of victim, max is 3.0
                if hlth < 0:
                    hlth = 0.0
                
                if hlth < tempminhealth[ksite]:
                    tempminhealth[ksite] = hlth
                
                if tempavghealth[ksite] == utopia_value:
                    tempavghealth[ksite] = hlth
                else:
                    tempavghealth[ksite] += hlth
                
            for i in range(len(tempavghealth)): 
                if tempavghealth[i] != utopia_value: tempavghealth[i] /= tempcount[i]
                
                tempcount[i] /=  15 ##### assume victim count on scale of 30
                
            
            self.criticalvictim_map = tempcount
            
            self.situ += tempcount
            
            self.situ += tempminhealth
            
            self.situ += tempavghealth

        
        
        if 1: ### features for stable victims
            
            tempcount = [0 for x in range(len(self.state_centroids))] 
            tempminhealth = [utopia_value for x in range(len(self.state_centroids))]
            tempavghealth = [utopia_value for x in range(len(self.state_centroids))]

            for p in self.w.attrList[Cell.victim_stable]:
                
                C, ksite, _ = self.find_feature_membership_kernel(p)
                
                tempcount[ksite] += 1
                
                hlth = self.w.health_mat[p[0], p[1], 0] / 3.0 ## health of victim, max is 3.0
                if hlth < 0:
                    hlth = 0.0
                
                if hlth < tempminhealth[ksite]:
                    tempminhealth[ksite] = hlth
                
                if tempavghealth[ksite] == utopia_value:
                    tempavghealth[ksite] = hlth
                else:
                    tempavghealth[ksite] += hlth
                
            for i in range(len(tempavghealth)): 
                if tempavghealth[i] != utopia_value: tempavghealth[i] /= tempcount[i]
                
                tempcount[i] /=  15 ##### assume victim count on scale of 15
                
            self.stablevictim_map = tempcount
            
            self.situ += tempcount
            
            self.situ += tempminhealth
            
            self.situ += tempavghealth
        
        
        if 1:
            self.is_debris = [0 for x in range(len(self.state_centroids))]
            self.is_blockage = [0 for x in range(len(self.state_centroids))]
            
            for p in self.w.attrList[Cell.debris]:              
                C, ksite, _ = self.find_feature_membership_kernel(p)
                self.is_debris[ksite] = 1
                
            self.situ += self.is_debris
            
            for p in self.w.attrList[Cell.path_blockage]:              
                C, ksite, _ = self.find_feature_membership_kernel(p)
                self.is_blockage[ksite] = 1
                
            self.situ += self.is_blockage
        
    
            
        ####### Observe other agents ##################
        self.agents_sites = []
        for i in range(len(self.w.agentTable)): #### iterate over agents
            
                if i == self.agent_id: 
                    self.agents_sites.append(self._kernel_site)
                    continue
                
                C, kernel_site, _ = self.find_feature_membership_kernel(self.w.agentTable[i][:2]) ### find site for the location of agent            
                C[0] = (C[0] - self.loc_centroid[0]) / self.w.size
                C[1] = (C[1] - self.loc_centroid[1]) / self.w.size
    
                self.situ += [kernel_site / (self.n**2)]
            
                self.agents_sites.append(kernel_site)
            
            
        ###### Co-ordination features ##################
        if self.args.coop:
            for i in range(len(self.w.agentTable)): #### iterate over agents
                
                if i == self.agent_id: continue

                if self.w.agentTable[i][3] != self.w.agentTable[self.agent_id][3]: continue
                
                o = self.w.agentTable[i][2] ### option of other agent
 
                tmp = [0 for x in range(self.args.noptions)] #### one-hot init
                tmp[o] = 1
                
                self.situ += tmp
                
                
        
        
        
        #######################################
        self.option_space_update()
        
        end = time.time()
        
        #print(end-start)

        #print(self.situ)
        #sys.exit()

        
        
    def option_space_update(self):
            
            start = time.time()
    
            self.sitevictims1 = [[] for x in range(len(self.option_centroids[1]))]   #### for critical victims
            self.sitevictims2 = [[] for x in range(len(self.option_centroids[2]))]   #### for stable victims
            self.sitedebris = [[] for x in range(len(self.option_centroids[3]))]
            self.siteblockage = [[] for x in range(len(self.option_centroids[4]))]
    
    
            if 1: ## add search case
    
                for p in self.w.scanlist:             
                    C, ksite, kernel_size = self.find_option_membership_kernel(p,0)
                    if list(p) in self.siteunknowns[ksite]:
                        self.siteunknowns[ksite].remove(list(p))
                        self.siteknowns[ksite].append(list(p))
    
    
    
            if 1: ## add aid case
    
                for p in self.w.attrList[Cell.victim_critical]:             
                    C, ksite, _ = self.find_option_membership_kernel(p,1)              
                    self.sitevictims1[ksite].append(list(p))
    
    
    
            if 1: ## add reloc case
    
                for p in self.w.attrList[Cell.victim_stable]:               
                    C, ksite, _ = self.find_option_membership_kernel(p,2)              
                    self.sitevictims2[ksite].append(list(p))
                    
                    
            if 1: 
    
                for p in self.w.attrList[Cell.debris]:               
                    C, ksite, _ = self.find_option_membership_kernel(p,3)              
                    self.sitedebris[ksite].append(list(p))
                    
                    
            if 1: 
    
                for p in self.w.attrList[Cell.path_blockage]:               
                    C, ksite, _ = self.find_option_membership_kernel(p,4)              
                    self.siteblockage[ksite].append(list(p))
   