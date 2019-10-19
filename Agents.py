from Skills import Skills
from Utils import Cell, Params, edist
from AgentEnv import Agent_Environment
from astar import astar
import sys
import numpy as np
import copy

class Agent():
    
    def __init__(self, env, agid):
        
        self.id = agid  ###starts with 0
                
        self.env =  env ###env object

        self.type = None
        
        self.mu = []
        
        self.oset = []

        self.option = -1
        
        self.obsBound = self.env.scan_region  ##observation boundary
        
        self.savecount = 0
        self.savemax = Params.params[1]
        
        self.Loc = [-1, -1] ##location of agent ##during init, it is the start location

        self.skills = Skills(self)
        
        self.env.w.agentTable.append([self.Loc[0], self.Loc[1], self.option, self.type])
        self.idx = len(self.env.w.agentTable)-1  ###index in table
        
        self.path = []
        self.tar = []
        
        self.color = ''
        
    
    
    def clear(self):  ####clear any data used for an option
        
        if list(self.tar) in self.env.w.trgtq:
            self.env.w.trgtq.remove(list(self.tar))  ###remove target from queue
            
        self.path = []
        self.tar = []
        
        
    def save_to_table(self):
        self.Loc = self.env.currentcell
        self.option = -1
        self.env.w.agentTable[self.idx] [:2] = self.Loc  ####update location
        self.env.w.agentTable[self.idx] [2] = -1 ###no option
        self.env.w.agentTable[self.idx] [3] = self.type
        
        
    def reset(self):
        
        self.env.agent_id = self.id
        self.env.agent_type = self.type
        
        self.env.reset()
        
        self.savecount = 0
        self.path = []
        self.tar = []
        
        self.Loc = self.env.currentcell
        self.option = -1
        self.env.w.agentTable[self.idx] [:2] = self.Loc  ####update location
        self.env.w.agentTable[self.idx] [2] = -1 ###no option
        self.env.w.agentTable[self.idx] [3] = self.type
        
        
        self.env.situupdate(self.Loc)
        s = self.env.getstate()
        
        
        return s
        
    
    
    def filterOption(self, option, typ):
        
                if option not in self.oset:   ###### Unit's rejection
                        return 1

                if typ > 0:

                    l1 = 0
                    l2 = l1 + ( len(self.env.option_centroids[0]) + 1 )  ### 
                    l3 = l1 + l2 + ( len(self.env.option_centroids[1]) + 1 )
                    l4 = l1 + l2 + l3 + ( len(self.env.option_centroids[2]) + 1 )
                    l5 = l1 + l2 + +l3 + l4 + ( len(self.env.option_centroids[3]) + 1 )
                    
       
                    if option >= l1+1  and option < l1+1+len(self.env.option_centroids[0]): ### move                     
                        site = option - (l1+1)            
                        if (self.env.unknownmap[site] == 0):                           
                            return 1
 
       
                    elif option >= l2+1  and option < l2+1+len(self.env.option_centroids[1]): ### save                                       
                        site = option - (l2+1)
                        if self.env.criticalvictim_map[site] == 0:                     
                            return 1   
    

                    elif option >= l3+1  and option < l3+1+len(self.env.option_centroids[2]): ###fetch victim                 
                        site = option - (l3+1)                
                        if self.env.stablevictim_map[site] == 0:
                            return 1
                            
                            
                    elif option >= l4+1  and option < l4+1+len(self.env.option_centroids[3]): ###clear debris             
                        site = option - (l4+1)
                        if self.env.is_debris[site] == 0:
                            return 1
                    
                                        
                    elif option >= l5+1  and option < l5+1+len(self.env.option_centroids[4]): ###clear blockage           
                        site = option - (l5+1)
                        if self.env.is_blockage[site] == 0:
                            return 1
                    
                    
                return 0
                
                
        
    def stepLevel1(self, option, prnt=0):  ### primitive step
        """
        The agent takes an option, and local observation...invokes a skill
        """
        
        self.terminal = 0
        
        self.insteps = 1

        self.env.w.agentTable[self.idx][:2] = self.Loc
        self.env.w.agentTable[self.idx][2] = option  ##store option

        
        def execute(option):

                if option not in self.oset:   ###### Unit's rejection
                    option = -2
                    return option
                                
                l1 = 0
                l2 = l1 + ( len(self.env.option_centroids[0]) + 1 )  ### 
                l3 = l1 + l2 + ( len(self.env.option_centroids[1]) + 1 )
                l4 = l1 + l2 + l3 + ( len(self.env.option_centroids[2]) + 1 )
                l5 = l1 + l2 + +l3 + l4 + ( len(self.env.option_centroids[3]) + 1 )
                
                
                
                if option == l1: ### scan
                    
                    if not self.env.situ[1]:   #### local unexplored area flag -- homeostatic
                        option = -2                
                        self.terminal = 1
                        return option   
                                                
                    ret = self.skills.scan(Cell.unknown)

          
                              
                elif option >= l1+1  and option < l1+1+len(self.env.option_centroids[0]): ### navigate (only for search agents) 
                    
                    site = option - (l1+1)

                    if (self.env.situ[1] > 0) or (self.env.unknownmap[site] == 0):  
                        option = -2 
                        self.terminal = 1
                        return option
                    
                    
                    if self.tar == []:
                                ###choose nearest tar
                                
                                mn = np.inf ##min
                                t = []
                                for p in self.env.siteunknowns[site]:
                                    if list(p) not in self.env.w.trgtq:
                                        d = edist(p, self.Loc)  ###distance
                                        if d<mn:
                                            mn = d
                                            t = p
                                  
                                self.tar = t

                    
                    self.skills.move_to(tuple(self.tar), Cell.unknown)
                            
                
        
                elif option == l2: ### fetch medicine/life supply
                    
                    if self.env.situ[2] == 1: ###medicine ##life supply
                        option = -2
                        self.terminal = 1
                        return option
                    
                    self.tar = self.env.w.attrList[Cell.station][0]
                    
                    self.skills.move_to(tuple(self.tar), Cell.station)
                    ret = self.skills.fetch(Cell.station, option)

                        
                        
                        
                elif option >= l2+1  and option < l2+1+len(self.env.option_centroids[1]): ### save 
                                        
                    site = option - (l2+1)
                    
                    if self.env.criticalvictim_map[site] == 0:
                        option = -2
                        self.terminal = 1
                        return option

                    if self.tar == []:
                                ###choose nearest tar
                                
                                mn = np.inf ##min
                                t = []
                                for p in self.env.sitevictims1[site]:
                                    if (list(p) not in self.env.w.trgtq) or (len(self.env.sitevictims1[site])==1):
                                        pth = astar(self, p)
                                        d = len(pth)  ###distance
                                        
                                        ####### now path length can be used to forecast the expected health of victim ######
                                        '''exp_health = self.env.w.health_mat[(p[0], p[1], 0)] - (d*self.env.w.health_mat[(p[0], p[1], 1)]) ###current_health - d*decay_rate + err
                                        
                                        if exp_health <= 0.1:   ########## forecast says that this victim is beyond saving
                                            exp_health = 10**16
                                            d = 10**16   ######## if dead or expected to be dead, ignore'''
                
                                        exp_health = d
                                        
                                        if exp_health < mn:
                                            mn = exp_health
                                            t = p  
                                
                                                
                                self.tar = t
                                
                                #self.env.w.trgtq.append(list(t))
                                
                    
                    self.skills.move_to(tuple(self.tar), Cell.victim_critical)
                    ret = self.skills.save(Cell.victim_critical, option)
 
        
        
                elif option == l3: ### relocate victim
                    
                    if not self.env.situ[3]:  ### carried victim count
                        option = -2
                        self.terminal = 1
                        return option

                    self.tar = self.env.w.attrList[Cell.station][0]
                    
                    self.skills.move_to(tuple(self.tar), Cell.station)  ###because base station is at same location, change this later...make it generic!!!
                    ret = self.skills.relocate(option) 

            
            
                elif option >= l3+1  and option < l3+1+len(self.env.option_centroids[2]): ###fetch victim
                    
                    site = option - (l3+1)
                    
                    if self.env.stablevictim_map[site] == 0:
                        option = -2
                        self.terminal = 1
                        return option
                        


                    if self.tar == []:
                                ###choose nearest tar
                                
                                mn = np.inf ##min
                                t = []
                                for p in self.env.sitevictims2[site]:
                                    if (list(p) not in self.env.w.trgtq) or (len(self.env.sitevictims2[site])==1):
                                        pth = astar(self, p)
                                        d = len(pth)  ###distance
                                        
                                        ####### now path length can be used to forecast the expected health of victim ######
                                        '''exp_health = self.env.w.health_mat[(p[0], p[1], 0)] - (d*self.env.w.health_mat[(p[0], p[1], 1)]) ###current_health - d*decay_rate + err
                                        
                                        if exp_health <= 0.1:   ########## forecast says that this victim is beyond saving
                                            exp_health = 10**16
                                            d = 10**16   ######## if dead or expected to be dead, ignore'''
                
                                        exp_health = d
                                        
                                        if exp_health < mn:
                                            mn = exp_health
                                            t = p  
                                
                                                
                                self.tar = t
                                
                                #self.env.w.trgtq.append(list(t))
                                

                    self.skills.move_to(tuple(self.tar), Cell.victim_stable)
                    ret = self.skills.fetch(Cell.victim_stable, option)
                
                
                
                elif option >= l4+1  and option < l4+1+len(self.env.option_centroids[3]): ###clear debris
                    
                    site = option - (l4+1)
                    
                    if self.env.is_debris[site] == 0:
                        option = -2
                        self.terminal = 1
                        return option
                        
                    if self.tar == []:
                                ###choose nearest tar
                                
                                mn = np.inf ##min
                                t = []
                                for p in self.env.sitedebris[site]:
                                    if list(p) not in self.env.w.trgtq:
                                        d = edist(p, self.Loc)  ###distance
                                        if d<mn:
                                            mn = d
                                            t = p
                                  
                                self.tar = t
                                
                    self.skills.move_to(tuple(self.tar), Cell.debris)
                    ret = self.skills.clear_debris()
                    
                    
                elif option >= l5+1  and option < l5+1+len(self.env.option_centroids[4]): ###clear blockage
                    
                    site = option - (l5+1)
                    
                    if self.env.is_blockage[site] == 0:
                        option = -2
                        self.terminal = 1
                        return option
                    
                    if self.tar == []:
                                ###choose nearest tar
                                
                                mn = np.inf ##min
                                t = []
                                for p in self.env.siteblockage[site]:
                                    if list(p) not in self.env.w.trgtq:
                                        d = edist(p, self.Loc)  ###distance
                                        if d<mn:
                                            mn = d
                                            t = p
                                  
                                self.tar = t
                           
                    self.skills.move_to(tuple(self.tar), Cell.path_blockage)
                    ret = self.skills.clear_blockage()
                    
                        
                elif option == Params.NONE:
                    self.skills.do_nothing()

                        
                return option
                    
                
                
        option = execute(option)
        
        self.env.w.agentTable[self.idx][:2] = self.Loc
        
        return option
        
        

def registerAgents(w, c, args):
    
    numagents = 0
    
    agents = []
    
    
    if c == 1:
        pass

    elif c==6:
        
        num_agents_of_helper_type = 1
        num_agents_of_other_type = 4
        
        numagents = num_agents_of_other_type * 3 + num_agents_of_helper_type * 1 ### num_agents_of_other_type * 3 because we have 3 types: search, aid, relocate
        agid = 0
        
        for i in range(numagents):
            env = Agent_Environment(w, args)
            agents.append(Agent(env, agid))   
            agid += 1
    
        l1 = 0

        agn = -1
        
        for j in range(num_agents_of_other_type):
            agn += 1
            agents[agn].oset = [l1] + [l1+i+1 for i in range(len(agents[agn].env.option_centroids[0]))] 
            agents[agn].type = 'Search'
            agents[agn].color = 'yellow'
        
        
        l1 += len(agents[agn].oset)
        
        for j in range(num_agents_of_other_type):
            agn += 1
            agents[agn].oset = [l1] + [l1+i+1 for i in range(len(agents[agn].env.option_centroids[1]))]
            agents[agn].type = 'Aid'
            agents[agn].color = 'magenta'

        
        l1 += len(agents[agn].oset)
        
        
        for j in range(num_agents_of_other_type):
            agn += 1
            agents[agn].oset = [l1] + [l1+i+1 for i in range(len(agents[agn].env.option_centroids[2]))]
            agents[agn].type = 'Relocate'
            agents[agn].color = 'cyan'
        
        l1 += len(agents[agn].oset)
        
        for j in range(num_agents_of_helper_type):
            agn += 1
            agents[agn].oset = [l1] + [l1+i+1 for i in range(len(agents[agn].env.option_centroids[3]) + len(agents[agn].env.option_centroids[4]))]
            agents[agn].type = 'Helper'
            agents[agn].color = 'green'

        l1 += len(agents[agn].oset)
        

        x = l1
            
        Params.NONE = x  ### NONE option
        
        for i in range(numagents):
            agents[i].oset = agents[i].oset + [Params.NONE]
            agents[i].save_to_table()
            #print('.', agents[i].oset)
                
        args.noptions = x + 1
        
        
        
    return agents
    