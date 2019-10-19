import numpy as np
from Utils import Cell, edist
from astar import astar
import copy, sys

class Skills():
    def __init__(self, caller):
        self.cag = caller ###calling agent
        
        self.surround = [np.array((1,0)), np.array((1,1)), np.array((0,1)), np.array((-1,1)), np.array((-1,0)), np.array((-1,-1)), np.array((0,-1)), np.array((1,-1))]
                    

        self.timeout = 1
        

        
    def do_nothing(self):
        return
        
        
    def self_directed_move(self, direction):
        ### agent moves to the edge of observed region
        
        pass
        
 
                    
        
    def move_to(self, target, arg):
        ### targeted movement  


        target = tuple(self.cag.tar)
        
        if not target:
            return
            
                    
        in_path = 0
        if (self.cag.Loc in self.cag.path) and (target in self.cag.path):
            if self.cag.path.index(self.cag.Loc) < self.cag.path.index(target):
                in_path = 1
        if in_path:        
            path = copy.deepcopy(self.cag.path[self.cag.path.index(self.cag.Loc):])
        else:
            path = astar(self.cag, target, allowun = int(arg == Cell.unknown))        
            self.cag.path = copy.deepcopy(path)       
        
        
        cnt = 0
        

        for p in path[1:len(path)-1]:
                
            cnt+=1
            #print(p)
            if self.cag.env.grid[tuple(p)] != Cell.filled:    ###prevent invalid move
                self.cag.Loc = tuple(p)
                #print('newloc', self.cag.Loc)
                
                if list(p) in self.cag.env.w.attrList[Cell.unknown]:
                    self.cag.env.w.situation[tuple(p)] = self.cag.env.grid[tuple(p)]
                    self.cag.env.w.attrList[Cell.unknown].remove(list(p))

                    
            self.cag.insteps = cnt
            
            #self.cag.env.w.show()
            
            if cnt>=self.timeout:                    
                break

 
 
           
    #### this is crucial; line-of-sight scanning ####
    def scanCell(self, lo, cur, b, scan_count, dmax, moveto, sgn1, sgn2, typ):
        
        for x in range(b):   #### base
            
                    if typ == 'horizontal': 
                        dx = sgn1*x
                        dy = sgn2*(lo*dx)/float(b)
                        cel = tuple([int(cur[0]+(dy)), int(cur[1]+(dx))])
                    elif typ == 'vertical':
                        dx = sgn1*x
                        dy = sgn2*(lo*dx)/float(b)
                        cel = tuple([int(cur[0]+(dx)), int(cur[1]+(dy))])
                        
                    
                    if cel[0]>=0 and cel[0]<self.cag.env.size and cel[1]>=0 and cel[1]<self.cag.env.size:
                        celval = self.cag.env.grid[cel]  ##### note that the gridmap is the true world underlying the partial situation map
                        
                        if celval == Cell.filled:
                            break  ###break scan line
                    
                        else:
                            if list(cel) in self.cag.env.w.attrList[Cell.unknown]:
                                scan_count += 1
                                self.cag.env.w.situation[cel] = celval         
                                if celval != Cell.vacant: self.cag.env.w.attrList[celval].append(list(cel))
                                                        
                                self.cag.env.w.attrList[Cell.unknown].remove(list(cel)) ##remove from unknown
                                self.cag.env.w.scanlist.append(list(cel))
  
                                if celval == Cell.victim_critical:
                                    ##add to object list
                                    self.discovered_victims.append(cel)
                                        
                                d = edist(cel, self.cag.Loc)

                                if (d > dmax) and self.cag.env.w.situation[cel]==Cell.vacant:
                                    moveto = cel
                                    dmax = d  

        
        return moveto, dmax, scan_count
                          

        
                              
    def scan(self, arg):
        
        ret = 0
        
        cnt = 0
        b = self.cag.obsBound

        while (cnt < 1):
            cnt += 1 ##scan iterations
            
            cur = self.cag.Loc
            
            scan_count = 0 ##cells converted
            dmax = 0  ###farthest scanned cell dist
            moveto = cur
            
            self.discovered_victims = []
            
            
            #### line-of-sight scanning ######
            ### line orientation multiplier####
            typ = 'horizontal'
            for lo in np.linspace(0,b,5): ####line or ray
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, 1, 1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, 1, -1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, -1, 1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, -1, -1, typ)
                
            typ = 'vertical'
            for lo in np.linspace(0,b,5): ####line or ray
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, 1, 1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, 1, -1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, -1, 1, typ)
                moveto, dmax, scan_count = self.scanCell(lo, cur, b, scan_count, dmax, moveto, -1, -1, typ)


            self.cag.Loc = moveto     ###move agent to boundary 
            

            ret = self.cag.env.w.on_scan(scan_count, self.discovered_victims, self.cag.id)
            
            
        return ret
            
            
            
    def fetch(self, arg, option):

            ret = 0
            
            for sn in self.surround:
    
                cel = tuple(self.cag.Loc + sn)
                if self.cag.env.w.situation[cel] == arg:
                    
                    if arg == Cell.station:
                        self.cag.env.med = 1                      
                        ret = 0
                        
                        
                    elif arg == Cell.victim_stable:
                        if self.cag.env.victims_for_relocation == self.cag.env.maxreloc:
                            return ret #### cannot carry more than 3 victims at a time
                            
                        self.cag.env.victims_for_relocation += 1
                        
                        ret = self.cag.env.w.on_carry(cel, self.cag.id)
                        
                        
            return ret


                                          
    def save(self, arg, option):
        
            ret = 0
            
            for sn in self.surround:
                    cel = tuple(self.cag.Loc + sn)  ####validate cel
                    #print("save", cel)
                    if self.cag.env.w.situation[cel] == Cell.victim_critical and self.cag.env.med:
    
                                self.cag.savecount +=1 
           
                                if self.cag.savecount == self.cag.savemax:
                                    self.cag.env.med = 0 ###limited supply!!
                                    self.cag.savecount = 0

                                ret = self.cag.env.w.on_aid(cel, self.cag.id)
                                
                                
            return ret
        
                                   
    def relocate(self, option):
            
            ret = 0
            
            for sn in self.surround:
                cel = tuple(self.cag.Loc + sn)
                if self.cag.env.w.situation[cel] == Cell.station:
                    ###agent is at base station
                    

                    ret = self.cag.env.w.on_relocation(self.cag.id)
                    self.cag.env.victims_for_relocation = 0
                    
                    
            return ret
            
            
    def clear_debris(self):
        
        for sn in self.surround:
                cel = tuple(self.cag.Loc + sn)
                if self.cag.env.w.situation[cel] == Cell.debris:
                    self.cag.env.w.on_clear_debris(list(cel))
                    
                    
    def clear_blockage(self):
        
        for sn in self.surround:
                cel = tuple(self.cag.Loc + sn)
                if self.cag.env.w.situation[cel] == Cell.path_blockage:
                    self.cag.env.w.on_clear_blockage(list(cel))
                    