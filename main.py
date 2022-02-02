
# coding: utf-8

# In[ ]:

import argparse 
import dill
import sys
from World import World
from ISEMO import runISEMO
from CoHRL import runCoHRL


class args():
    
    discount = 0.99
        
    nruns = 5
    
    nepisodes = 250
    
    nsteps = 25000
    
    eta = 1
    
    coop = 1
    
    agconfig = 6

    lr_term = 0.01
    lr_critic = 0.01
    #epsilon= 0.01
    baseline = True
    temperature=1.0
    
    size = 100
    
    testing = False
    testID = None
    


def makeWorlds():
    
    for i in range(args.nruns):
        
        w = World(args.size)

        print('world made')
        dill.dump({'World':w, 'args': args}, open('MA-World-' + str(i) + '.pl', 'wb')) 
        print('world saved')
        
        
def mainISEMO(noISER):
        fileidx = 'ISEMO_CoopAgents'
        runISEMO(args, fileidx, noISER)


def mainCoHRL(termination_limit, noISER):
        fileidx = 'CoHRL_CoopAgents'
        runCoHRL(args, fileidx, termination_limit, noISER)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_worlds', help='create search and rescue worlds', default=False, action='store_true')
    parser.add_argument('--runCoHRL', help='run CoHRL', default=False, action='store_true')
    parser.add_argument('--testing', help='run test mode', default=False, action='store_true')
    parser.add_argument('--testID', help='world to choose for testing', type=int, default=0)
    

    cmd_args = parser.parse_args()
    
    if cmd_args.make_worlds:
        makeWorlds()
        sys.exit()   
        
    if cmd_args.testing:  ### test execution, no training
        args.testing = True
        args.testID = cmd_args.testID
        args.temperature = 0.5
        
    if cmd_args.runCoHRL:
        noISER_choices = [None, 'Search', 'Aid', 'Relocate', 'Helper', 'All']
        if not cmd_args.testing:
            termination_limit = 150   
            noISER = noISER_choices[5]
        else:
            termination_limit = 150   
            noISER = noISER_choices[5]
  
        mainCoHRL(termination_limit, noISER)
        
    else:
        noISER_choices = [None, 'Search', 'Aid', 'Relocate', 'Helper', 'All']
        if not cmd_args.testing:
            noISER = noISER_choices[0]
        else:  
            noISER = noISER_choices[0]
        
        mainISEMO(noISER)


