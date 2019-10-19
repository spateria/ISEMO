import numpy as np
import Utils as u
from Agents import registerAgents
import dill
import sys
import os, copy
import time


class OptionControl:
    def __init__(self, args, agent, rng, model_name):
        
        self.noptions = args.noptions
        self.temperature = args.temperature
        self.discount = args.discount
        self.lr_critic = args.lr_critic
        self.lr_term = args.lr_term

        self.estimator = u.Estimator(self.lr_critic, self.lr_term, self.noptions, agent, args.coop, args.testing, model_name)
        
        
        self.option_terminations = [u.SigmoidTermination(rng, self.estimator, o) for o in range(self.noptions)]
        
        # E-greedy policy over options
        #self.policy = u.EgreedyPolicy(rng, self.nstates, self.noptions, args.epsilon)
        self.policy = u.SoftmaxPolicy(rng, self.noptions, self.estimator, agent, self.temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        self.critic = u.ISEMOCritic(self.discount, self.lr_critic,  self.estimator, self.option_terminations)
 
        ############### Termination Functions ########################
        self.termination_improvement = u.TerminationGradient(self.option_terminations, self.critic, self.lr_term, self.estimator)



def runISEMO(args, fileidx, noISER = None):
    

    rng = np.random.RandomState(1234)
    
    agconfig = args.agconfig
    coop = args.coop   

    if args.testing: 
        args.nruns = 1
        args.nepisodes = 1

    history = np.zeros((args.nruns, args.nepisodes, 50))  ###### training record, keep buffer of 50

    start = time.time()
  
  
    for run in range(args.nruns):
        
        if args.testing: 
            run = args.testID  ### only one chosen world
            
            
        if noISER == 'Search':
            model_name = "ISEMO"+ "-noISER-A1-" + str(run) 
        elif noISER == 'Aid':
                model_name = "ISEMO"+ "-noISER-A2-" + str(run)
        elif noISER == 'Relocate':
                model_name = "ISEMO"+ "-noISER-A3-" + str(run)
        elif noISER == 'Helper':
                model_name = "ISEMO"+ "-noISER-A4-" + str(run)
        elif noISER == 'All':
                model_name = "ISEMO"+ "-noISER-A5-" + str(run)
        else:
                model_name = "ISEMO"+ str(run)
        
        
        wload = dill.load(open('MA-World-' + str(run) + '.pl','rb'))
        w = copy.deepcopy(wload['World'])
        w.reset()

        
        agents = registerAgents(w, agconfig, args)
        nagents = len(agents)
        
        if args.testing: option_probas = [[] for _ in range(nagents)]
            
        mu = []   #### high-level controller
        for i in range(nagents):               
            mu.append(OptionControl(args, agents[i], rng, "agent"+str(i)+"-"+model_name))
            agents[i].mu = mu[i]
            agents[i].mu.policy.oset = agents[i].oset

        
        for episode in range(args.nepisodes):
            
            w.reset()  ##### reset the world data after every episode, world configuration (such as number and locations of victims) remains same
            
            phi = []
            last_phi = []
            option = []
            
            ############################  INIT ########################################################
            for i in range(nagents):
          
                phi.append(agents[i].reset()) 
                last_phi.append(phi[i])
                
                option.append(agents[i].mu.policy.sample(phi[i])) #stochastic option sampling
                if args.testing: option_probas[i].append(tuple(agents[i].mu.policy.get_output_probas(phi[i])))
                
                if not args.testing: agents[i].mu.critic.start(phi[i], option[i])  
            ##########################################################################################
            

            

            ################## Measures ######################
            cumreward = np.zeros(nagents)         
            
            option_switches = np.zeros(nagents)
            option_changes = np.zeros(nagents)
            ###################################################
            
            oprev = np.zeros(len(option))*(-1)                
            step_out = [-1 for x in range(len(option))]
            termit = 0


            for step in range(args.nsteps):
                
                if args.testing: print(step)
                print(step)
                
                timeout = bool(step == (args.nsteps-1))
                
                w.scanlist = [] ### clear scan list
                ### reduce health of victims ##############
                w.decayHealth()
                ###########################################
                
                
                '''if run==0 and (episode==0 or episode == args.nepisodes-1):
                    w.show(option, agents, phi, step, coop)'''
                                        
                
                
                ### Execution Block #####################################################################################
                #print("==============================================================================")
                for i in range(nagents):
                    
                    start2 = time.time()
                    
                    if step%10 == 0:  #### clear target after 10 steps for latest location update (equivalent to location selection policy switching)
                        agents[i].clear()
                        1
        
                    if (episode == args.nepisodes-1) or (episode == 0):
                        if 0: 
                            print("AGENT:", i, "obs:", agents[i].env.getstate())
                            print("OPTION:", option[i])
                            print("VALS:", agents[i].mu.policy.value(phi[i]), '\n')
         
                    step_out[i] = agents[i].stepLevel1(option[i])   ##### agent takes an action as per the given option, attributes of the World change                            


                ########################################################################################
                
                ISER = np.zeros(nagents)
                for i in range(nagents):
                    termit = 1
                    next_state, ISER_to = agents[i].env.update(agents[i].Loc, agents[i].id, step_out[i], termit)       #### update the observed features according to the new World state 
                                                                                                                     #### also checks the precondition enabling (or empowerment)           
                    last_phi[i] = copy.deepcopy(phi[i])                                                                        
                    phi[i] = copy.deepcopy(next_state)
                    
                    for j in ISER_to:
                        if ISER[j] == 0:
                            ISER[j] = 1
                   
                                                                               
                Reward = u.reward_blender(w, agents, ISER, noISER)

                #########################################################################################


                start3 = time.time()

                termit = 0
                ###Sampling and update#########################################################    
                for i in range(nagents):
                                                        
                    if option[i] == u.Params.NONE:
                        agents[i].mu.option_terminations[option[i]].force = 1   #### force termination at every step
                
                    if step%1==0:  #### add minimum commitment before switching ----- determinism
                        
                        if agents[i].mu.option_terminations[option[i]].sample(phi[i]):  ###probabilistic termination
                            termit = 1
                            oprev[i] = option[i]
                            
                            option[i] = agents[i].mu.policy.sample(phi[i])   
                            if args.testing: option_probas[i].append(tuple(agents[i].mu.policy.get_output_probas(phi[i])))

                            option_switches[i] += 1
                            
                            if oprev[i] != option[i]:
                                agents[i].clear()
                                option_changes[i] += 1


                    #### continuous updates
                    if not args.testing:
                        if (args.nepisodes>1 and (episode != args.nepisodes-1)) or (args.nepisodes == 1):
                            # Critic update
                            agents[i].mu.critic.update(phi[i], option[i], Reward[i], w.finish(timeout))
                            
                            # Termination update
                            if option[i] == u.Params.NONE:
                                eta = 0.01
                            else:
                                eta = args.eta
                                
                            agents[i].mu.termination_improvement.update(phi[i], option[i], eta)
                ####################################################################################################


                cumreward += Reward
                if len(w.attrList[u.Cell.unknown]) > 0:
                    w.searchsteps += 1


                if w.finish(timeout):               
                    if (not args.testing) and (episode == args.nepisodes-1):  
                        for i in range(nagents):
                            agents[i].mu.estimator.save_models("agent"+str(i)+"-"+model_name)                   
                    break 
                    
  
            
            if args.testing: 
                idx = 0
            else:
                idx = run 
            history[idx, episode, 0:nagents] = cumreward                         
            history[idx, episode, nagents] = w.searchsteps
            history[idx, episode, nagents+1] = w.relocation_points            
            history[idx, episode, nagents+2] = w.num_of_deaths
            history[idx, episode, nagents+3] = step + 1             
            

            print('Run {} episode {} steps {} searchcycles {} relocation_points {} cumreward {} numDeaths {} option_changes {}'.format(run, episode, step, w.searchsteps, 
                                                                        w.relocation_points, cumreward, w.num_of_deaths, option_changes))
            print('Time elapsed: ', (time.time() - start)/60, 'min.\n')
            
            if noISER == 'Search':
                np.save("historyISEMO"+ "-noISER-A1" + "_testing"+str(bool(args.testing))+ "_.npy", history)
            elif noISER == 'Aid':
                np.save("historyISEMO"+ "-noISER-A2" + "_testing"+str(bool(args.testing))+ "_.npy", history)
            elif noISER == 'Relocate':
                np.save("historyISEMO"+ "-noISER-A3" + "_testing"+str(bool(args.testing))+ "_.npy", history)
            elif noISER == 'Helper':
                np.save("historyISEMO"+ "-noISER-A4" + "_testing"+str(bool(args.testing))+ "_.npy", history)
            elif noISER == 'All':
                np.save("historyISEMO"+ "-noISER-ALL" + "_testing"+str(bool(args.testing))+ "_.npy", history)
            else:
                np.save("historyISEMO"+ "_testing"+str(bool(args.testing))+ "_.npy", history)
            

        
            '''if run==0 and (episode==0 or episode == args.nepisodes-1):
                u.makeVid(videoDir , w.figdir, toggle, episode)'''
                

