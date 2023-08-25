# ISEMO: Multi-agent Reinforcement Learning in Spatial Domain Tasks using Inter Subtask Empowerment Rewards.
[Paper source](https://ieeexplore.ieee.org/abstract/document/9002777) 
In the complex multi-agent tasks, various agents must cooperate to distribute relevant subtasks among each other to achieve joint task objectives. An agent's choice of the relevant subtask changes over time with the changes in the task environment state. Multi-agent Hierarchical Reinforcement Learning (MAHRL) provides an approach for learning to select the subtasks in response to the environment states, by using the joint task rewards to train various agents. When the joint task involves complex inter-agent dependencies, only a subset of agents might be capable of reaching the rewarding task states while other agents take precursory or intermediate roles. The delayed task reward might not be sufficient in such tasks to learn the coordinating policies for various agents. In this paper, we introduce a novel approach of MAHRL called Inter-Subtask Empowerment based Multi-agent Options (ISEMO) in which an Inter-Subtask Empowerment Reward (ISER) is given to an agent which enables the precondition(s) of other agents' subtasks. ISER is given in addition to the domain task reward in order to improve the inter-agent coordination. ISEMO also incorporates options model that can learn parameterized subtask termination functions and relax the limitations posed by hand-crafted termination conditions. Experiments in a spatial Search and Rescue domain show that ISEMO can learn the subtask selection policies of various agents grounded in the inter-dependencies among the agents, as well as learn the subtask termination conditions, and perform better than the standard MAHRL technique.


# Baseline method is Cooperative HRL (CoHRL) 
Ghavamzadeh, Mohammad, Sridhar Mahadevan, and Rajbala Makar. "Hierarchical multi-agent reinforcement learning." Autonomous Agents and Multi-Agent Systems 13.2 (2006): 197-229.

This code includes Python implementation of CoHRL.

# Dependencies
Python >= 3.5.0

scikit-learn == 0.19.1

scipy == 1.0.0

opencv-python == 4.1.1.26

# Setup before training
Before training, it is required to make the _World_ objects. A _World_ object contains attributes and configuration of the simulated Search & Rescue environment in which the multi-agent team is trained.  
To make the _World_ objects, give the following command: 

`python main.py --make` 

The result will be saved in files named as MA-World-{i}.pl, where {i} ranges from _0_ to _nruns-1_. Here, _nruns_ is defined in the _args_ class in main.py. In each run, the configuration of the _World_ changes (such as locations and/or numbers of certain objects)

# Training mode
To run the software in the training mode, give the following command:

`python main.py`  

By default, this runs ISEMO. To run CoHRL instead, give the following command:

`python main.py −−runCoHRL`

During training, data is saved in files with the names as: _historyISEMO\_testingFalse\_.npy_ when using ISEMO and _historyCoHRL\_testingFalse\_.npy_. 
You can check the list of recorded data items in ISEMO.py (refer to the multi-dimensional array named _history_). 

The learned models for the Q-value functions and the termination functions are saved in the _models_ folder.

# Testing mode
To  run  the  software  in  thetesting mode,  give  the  following  command:

`python main.py −−testing −−testID {i}`  

Here, testID {i} is the index of the saved _World_ object (MA-World-{i}.pl) to be used for testing.

# Further...
Please refer to ISEMO_SW_Doc.pdf for more details.
