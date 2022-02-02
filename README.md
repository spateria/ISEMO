# ISEMO
Multi-agent Reinforcement Learning in Spatial Domain Tasks using Inter Subtask Empowerment Rewards.

# Baseline CoHRL

# Dependencies
Python >= 3.5.0

scikit-learn == 0.19.1

scipy == 1.0.0

opencv-python == 4.1.1.26

# Setup before training
Before training, it is required to make the _World_ objects. To make the _World_ objects, give the following command: 

`python main.py --make` 

The result will be saved in files named as MA-World-{i}.pl, where {i} ranges from _0_ to _nruns-1_. Here, _nruns_ is defined in the _args_ class in main.py.

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

Please refer to ISEMO_SW_Doc.pdf for more details.
