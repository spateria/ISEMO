# ISEMO
Multi-agent Reinforcement Learning in Spatial Domain Tasks using Inter Subtask Empowerment Rewards.

# Dependencies
Python >= 3.5.0

scikit-learn == 0.19.1

scipy == 1.0.0

opencv-python == 4.1.1.26

# Setup before training
Before training, it is required to make the _World_ objects. To make the _World_ objects, give the following command: 

`python main.py --make` 

The result will be saved in files named as MA-World-{i}.pl, where _i_ ranges from _0_ to _nruns-1_. Here, _nruns_ is defined in the _args_ class in main.py.


Please refer to ISEMO_SW_Doc.pdf for details and instructions.
