# MARL_ESN_pedestrian

Source codes for "Multi-agent reinforcement learning using echo-state network and its application to pedestrian dynamics".

Requires numpy, scipy, and Pillow.

Note: Codes of deep reinforcement learning (DRL) algorithms stored in 'DRL_algorithms' require Pytorch, instead of scipy.

Most of hyperparameters are given in class 'hyperparameters'.

Whether parameter sharing is adopted or not is decided by Env.experience_sharing.

# updates

Note: The dates in the directory names reflect the date when the erratum was submitted to the journal.

- 2026/02/23
Modified the codes in the following directory to improve efficiency:  
erratum_Sec3_Dec22_2025

- 2025/12/25
Uploaded modified codes for the Erratum correcting inefficient processes. 
Directory: erratum_Sec3_Dec22_2025.

- 2025/12/24
Uploaded modified codes for the Erratum mainly correcting the non-standard algorithm.
Directory: erratum_Sec1and2_Dec22_2025

- 2024/03/22
modified inefficient calculations.

- 2024/03/22
uplaoded DRL codes used for comparison of performance.

# Notes on Reproducibility

These hyperparameters were omitted in the paper:

- update frequency of DQN per environmental step: 1

- target network update frequency of DQN: hard update at the end of each episode (=500 environmental steps)


# Credits
[Added on 2025-07-31 to increase transparency.]
The deep learning codes used for comparison were influenced by multiple open-source implementations, including:

- https://github.com/ikostrikov/pytorch-a3c (MIT license)
- https://github.com/nikhilbarhate99/PPO-PyTorch (MIT license)

In particular, the function `memory_sample` in pedestrian_task1_DQN.py and pedestrian_task2_DQN.py is based on the following implementation:

- https://qiita.com/Rowing0914/items/eeba790401bcaf2c723c  

We sincerely appreciate the contributions of the original authors and acknowledge their impact on this work.
