# MARL_ESN_pedestrian

Source codes for "Multi-agent reinforcement learning using echo-state network and its application to pedestrian dynamics".

Requires numpy, scipy, and Pillow.

Note: Codes of deep reinforcement learning (DRL) algorithms stored in 'DRL_algorithms' require Pytorch, instead of scipy.

Most of hyperparameters are given in class 'hyperparameters'.

Whether parameter sharing is adopted or not is decided by Env.experience_sharing.

# updates

2024/03/22 modified inefficient calculations.

2024/03/22 uplaoded DRL codes used for comparison of performance.


# Credits
[Added on 2025-08-10 to increase transparency.]
The deep learning codes used for comparison were influenced by multiple open-source implementations, including:

- https://github.com/ikostrikov/pytorch-a3c  
- https://github.com/nikhilbarhate99/PPO-PyTorch  

In particular, the function `memory_sample` in pedestrian_task1_DQN.py and pedestrian_task2_DQN.py is based on the following implementation:

- https://qiita.com/Rowing0914/items/eeba790401bcaf2c723c  

We sincerely appreciate the contributions of the original authors and acknowledge their impact on this work.
