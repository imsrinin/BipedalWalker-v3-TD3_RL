# BipedalWalker-v3-TD3_RL
Teaching an bipedal bot how to walk using a TD3 algorithm (variant of Reinforcement Learning - Actor &amp; Critic method)  


The experiment focusses on training the BipedalWalker using reinforcement learning algorithm (TD3) using 3 FC layers for both actor and critic and then explore how different variations in the state and action space effect the walking styles and learning patterns of the model.  
The experiments have been conducted with the following configurations of the environment using custom gym wrapper functions:  
1. Reduced state space (17/24 states available for training)  
2. Reduced action space (3/4 actions available for the model)  
3. Limited action space (action range limited to half of its potential [-0.5,0.5])  
4. Limited action and reduced state space (19/24 states and config-3)  
Paper for the TD3 algorithm: https://arxiv.org/pdf/1802.09477.pdf  
/https://share.vidyard.com/watch/tDm31KXAXSTrkoDunmM2od?
