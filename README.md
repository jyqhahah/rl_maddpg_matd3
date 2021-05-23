# MAProj

This is a multi-agent project in pytorch for the environment of Multi-Agent Particle Environment "simple_spread"(https://github.com/openai/multiagent-particle-envs). The model "MATD3" combines MADDPG(https://arxiv.org/abs/1706.02275) and TWIN DELAYED DDPG (https://www.researchgate.net/profile/Wenfeng-Zheng/publication/341648433_Twin-Delayed_DDPG_A_Deep_Reinforcement_Learning_Technique_to_Model_a_Continuous_Movement_of_an_Intelligent_Robot_Agent/links/5ed9ae3e92851c9c5e816d19/Twin-Delayed-DDPG-A-Deep-Reinforcement-Learning-Technique-to-Model-a-Continuous-Movement-of-an-Intelligent-Robot-Agent.pdf)

INFERENCE: 
- https://github.com/isp1tze/MAProj
- https://github.com/openai/multiagent-particle-envs

## MADDPG After 10000 Episode Training:
![maddpg](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/maddpg_10000.gif)

## MADDPG After 10000 Episode Training:
![matd3](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/matd3_10000.gif)

## Training curves:
![curves](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/curve.png)

## How to use
- pip install -r requirements.txt
- cd MAProj/algo
  (if you want to train)
- python ma_main.py --algo matd3 --mode train
  (if you want to eval)
- python ma_main.py --algo matd3 --mode eval --model_episode 20000

## To do list
- train MATD3 with a variety of different hyper-parameters