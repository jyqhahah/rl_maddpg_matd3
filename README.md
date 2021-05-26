# MATD3

This is a multi-agent project in pytorch for the environment of Multi-Agent Particle Environment "simple_spread"(https://github.com/openai/multiagent-particle-envs). The model "MATD3" combines MADDPG(https://arxiv.org/abs/1706.02275) and TWIN DELAYED DDPG (https://arxiv.org/abs/1802.09477)



## MADDPG After 15000 Episode Training:
![maddpg](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/maddpg_15000.gif)

## MATD3 After 15000 Episode Training:
![matd3](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/matd3_15000.gif)

## Training curves:
![curves](https://github.com/jyqhahah/rl_maddpg_matd3/blob/main/asset/curve.png)

## How to use
- pip install -r requirements.txt
- cd rl_maddpg_matd3/algo
### (if you want to train)
- python ma_main.py --algo matd3 --mode train
### (if you want to eval)
- python ma_main.py --algo matd3 --mode eval --model_episode 20000

## To do list
- train MATD3 with a variety of different hyper-parameters