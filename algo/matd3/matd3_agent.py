from torch.distributions import Normal

from algo.matd3.network import Critic, Actor
import torch
from copy import deepcopy
from torch.optim import Adam
from algo.memory import ReplayMemory, Experience
from algo.random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import numpy as np
from algo.utils import device
scale_reward = 0.01


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MATD3:
    def __init__(self, dim_obs, dim_act, n_agents, policy_target_update_interval, args):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics1 = []
        self.critics2 = []
        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critic = Critic(n_agents, dim_obs, dim_act)
        self.critics1 = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
        self.critics2 = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.policy_target_update_interval = policy_target_update_interval

        self.actors_target = deepcopy(self.actors)
        self.critics1_target = deepcopy(self.critics1)
        self.critics2_target = deepcopy(self.critics2)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = args.episode_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]

        self.critic1_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics1]
        self.critic2_optimizer = [Adam(x.parameters(),
                                       lr=0.001) for x in self.critics2]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        self.c1_loss = []
        self.c2_loss = []
        self.a_loss = []

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics1:
                x.cuda()
            for x in self.critics2:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics1_target:
                x.cuda()
            for x in self.critics2_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0
        self.epsilon = 1e-6
        self.eval_noise_scale = 0.0

    def load_model(self):
        if self.args.model_episode:
            path_flag = True
            for idx in range(self.n_agents):
                path_flag = path_flag \
                            and (os.path.exists("trained_model/matd3/actor["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth")) \
                            and (os.path.exists("trained_model/matd3/critic1["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth")) \
                            and (os.path.exists("trained_model/matd3/critic2[" + str(idx) + "]_"
                                                +str(self.args.model_episode) + ".pth"))


            if path_flag:
                print("load model!")
                for idx in range(self.n_agents):
                    if self.use_cuda:
                        actor = torch.load(
                            "trained_model/matd3/actor[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth")
                        critic1 = torch.load(
                            "trained_model/matd3/critic1[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth")
                        critic2 = torch.load(
                            "trained_model/matd3/critic2[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth")
                    else:
                        actor = torch.load(
                            "trained_model/matd3/actor[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth",
                            map_location=torch.device('cpu'))
                        critic1 = torch.load(
                            "trained_model/matd3/critic1[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth",
                            map_location=torch.device('cpu'))
                        critic2 = torch.load(
                            "trained_model/matd3/critic2[" + str(idx) + "]_" + str(self.args.model_episode) + ".pth",
                            map_location=torch.device('cpu'))
                    self.actors[idx].load_state_dict(actor.state_dict())
                    self.critics1[idx].load_state_dict(critic1.state_dict())
                    self.critics2[idx].load_state_dict(critic2.state_dict())

        self.actors_target = deepcopy(self.actors)
        self.critics1_target = deepcopy(self.critics1)
        self.critics2_target = deepcopy(self.critics2)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
            os.makedirs("./trained_model/" + str(self.args.algo) + "/")
        for i in range(self.n_agents):
            torch.save(self.actors[i],
                       'trained_model/matd3/actor[' + str(i) + ']' + '_' + str(episode) + '.pth')
            torch.save(self.critics1[i],
                       'trained_model/matd3/critic1[' + str(i) + ']' + '_' + str(episode) + '.pth')
            torch.save(self.critics2[i],
                       'trained_model/matd3/critic2[' + str(i) + ']' + '_' + str(episode) + '.pth')

    def update(self,i_episode):

        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        if self.train_num % self.policy_target_update_interval == 0:
            self.c1_loss = []
            self.c2_loss = []
            self.a_loss = []

        for agent in range(self.n_agents):

            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            non_final_next_actions = []
            self.actor_optimizer[agent].zero_grad()
            self.critic1_optimizer[agent].zero_grad()
            self.critic2_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics1[agent].zero_grad()
            self.critics2[agent].zero_grad()
            current_Q1 = self.critics1[agent](whole_state, whole_action)
            current_Q2 = self.critics2[agent](whole_state, whole_action)
            # non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :])[0] for i in range(self.n_agents)]
            for i in range(self.n_agents):
                mean, log_std = self.actors_target[i](non_final_next_states[:, i, :])
                std = log_std.exp()  # no clip in evaluation, clip affects gradients flow
                normal = Normal(0, 1)
                z = normal.sample()
                action = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
                # log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + self.epsilon)
                # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
                # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
                # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
                # log_prob = log_prob.sum(dim=1, keepdim=True)
                ''' add noise '''
                eval_noise_clip = 2 * self.eval_noise_scale
                noise = normal.sample(action.shape) * self.eval_noise_scale
                noise = torch.clamp(noise, -eval_noise_clip, eval_noise_clip)
                action = action + noise.to(device)
                non_final_next_actions.append(action)
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())
            target_Q1 = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q2 = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q1[non_final_mask] = self.critics1_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze() # .view(-1, self.n_agents * self.n_actions)
            target_Q2[non_final_mask] = self.critics2_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze()  # .view(-1, self.n_agents * self.n_actions)
            # scale_reward: to scale reward in Q functions
            # reward_sum = sum([reward_batch[:,agent_idx] for agent_idx in range(self.n_agents)])
            target_Q = torch.min(target_Q1,target_Q2)
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1)*0.1)# + reward_sum.unsqueeze(1) * 0.1

            loss_Q1 = nn.MSELoss()(current_Q1, target_Q.detach())
            loss_Q2 = nn.MSELoss()(current_Q2, target_Q.detach())
            loss_Q1.backward()
            loss_Q2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics1[agent].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics2[agent].parameters(), 1)
            self.critic1_optimizer[agent].step()
            self.critic2_optimizer[agent].step()

            if self.train_num % self.policy_target_update_interval == 0:
                self.actor_optimizer[agent].zero_grad()
                self.critic1_optimizer[agent].zero_grad()
                self.actors[agent].zero_grad()
                self.critics1[agent].zero_grad()
                state_i = state_batch[:, agent, :]
                action_i, _ = self.actors[agent](state_i)
                ac = action_batch.clone()
                ac[:, agent, :] = action_i
                whole_action = ac.view(self.batch_size, -1)
                actor_loss = -self.critics1[agent](whole_state, whole_action).mean()
                # actor_loss += (action_i ** 2).mean() * 1e-3
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critics1[agent].parameters(), 1)
                self.actor_optimizer[agent].step()
                # self.critic_optimizer[agent].step()
                self.c1_loss.append(loss_Q1)
                self.c2_loss.append(loss_Q2)
                self.a_loss.append(actor_loss)

        if self.train_num % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics1_target[i], self.critics1[i], self.tau)
                soft_update(self.critics2_target[i], self.critics2[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return sum(self.c1_loss)/self.n_agents, sum(self.a_loss)/self.n_agents

    def choose_action(self, state, noisy=True):
        obs = torch.from_numpy(np.stack(state)).float().to(device)
        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = obs[i].detach()
            act, log_std = self.actors[i](sb.unsqueeze(0))
            act = act.squeeze()
            log_std = log_std.squeeze()
            if noisy:
                std = log_std.exp()
                normal = Normal(0, 1)
                z = normal.sample().to(device)
                act = (act + std*z).detach().type(FloatTensor)
                if self.episode_done > self.episodes_before_train and \
                        self.var[i] > 0.05:
                    self.var[i] *= 0.999998
            act = torch.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1
        return actions.data.cpu().numpy()
