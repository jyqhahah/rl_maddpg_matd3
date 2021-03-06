import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Critic(nn.Module):
	def __init__(self,n_agent,dim_observation,dim_action):
		super(Critic,self).__init__()
		self.n_agent = n_agent
		self.dim_observation = dim_observation
		self.dim_action = dim_action
		obs_dim = self.dim_observation * n_agent
		act_dim = self.dim_action * n_agent
		
		self.FC1 = nn.Linear(obs_dim,1024)
		self.FC2 = nn.Linear(1024+act_dim,512)
		self.FC3 = nn.Linear(512,300)
		self.FC4 = nn.Linear(300,1)
		
	# obs:batch_size * obs_dim
	def forward(self, obs, acts):
		result = F.relu(self.FC1(obs))
		combined = th.cat([result, acts], dim=1)
		result = F.relu(self.FC2(combined))
		return self.FC4(F.relu(self.FC3(result)))
		
class Actor(nn.Module):
	def __init__(self,dim_observation,dim_action,log_std_min=-20,log_std_max=2):
		#print('model.dim_action',dim_action) 
		super(Actor,self).__init__()
		self.log_std_min=log_std_min
		self.log_std_max=log_std_max
		self.FC1 = nn.Linear(dim_observation,500)
		self.FC2 = nn.Linear(500,128)
		self.FC3 = nn.Linear(128, 128)
		self.mean_linear = nn.Linear(128,dim_action)
		self.log_std_linear = nn.Linear(128, dim_action)


	def forward(self,obs):
		result = F.relu(self.FC1(obs))
		result = F.relu(self.FC2(result))
		result = F.tanh(self.FC3(result))
		mean = F.tanh(self.mean_linear(result))
		log_std = self.log_std_linear(result)
		log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std

	
