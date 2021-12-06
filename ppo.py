import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import namedtuple
import random
import numpy as np
import torch
from torch import nn
import pandas as pd

# UTILS...............................................................................................................

def lst_to_df(lst1, lst1_name, lst2, lst2_name, var1, var2):
	df = pd.DataFrame({lst1_name: lst1, lst2_name: lst2})
	df[lst1_name] = np.where(df[lst1_name] == var1, 1, 0)
	df[lst2_name] = np.where(df[lst2_name] == var2, 1, 0)
	df['count'] = 1
	return pd.pivot_table(df, values='count', index=[lst1_name], columns=[lst2_name], aggfunc=np.size, fill_value=0)

def con_prob(state_lst, reward_lst, cur_state, cur_reward):
	table = lst_to_df(state_lst, 'state', reward_lst, 'reward', cur_state, cur_reward)
	table_sum = table[0][0]+table[0][1]+table[1][0]+table[1][1]
	p_a = (table[1][0]+table[1][1])/table_sum
	p_b = (table[0][1]+table[1][1])/table_sum
	p_a_b = table[1][1]/table_sum
	return p_a_b/p_b

# ....................................................................................................................

class Worker:
    def __init__(self, env_name="PongNoFrameskip-v4", num_processes=8):
        self.name = env_name
        envs = [self.create_env(env_name, i) for i in range(num_processes)]
        self.envs = SubprocVecEnv(envs)

    def create_env(self, name, seed=0):
        def create_fn_():
            env = gym.make(name)
            env.seed(seed)
            return env
        return create_fn_

    def reset(self):
        return self.envs.reset()

    def step(self, action):
        return self.envs.step(action)

    def episode_length(self, env_name):
    	spec = gym.envs.registration.spec(env_name)
    	return spec.max_episode_steps

    @property
    def action_space(self):
        dummy_env = gym.make(self.name)
        action_space = dummy_env.action_space
        dummy_env.close()
        return action_space

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return [self.action_space.sample() for _ in obs]

class ReplayBuffer():
	def __init__(self, buffer_size=2, batch_size=2):
		self.batch_size = batch_size
		self.memory = []
		self.experience = namedtuple('Experience', field_names=['obs', 'reward', 'done', 'next_obs'])

	def add(self, obs, reward, done, next_obs):
		e = [obs, reward, done, next_obs]
		e = self.experience(obs, reward, done, next_obs)
		self.memory.append(e)

	def sample(self):

		sample = self.memory.pop()
		obs, reward, done, next_obs = sample

		return obs, reward, done, next_obs

	def get_sample(self):

		ret = []
		for i in range(self.batch_size):
			ret.append(self.sample())

		return ret

class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.conv1 = nn.Conv2d(D_in, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, D_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        y = self.pool(F.relu(self.conv2(y)))
        y = torch.flatten(y, 1) # flatten all dimensions except batch
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        # return Normal(y)
        return y

class ppo():
	def __init__(self, batch, action, gamma, steps, delta_steps, model, weight_path, sigma, c):
		self.batch = batch
		self.action = action
		self.gamma = gamma
		self.steps = steps
		self.model = model
		self.weight_path = weight_path
		self.sigma = sigma
		self.c = c

	# def probs(self.batch):
	
	# 	obs = []
	# 	acts = []
	# 	log_probs = []
	# 	rewards = []

	# 	for i in range(self.batch.size()):
	# 		obs.append(self.batch[i][0])
	# 		acts.append(self.action[i])
	# 		log_probs.append(self.)
	# 		rewards.append(self.)

	def get_state(self, batch):
		state = []
		batch = self.batch
		for i in range(self.batch_size()):
			state.append(self.batch[i][0])
		return state

	def policy(self, batch, steps, model, path):
		outs = []
		r = []
		batch = self.batch
		steps = self.steps
		model = self.model
		path = self.weight_path
		state = get_state(batch)

		for i in range(len(state)):
			out = model(state[i])
			torch.save(model.state_dict, path)
			outs.append(out)
		for i in range(steps):
			if i == 0:
				r.append(torch.max(outs[i])/1)
			else:
				r.append(torch.max(outs[i])/torch.max(outs[i-1]))

		return r

	def get_actions(self, delta_steps, path, state):
		model = torch.load(path)
		actions = []

		for i in range(steps)-delta_steps:
			out = model(state)
			action = torch.max(out)
			actions.append(action)
		return actions

	def get_reward(self, delta_steps, state, path):
		agent = Worker()
		action_space = get_actions(delta_steps, path, state)
		return agent.step(action_space)


	def state_value(self, gamma, delta_steps, state, path):
		gamma = self.gamma
		rewards = self.get_reward()
		states = self.get_state()

		# states = batch[0]
		# rewards = batch[1]

		for i in range(len(reward)):
			gamma**i*con_prob(states, rewards, states[i], rewards[i])

	def adv_est(self):
		state_value = self.state_value(gamma, delta_steps, state, path)
		model = self.model
		action = self.get_action
		state = self.get_state

		a_e = []
		for i in range(len(reward)):
			a_e.append(model(action[i], state[i]) - state_value(gamma, delta_steps, state[i], path))

				# gamma**(len(reward) - t) * reward[t] * state_value(state[t])
		return a_e

	def l_clip(self):
		sigma = self.sigma
		policy = self.policy()
		adv_est = self.adv_est()
		pa = [i * adv_est for i in policy]

		return min(pa,clip(policy, 1-sigma, 1+sigma)*adv_est)

	def vf(self, max_reward):
		gamma = self.gamma
		states = self.get_state
		c = self.c
		state_value = self.state_value()
		v_targ = self.state_value(gamma, max_reward, states)
		return c*(state_value - v_targ)**2

	def ppo_loss(self):
		l_clip = self.l_clip()
		vf = self.vf()
		c = self.c
		return l_clip - c*vf

# def main():
#     envs = Worker()
#     agent = Agent(envs.action_space)
#     storage = ReplayBuffer()

#     obs = envs.reset()

#     while True:

#     	action = agent.get_action(obs)
#     	# print(action)
#     	next_obs, reward, done, info = envs.step(action)
#     	storage.add(obs, reward, done, next_obs)
#     	obs = next_obs

#     # with open('obs.txt', 'w') as f:
#     # 	for i in range(2):

#     # 		action = agent.get_action(obs)
#     # 		# print(action)
#     # 		next_obs, reward, done, info = envs.step(action)
#     # 		f.write('next_obs\n')
#     # 		f.write(str(next_obs)+'\n')
#     # 		f.write('reward\n')
#     # 		f.write(str(reward)+'\n')
#     # 		f.write('done')
#     # 		f.write(str(done)+'\n')
#     # 		storage.add(obs, reward, done, next_obs)
#     # 		obs = next_obs

#     # 	f.write('buffer\n')
#     # 	f.write(str(storage.get_sample()))

# if __name__ == "__main__":
#     main()