import gym
# import model
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque, namedtuple
import random



class Env():

	def __init__(self, num_processes=1):
		env = gym.make('Breakout-v0')
		envs = [env for i in range(num_processes)]
		if len(envs) > 1:
			envs = SubprocVecEnv(envs)
		else:
			envs = DummyVecEnv(envs)

	def reset(self):
		init_obs = env.reset()
		return init_obs

	def action():
		return env.action_space.sample()

class Agent():
	env = Env()

	def __init__(self, action):
		self.action = action

	def step(self):
		obs, reward, done, info = env.step(self.action)
		return obs, self.action, reward, done

	def get_action():
		return env.action()

class replay_buffer():
	def __init__(self, state_size, action_size, buffer_size, batch_size):
		self.state_size = state_size
		self.action_size = action_size
		self.batch_size = batch_size
		self.memory = deque(range(buffer_size))
		self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done'])

	def add(self, state, action, reward, done):
		e = self.experience(state, action, reward, done)
		self.memory.append(e)

	def sample(self):
		e = random.sample(self.memory, k=self.batch_size)

		# states = self.add()

		states = self.add([e.state for e in experiences if e is not None])
		actions = self.add([e.action for e in experiences if e is not None])
		rewards = self.add([e.reward for e in experiences if e is not None])
		dones = self.add([e.done for i in experiences if e is not None])
		# return states
		return (states, actions, rewards, next_states, dones)

def main():
	envs = Env()
	agent = Agent()
	storage = ReplayBuffer()

	obs = envs.reset()

	while True:
		action = agent.get_action(obs)
		next_obs, reward, done, info = envs.step(action)
		storage.add(obs, reward, done, next_obs)
		obs = next_obs

if __name__ == '__main__':
	main()