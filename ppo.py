import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from collections import deque, namedtuple
import random
import numpy as np

class Worker:
    def __init__(self, env_name="PongNoFrameskip-v4", num_processes=8):
        self.name = env_name
        envs = [self.create_env(env_name, i) for i in range(num_processes)]
        self.envs = SubprocVecEnv(envs)

    def create_env(selv, name, seed=0):
        def create_fn_():
            env = gym.make(name)
            env.seed(seed)
            return env
        return create_fn_

    def reset(self):
        return self.envs.reset()

    def step(self, action):
        return self.envs.step(action)

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

class model()

class ppo():
	def __init__(self, batch, action, model):
		self.batch = batch
		self.action = action

	def probs(self.batch):
	
		obs = []
		acts = []
		log_probs = []
		rewards = []

		for i in range(self.batch.size()):
			obs.append(self.batch[i][0])
			acts.append(self.action[i])
			log_probs.append(self.)
			rewards.append(self.)

	def policy():

	def learn(self, policy, model):
		clip_loss = 


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