import gym
import model
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class env(VecEnvWrapper):
	def __init__(self, num_processes, action):
		envs = [gym.make('Breakout-v0') for i in range(num_processes)]
		obs_space = env.observation_space
		if len(envs) > 1:
			envs = SubprocVecEnv(envs)
		else:
			envs = DummyVecEnv(envs)

	def step(self):
		action = get_action()
		self.envs.step_async(action)
		obs, reward, done, info = self.envs.step_wait()
		return obs, reward, done

	def get_action(self):
		if action == True:
			action = action
		else:
			action = env.action_space()
		return action

	def reset(self):
		init_obs = self.envs.reset()
		return init_obs

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        y = self.pool(F.relu(self.conv2(y)))
        y = torch.flatten(y, 1) # flatten all dimensions except batch
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return Normal(y)

class replay_buffer():
	def __init__(self, state_size, action_size, buffer_size, batch_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(max_len=buffer_size)
		self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

	def add(self, state, action, reward, next_state, done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)

		states = add([e.state for e in experiences if e is not None])
		actions = add([e.action for e in experiences if e is not None])
		rewards = add([e.reward for e in experiences if e is not None])
		next_states = add([e.next_state for e in experiences if e is not None])
		dones = add([e.done for i in experiences if e is not None])

		return (states, actions, rewards, next_states, dones)

class ppo

# class replay_buffer():
# 	def __init__(self, obs, action, reward, next_obs):
# 		self.obs = []
# 		self.action = []
# 		self.reward = []
# 		self.next_obs = []

# 	def get_attr(self):
# 		obs = self.obs.append(env.step()[0])
# 		action = self.action.append(env.get_action(obs))
# 		reward = self.reward.append(env.step()[1])
# 		if len(obs) > 1:
# 			next_obs = 
# 		return obs, action, reward, next_obs



# def __init__(self, venv: VecEnv, key:str, action, num_steps):
	# 	self.key = key
	# 	if action == True:
	# 		self.action = action
	# 	else:
	# 		self.action = env.action_space.sample()
	# 	self.num_steps = num_steps

	# def env(self.action)
	# 	env = gym.make('Breakout-v0')
	# 	env.reset()
	# 	for t in range(self.num_steps):
	# 		obs = self.venv.reset()
	# return obs[self.key]

