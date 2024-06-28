import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from copy import deepcopy

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import torch

import numpy as np
class ll(gym.Env):
    def __init__(self):
        super(ll, self).__init__()
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def seed(self, seed=None):
        # Implement the seed method to set seeds for the environment's RNG
        return self.env.seed(seed)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def set_state(self, state):
        self.env.unwrapped.s = state

    def get_state(self):
        return deepcopy(self.env.s)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

class customDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(customDQN, self).__init__(*args, **kwargs)

    def predict_q(self, observation):
        """
        Predict Q-values for the given observation using the current policy network.
        The observation is first one-hot encoded.
        """
        # Preprocess the observation
        #observation_encoded = self.one_hot_encode(observation)
        obs_tensor = torch.tensor([observation]).to(self.device)
        
        # Obtain Q-values from the network
        with torch.no_grad():
            q_values = self.policy.q_net(obs_tensor)
        
        return q_values.cpu().numpy()

""" # Create the environment
env = ll()

# Stable Baselines3 expects vectorized environments
env = make_vec_env(lambda: env, n_envs=1)

# Instantiate the DQN model
model = customDQN("MlpPolicy", env, verbose=1, buffer_size=50000, learning_rate=1e-3, batch_size=32, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1, max_grad_norm=10)
# Train the model
model.learn(total_timesteps=int(1e7))

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

#Save model
model.save('trainedModel/ll-test/taxiSB')

obs = env.reset()
print(obs)
print(type(obs))

action, _state = model.predict(obs, deterministic=True)
print(model.predict_q(obs))

env.close()  """
