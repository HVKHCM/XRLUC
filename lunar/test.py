from lunarSB import *
from stable_baselines3 import DQN

env = ll()

agent = customDQN.load('trainedModel/ll-test/taxiSB.zip')

obs, info = env.reset()
action, _state = agent.predict(obs, deterministic=True)
print(action)
env.set_state(np.array([1,1,1,1,1,1,0,0]))
obs = env.get_state()
action, _state = agent.predict(obs, deterministic=True)
print(action)
print(obs)
print(type(obs))

