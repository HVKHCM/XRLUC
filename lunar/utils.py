import numpy as np
from log import *
import pydotplus
from IPython.display import Image
from sklearn import tree
from gymnasium.envs.toy_text.taxi import *
from lunarSB import *
import gymnasium
from sklearn.tree import DecisionTreeClassifier

def state_transformer(obs):
    return list(obs)

def decoder(obs):
    row, col, passe, des = TaxiEnv().decode(obs)
    feature = []
    feature.append(row)
    feature.append(col)
    feature.append(passe)
    feature.append(des)
    return feature

def q_transformer(obs):
    obs = np.reshape(obs, [1, 1])
    return obs