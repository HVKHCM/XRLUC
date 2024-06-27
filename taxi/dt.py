# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from log import *
import graphviz
import rl2

def accuracy(policy, obss, acts):
    pred = policy.predict_before(obs for obs in obss)

    sum = 0
    for i in range(len(acts)):
        if acts[i] == pred[i]:
            sum += 1
    return sum/len(obss)

def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = []
    acts_train = []
    obss_test = []
    acts_test = []
    for i in idx[:n_train]:
        obss_train.append(obss[i])
        acts_train.append(acts[i])

    for i in idx[n_train:]:
        obss_test.append(obss[i])
        acts_test.append(acts[i])

    #obss_train = obss[idx[:n_train]]
    #acts_train = acts[idx[:n_train]]
    #obss_test = obss[idx[n_train:]]
    #acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test

def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()

def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    classes = ["0", "1", "2", "3", "4", "5"]
    export_graphviz(dt_policy.tree, dirname + '/' + fname,
                                class_names=classes,
                                filled=True, rounded=True,
                                special_characters=True)

def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy

def load_grahviz(dirname, fname):
    print(dirname + '/' + fname)
    return graphviz.Source.from_file(dirname + '/' + fname)

class DTPolicy:
    def __init__(self, max_depth, state_transformer):
        self.max_depth = max_depth
        self.state_transformer = state_transformer
    
    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)), INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)), INFO)
        log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)

    def predict(self, obs):
        return self.tree.predict([obs])
    
    def predict_before(self, obs):
        return self.tree.predict(self.state_transformer(obs))

    def clone(self):
        clone = DTPolicy(self.max_depth, self.state_transformer)
        clone.tree = self.tree
        return clone

    def decision_path(self, obs):
        return self.tree.decision_path(obs)

    def branches_retrieve(self):
        n_nodes = self.tree.tree_.node_count
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        feature = self.tree.tree_.feature
        threshold = self.tree.tree_.threshold
        impurity = self.tree.tree_.impurity
        value = self.tree.tree_.value
    
        # Calculate if a node is a leaf
        is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left, children_right)]
    
        # Store the branches paths
        paths = []
    
        for i in range(n_nodes):
            if is_leaves_list[i]:
            # Search leaf node in previous paths
                end_node = [path[-1] for path in paths]

            # If it is a leave node yield the path
                if i in end_node:
                    output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                    yield output

            else:
            
            # Origin and end nodes
                origin, end_l, end_r = i, children_left[i], children_right[i]

            # Iterate over previous paths to add nodes
                for index, path in enumerate(paths):
                    if origin == path[-1]:
                        paths[index] = path + [end_l]
                        paths.append(path + [end_r])

            # Initialize path in first iteration
                if i == 0:
                    paths.append([i, children_left[i]])
                    paths.append([i, children_right[i]])
        return paths
