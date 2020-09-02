#!/usr/bin/python3

import numpy as np
import keras
import math
from random import randint
from keras import activations
from keras.layers.core import Dense

# Q-Table: (State x Action) -> quality
# State
#    1. who_lord                       1
#    2. who_prev_take [-1, 0, 1]       1
#    3. cards                          20
#    4. prev_records                   20
#    5. my_records                     20
#    6. next_records                   20
# Action
#    1. [0..100]                       1
#                                total:
#                                      83
#
# Update Q-Table
# Q(s, a) = Q(s, a) + alpha * (reward + gamma*max_on_a(Q(next_s, a)) - Q(s, a))
# ANN fit
# model.fit([s, a], Q(s, a))

learning_rate = 0.001
gamma = 0.9


# TODO remember training
class LordAgent():
    def __init__(self):
        self.model = LordAgent.network()

    @classmethod
    def network(cls):
        model = keras.Sequential()
        model.add(Dense(100, activation=activations.relu, input_dim=83))
        model.add(Dense(100, activation=activations.relu))
        model.add(Dense(1, activation=activations.relu))
        opt = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=opt, loss="mse")
        return model

    def train_q(self, old_state: [int], action: int,
                old_q: float, reward: float,
                new_state: [int]):
        assert(action >= 0)
        assert(old_state.__len__() == 82)
        assert(new_state.__len__() == 82)
        # TODO
        new_q, _ = self.predict(new_state, 10)
        v = old_q + learning_rate * (reward + gamma * new_q - old_q)
        old_state_data = old_state
        old_state_data.append(action)
        self.model.fit(np.asarray(old_state_data).reshape(1, 83),
                       np.asarray([v]).reshape(1, 1), verbose=1)

    def quality(self, state: [int], action: int) -> float:
        data = state.copy()
        data.append(action)
        return self.model.predict(np.asarray(data).reshape(1, 83))[0][0]

    def train_x(self, state_action: [([int], int)], new_quality: [float]):
        ll = state_action.__len__()
        assert(new_quality.__len__() == ll)
        data = []
        for s, a in state_action:
            s.append(a)
            data.append(s)
        self.model.fit(np.asarray(data).reshape(ll, 83),
                       np.asarray(new_quality).reshape(ll, 1), verbose=1)

    def predict(self, state: [int], avaliable: int) -> (float, int):
        assert(state.__len__() == 82)
        max_quality = -1
        max_index = -1
        for i in range(0, avaliable + 1):
            q = self.quality(state, i)
            if q > max_quality:
                max_quality = q
                max_index = i
        return max_quality, max_index

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)

    def try_load(self, filepath):
        m = self.model
        try:
            self.load(filepath)
        except OSError as err:
            print(err)
            self.model = m
