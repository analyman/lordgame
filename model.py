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
        model.add(Dense(1, activation=activations.relu, input_dim=83))
        # model.add(Dense(20, activation=activations.relu))
        # model.add(Dense(1, activation=activations.relu))
        opt = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=opt, loss="mse")
        return model

    def train_q(self, old_state: [int], action: int,
                old_q: float, reward: float,
                new_state: [int]):
        assert(action >= 0 and action < 100)
        assert(old_state.__len__() == 82)
        assert(new_state.__len__() == 82)
        # TODO
        new_q, _ = self.predict(new_state, 10)
        v = old_q + learning_rate * (reward + gamma * new_q - old_q)
        old_state_data = old_state
        old_state_data.append(action)
        self.model.fit(np.asarray(old_state_data).reshape(1, 83),
                       np.asarray([v]).reshape(1, 1), verbose=0)

    def predict(self, state: [int], avaliable: int) -> (float, int):
        data = state
        max_quality = -1
        max_index = -1
        for i in range(0, avaliable + 1):
            k = data.copy()
            k.append(i)
            assert(k.__len__() == 83)
            q = self.model.predict(np.asarray(k).reshape(1, 83))[0][0]
            if q > max_quality:
                max_quality = q
                max_index = i
        assert(max_index >= 0 and max_index < 100)
        return max_quality, max_index

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
