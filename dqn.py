import keras
import tensorflow as tf
import numpy as np
import retro
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, Conv1D
from PIL import Image
import cv2
import math
from keras.models import load_model

model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(56, 80, 1)))
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(Conv2D(16, 8, 8, activation='relu'))
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# model = load_model('my_model-1.h5')


# zdaje się, że tylko te przyciski miały jakiś sens
# 6 - left
# 7 - right
# 0 - jump
num2action = []
for i in range(0, 12):
    # num2action.append(np.identity(12, dtype=int)[i:i+1])
    num2action.append(np.zeros(12))

num2action[0][0]=1
num2action[1][6]=1
num2action[2][7]=1

def convert(s__):
    s__ = cv2.resize(s__, dsize=(80, 56), interpolation=cv2.INTER_CUBIC)
    s__ = cv2.cvtColor(s__, cv2.COLOR_BGR2GRAY)
    # s__ = s__.flatten()
    s__ = np.array([s__[:,:, None]])
    return s__

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='scenario2.json', record='.')
num_episodes = 1000
y = 0.9
eps = 0.99
decay_factor = 0.9
r_avg_list = []
for i in range(num_episodes):
    s = convert(env.reset())
    eps *= decay_factor
    s_eps = eps
    bored_helper = pow(s_eps,2000)
    bored_helper = max(bored_helper, 0.0001)
    s_decay = 0.999
    if i % 50 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    last_reward = 0
    max_reward = 0
    total_reward = 0
    until_done = 400
    stuck = False
    while not (done or stuck):
        stuck = False
        if np.random.random() < s_eps:
            a = np.random.randint(0, 3)
            old_a = a
            a = num2action[int(a)]
        else:
            a = np.argmax(model.predict(s))
            old_a = a
            a = num2action[int(a)]
        new_s, r, done, _ = env.step(a)

        # skrypt do generowania nagród nie działał
        # poniższy fragment wylicza nagrodę w inny sposób
        # i próbuje stwierdzić, czy Sonic utknął w miejscu na zbyt długo
        r-=0.05
        total_reward += r
        if(total_reward > max_reward):
            max_reward = total_reward
            until_done=400
        else:
            until_done -=1
            if (until_done == 0):
                until_done = 400
                stuck = True
                r = -5

        if (r > max_reward):
            r = 1.1 * r
            max_reward = r
            until_done = 400
            s_eps = eps
        else:
            r = 0
            until_done -=1

            # zwiększa szansę losowego ruchu, jeśli Sonic utknął
            if(until_done==200): s_eps = 0.1
            if (until_done == 0):
                until_done = 400
                stuck = True
                r = -5
        if done: r = -5

        # zmniejszanie rozmiaru danych wejściowych
        # (obrazu ekranu gry)
        new_s = convert(new_s)
        env.render()
        target = r + y * np.max(model.predict(new_s))
        target_vec = model.predict(new_s)[0]
        target_vec[old_a] = target
        model.fit(s, target_vec.reshape(-1, 4), epochs=1, verbose=0)
        s = new_s
        r_sum += r
    r_avg_list.append(r_sum / 1000)

# model.save('my_model-1.h5')