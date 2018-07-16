from collections import deque

from models.pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import retro
from PIL import Image
import cv2
import math


def convert(s__):
    s__ = cv2.resize(s__, dsize=(80, 56), interpolation=cv2.INTER_CUBIC)
    s__ = cv2.cvtColor(s__, cv2.COLOR_BGR2GRAY)
    s_aux =  np.array(s__[:, :])
    s__ = np.array(s__[:, :, None])
    return s__


env_name = 'CartPole-v0'
env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='scenario2.json', record='.')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format(env_name))

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n


def policy_network(states):
    # define policy neural network

    conv_1 = tf.layers.conv2d(states,
                              filters=32,
                              kernel_size=[5, 5],
                              padding="same",
                              activation=tf.nn.relu)

    # pooling 1
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1,
                                     pool_size=[2, 2],
                                     strides=2)

    conv_2 = tf.layers.conv2d(inputs=pool_1,
                              filters=64,
                              kernel_size=[5, 5],
                              padding='same',
                              activation=tf.nn.relu)

    pool_2 = tf.layers.max_pooling2d(inputs=conv_2,
                                     pool_size=[2, 2],
                                     strides=2)

    flatten_1 = tf.reshape(pool_2, [-1, 17920])

    dense_1 = tf.layers.dense(inputs=flatten_1, units=num_actions)

    return dense_1


pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       [1, 56, 80, 1],
                                       num_actions,
                                       summary_writer=writer)

MAX_EPISODES = 10000
MAX_STEPS = 30000
START_STEPS = 500
STEPS = START_STEPS
WAIT = 1000

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

    # initialize
    state = env.reset()
    print(state.shape)

    state = convert(state)
    print(state.shape)
    # exit(0)
    total_rewards = 0

    last_reward = 0
    sum_reward = 0
    max_reward = -1337
    total_reward = 0
    until_break = WAIT
    stuck = False

    print(STEPS)
    for t in range(STEPS):
        env.render()
        action = pg_reinforce.sampleAction(state[np.newaxis, :])
        action_array = np.identity(12, dtype=int)[action:action+1][0]
        next_state, reward, done, _ = env.step(action_array)

        reward -= 0.9
        # some crude tricks
        if max_reward ==-1337:
            max_reward = 0
            reward = 0
        if reward > max_reward:
            tmp = reward
            reward = max_reward - reward
            max_reward = tmp
            until_break = WAIT
            reward *= 5
        else:
            until_break-=1
            reward = 0
            if until_break < 0:
                stuck = True




        next_state = convert(next_state)

        reward -= last_reward
        last_reward = reward

        total_rewards += reward
        reward = -10 if done else 0.1  # normalize reward
        pg_reinforce.storeRollout(state, action, reward)

        state = next_state
        if done or stuck:
            break


    # STEPS *=2
    STEPS +=200
    STEPS = min(STEPS, MAX_STEPS)
    pg_reinforce.updateModel()

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t + 1))
    print("Reward for this episode: {}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
    if mean_rewards >= 195.0 and len(episode_history) >= 100:
        print("Environment {} solved after {} episodes".format(env_name, i_episode + 1))
        break
