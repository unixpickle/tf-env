"""
Watch a trained PPO agent on the Pong environment.
"""

import os
import pickle

from anyrl.models import CNN
from anyrl.rollouts import TruncatedRoller
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
from gym.envs.classic_control.rendering import SimpleImageViewer
import tensorflow as tf
from tf_env import Pong, TFBatchedEnv


def main():
    with tf.Session() as sess:
        print('Creating environment...')
        env = TFBatchedEnv(sess, Pong(), 1)

        print('Creating model...')
        model = CNN(sess,
                    gym_space_distribution(env.action_space),
                    gym_space_vectorizer(env.observation_space))

        print('Creating roller...')
        roller = TruncatedRoller(env, model, 1)

        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())

        if os.path.exists('params.pkl'):
            print('Loading parameters...')
            with open('params.pkl', 'rb') as in_file:
                params = pickle.load(in_file)
            for var, val in zip(tf.trainable_variables(), params):
                sess.run(tf.assign(var, val))
        else:
            print('Warning: parameter file does not exist!')

        print('Running agent...')
        viewer = SimpleImageViewer()
        while True:
            for obs in roller.rollouts()[0].step_observations:
                viewer.imshow(obs)


if __name__ == '__main__':
    main()
