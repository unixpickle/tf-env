"""
Train a PPO agent on the Pong environment.
"""

from itertools import count
import pickle

from anyrl.algos import PPO
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import CNN
from anyrl.rollouts import TruncatedRoller
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import tensorflow as tf
from tf_env import Pong, TFBatchedEnv


def main():
    with tf.Session() as sess:
        print('Creating environment...')
        env = TFBatchedEnv(sess, Pong(), 8)
        env = BatchedFrameStack(env)

        print('Creating model...')
        model = CNN(sess,
                    gym_space_distribution(env.action_space),
                    gym_space_vectorizer(env.observation_space))

        print('Creating roller...')
        roller = TruncatedRoller(env, model, 128)

        print('Creating PPO graph...')
        ppo = PPO(model)
        optimize = ppo.optimize(learning_rate=3e-4)

        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())

        print('Training agent...')
        for i in count():
            rollouts = roller.rollouts()
            for rollout in rollouts:
                if not rollout.trunc_end:
                    print('reward=%f steps=%d' % (rollout.total_reward, rollout.total_steps))
            total_steps = sum(r.num_steps for r in rollouts)
            ppo.run_optimize(optimize, rollouts,
                             batch_size=total_steps // 4,
                             num_iter=12,
                             log_fn=print)
            if i % 5 == 0:
                print('Saving...')
                parameters = sess.run(tf.trainable_variables())
                with open('params.pkl', 'wb+') as out_file:
                    pickle.dump(parameters, out_file)


if __name__ == '__main__':
    main()
