"""
Measure the speed of the Pong environment.
"""

import time

import tensorflow as tf

from tf_env import Pong


NUM_STEPS = 512


def main():
    env = Pong()
    with tf.Session() as sess:
        for batch_size in [2 ** i for i in range(10)]:
            for render in [False, True]:
                init_state = env.reset(batch_size)
                states = tf.placeholder(init_state.dtype, shape=init_state.get_shape())
                actions = tf.random_uniform([batch_size], minval=0,
                                            maxval=env.num_actions, dtype=tf.int32)
                new_states, rews, dones = env.step(states, actions)
                image = env.observe(states)
                cur_states = sess.run(init_state)
                start_time = time.time()
                for _ in range(NUM_STEPS):
                    cur_states, _, _ = sess.run([new_states, rews, dones],
                                                feed_dict={states: cur_states})
                    if render:
                        sess.run(image, feed_dict={states: cur_states})
                fps = NUM_STEPS * batch_size / (time.time() - start_time)
                print('fps is %f with render=%s batch=%d' % (fps, render, batch_size))


if __name__ == '__main__':
    main()
