"""
A gridworld environment where the agent has to reach cells
before the end of an episode.
"""

import tensorflow as tf

from .base import TFEnv
from .util import bcast_where, excluded_random


class PointSeeker(TFEnv):
    """
    A batched "point seeker" environment.

    Observations are continuous x,y positions for both the
    player and the target.
    However, there is also an observe_visual() method that
    produces a rendering of the gridworld.

    The action space is either [UP, DOWN, LEFT, RIGHT] or
    [NOP, UP, DOWN, LEFT, RIGHT] depending on whether or
    not no-ops are enabled.

    The state contains the following signed integers:
     - current X
     - current Y
     - goal X
     - goal Y
     - steps remaining

    The coordinate system works the same way as for Pong.
    """

    def __init__(self,
                 width=10,
                 height=10,
                 timestep_limit=64,
                 reward_decay=0.99,
                 nops=False):
        self.width = tf.convert_to_tensor(width, dtype=tf.int32)
        self.height = tf.convert_to_tensor(height, dtype=tf.int32)
        self.timestep_limit = tf.convert_to_tensor(timestep_limit, dtype=tf.int32)
        self.reward_decay = tf.convert_to_tensor(reward_decay, dtype=tf.float32)
        self.nops = nops

    @property
    def num_actions(self):
        if self.nops:
            return 5
        return 4

    def reset(self, batch_size):
        player_pos = tf.random_uniform(shape=[batch_size, 2],
                                       minval=0,
                                       maxval=self.width - 1,
                                       dtype=tf.int32)
        num_cells = tf.cast(self.width * self.height, tf.float32)
        same_x_prob = tf.cast(self.height - 1, tf.float32) / num_cells
        same_x = tf.random_uniform([batch_size]) < same_x_prob
        goal_x = tf.where(same_x,
                          player_pos[:, 0],
                          excluded_random(batch_size, self.width, player_pos[:, 0]))
        goal_y = tf.where(same_x,
                          excluded_random(batch_size, self.height, player_pos[:, 1]),
                          tf.random_uniform(shape=[batch_size],
                                            minval=0,
                                            maxval=self.height - 1,
                                            dtype=tf.int32))
        goal_pos = tf.stack([goal_x, goal_y], axis=-1)
        steps_remaining = tf.zeros([batch_size, 1], dtype=tf.int32) + self.timestep_limit
        return tf.concat([player_pos, goal_pos, steps_remaining], axis=-1)

    def step(self, states, actions):
        player_x = states[:, 0]
        player_y = states[:, 1]
        goal_x = states[:, 2]
        goal_y = states[:, 3]
        steps_remaining = states[:, 4]

        x_delta, y_delta = self._action_deltas(actions)

        player_x = wrapped_add(player_x, x_delta, self.width)
        player_y = wrapped_add(player_y, y_delta, self.height)
        steps_remaining = steps_remaining - 1

        new_states = tf.stack([player_x, player_y, goal_x, goal_y, steps_remaining], axis=-1)

        at_goals = tf.logical_and(tf.equal(player_x, goal_x), tf.equal(player_y, goal_y))
        dones = tf.logical_or(steps_remaining <= 0, at_goals)
        rews = self._rewards(at_goals, steps_remaining)

        return tf.where(dones, self.reset(tf.shape(states)[0]), new_states), rews, dones

    def observe(self, states):
        states = tf.cast(states, tf.float32)
        width = tf.cast(self.width, tf.float32)
        height = tf.cast(self.height, tf.float32)
        player_x = states[:, 0] / width
        player_y = states[:, 1] / height
        goal_x = states[:, 2] / width
        goal_y = states[:, 3] / height
        steps_remaining = states[:, 4] / tf.cast(self.timestep_limit, tf.float32)
        return tf.stack([player_x, player_y, goal_x, goal_y, steps_remaining], axis=-1)

    def observe_visual(self, states):
        batch = tf.shape(states)[0]
        xs = tf.tile(tf.expand_dims(tf.range(0, self.width), axis=0), [batch, 1])
        ys = tf.tile(tf.expand_dims(tf.range(0, self.height), axis=0), [batch, 1])

        def point_mask(x, y):
            x = tf.reshape(x, [batch, 1])
            y = tf.reshape(y, [batch, 1])
            x_mask = tf.reshape(tf.equal(x, xs), [batch, 1, self.width])
            y_mask = tf.reshape(tf.equal(y, ys), [batch, self.height, 1])
            return tf.logical_and(x_mask, y_mask)

        player_mask = point_mask(states[:, 0], states[:, 1])
        goal_mask = point_mask(states[:, 2], states[:, 3])
        stacked = tf.stack([player_mask, tf.zeros_like(player_mask), goal_mask], axis=-1)
        return tf.cast(stacked, tf.uint8) * tf.constant(255, dtype=tf.uint8)

    def _rewards(self, at_goals, steps_remaining):
        rews = tf.cast(at_goals, tf.float32)
        steps_taken = tf.cast(self.timestep_limit - steps_remaining, tf.float32)
        decays = tf.exp(steps_taken * tf.log(self.reward_decay))
        return decays * rews

    def _action_deltas(self, actions):
        up = 0
        down = 1
        left = 2
        right = 3
        if self.nops:
            up += 1
            down += 1
            left += 1
            right += 1
        positives = tf.ones(tf.shape(actions), dtype=tf.int32)
        negatives = tf.negative(positives)
        zeros = tf.zeros_like(positives)
        x_delta = tf.where(tf.equal(actions, left),
                           negatives,
                           tf.where(tf.equal(actions, right),
                                    positives,
                                    zeros))
        y_delta = tf.where(tf.equal(actions, up),
                           negatives,
                           tf.where(tf.equal(actions, down),
                                    positives,
                                    zeros))
        return x_delta, y_delta


def wrapped_add(vectors, values, maxval):
    sums = tf.add(vectors, values)
    return bcast_where(sums < 0,
                       maxval - 1,
                       tf.where(sums >= maxval,
                                tf.zeros_like(sums),
                                sums))
