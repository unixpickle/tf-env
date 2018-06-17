"""
A basic version of Pong, implemented in TensorFlow.
"""

import tensorflow as tf

from .base import TFEnv
from .util import bcast_where


class Pong(TFEnv):
    """
    A batched Pong environment in TensorFlow.

    The action space is [NOP, UP, DOWN].

    The state contains the following signed integers:
     - enemy paddle Y (from top of the paddle).
     - player paddle Y (from top of the paddle).
     - ball X (from left of the ball).
     - ball Y (from top of the ball).
     - ball velocity X
     - ball velocity Y
     - player paddle velocity

    The game coordinate system works as follows:
      The top left of the screen is (0, 0).
      Moving to the right advances by (1, 0).
      Moving down advances by (0, 1).
    """

    def __init__(self,
                 width=80,
                 height=100,
                 paddle_width=3,
                 paddle_height=15,
                 paddle_margin=5,
                 ball_width=2,
                 ball_height=3,
                 velocity_cap=3,
                 x_speed=1,
                 enemy_speed=2):
        self.raw_width = width
        self.raw_height = height
        self.width = tf.convert_to_tensor(width, dtype=tf.int32)
        self.height = tf.convert_to_tensor(height, dtype=tf.int32)
        self.paddle_width = tf.convert_to_tensor(paddle_width, dtype=tf.int32)
        self.paddle_height = tf.convert_to_tensor(paddle_height, dtype=tf.int32)
        self.paddle_margin = tf.convert_to_tensor(paddle_margin, dtype=tf.int32)
        self.ball_width = tf.convert_to_tensor(ball_width, dtype=tf.int32)
        self.ball_height = tf.convert_to_tensor(ball_height, dtype=tf.int32)
        self.velocity_cap = tf.convert_to_tensor(velocity_cap, dtype=tf.int32)
        self.x_speed = tf.convert_to_tensor(x_speed, dtype=tf.int32)
        self.enemy_speed = tf.convert_to_tensor(enemy_speed, dtype=tf.int32)

    @property
    def num_actions(self):
        return 3

    def reset(self, batch_size):
        paddle_ys = tf.random_uniform(shape=[batch_size, 2],
                                      minval=0,
                                      maxval=self.height - self.paddle_height,
                                      dtype=tf.int32)
        ball_positions = tf.zeros([batch_size, 2], dtype=tf.int32)
        ball_positions = ball_positions + tf.stack([self.width // 2, self.height // 2])
        ball_velocities = tf.random_uniform(shape=[batch_size, 2],
                                            minval=0,
                                            maxval=2,
                                            dtype=tf.int32)
        ball_velocities = ((ball_velocities * 2) - 1) * self.x_speed
        player_velocities = tf.zeros([batch_size, 1], dtype=tf.int32)
        return tf.concat([paddle_ys, ball_positions, ball_velocities, player_velocities], 1)

    def step(self, states, actions):
        enemy_paddle_y = states[:, 0]
        player_paddle_y = states[:, 1]
        ball_x = states[:, 2]
        ball_y = states[:, 3]
        ball_vel_x = states[:, 4]
        ball_vel_y = states[:, 5]
        player_vel = states[:, 6]

        ball_x = ball_x + ball_vel_x
        ball_y = ball_y + ball_vel_y

        player_vel = self._apply_player_action(player_vel, actions)
        player_paddle_y = self._clip_paddle_y(player_paddle_y + player_vel)
        enemy_vel = self._enemy_velocity(enemy_paddle_y, ball_y)
        enemy_paddle_y = self._clip_paddle_y(enemy_paddle_y + enemy_vel)

        ball_y, ball_vel_y = self._ball_vert_collision(ball_y, ball_vel_y)
        ball_x, ball_vel_x, ball_vel_y = self._player_collision(ball_x, ball_y,
                                                                ball_vel_x, ball_vel_y,
                                                                player_paddle_y, player_vel)
        ball_x, ball_vel_x, ball_vel_y = self._enemy_collision(ball_x, ball_y,
                                                               ball_vel_x, ball_vel_y,
                                                               enemy_paddle_y, enemy_vel)

        dones, rews = self._detect_finishes(ball_x)
        new_states = tf.stack([enemy_paddle_y, player_paddle_y, ball_x, ball_y, ball_vel_x,
                               ball_vel_y, player_vel], 1)
        new_states = tf.where(dones, self.reset(tf.shape(states)[0]), new_states)
        return new_states, rews, dones

    def observe(self, states):
        batch = tf.shape(states)[0]
        xs = tf.tile(tf.expand_dims(tf.range(0, self.width), axis=0), [batch, 1])
        ys = tf.tile(tf.expand_dims(tf.range(0, self.height), axis=0), [batch, 1])

        def rectangle_mask(min_x, max_x, min_y, max_y):
            min_x = tf.tile(min_x, [1, self.raw_width])
            max_x = tf.tile(max_x, [1, self.raw_width])
            min_y = tf.tile(min_y, [1, self.raw_height])
            max_y = tf.tile(max_y, [1, self.raw_height])

            x_mask = tf.logical_and(xs >= min_x, xs < max_x)
            y_mask = tf.logical_and(ys >= min_y, ys < max_y)
            x_mask = tf.reshape(x_mask, [batch, 1, self.raw_width])
            y_mask = tf.reshape(y_mask, [batch, self.raw_height, 1])

            return tf.logical_and(x_mask, y_mask)

        enemy_paddle_y = states[:, 0:1]
        player_paddle_y = states[:, 1:2]
        ball_x = states[:, 2:3]
        ball_y = states[:, 3:4]
        zeros = tf.zeros_like(enemy_paddle_y)

        ball_plane = rectangle_mask(ball_x, ball_x + self.ball_width,
                                    ball_y, ball_y + self.ball_height)
        enemy_plane = rectangle_mask(self.paddle_margin + zeros,
                                     self.paddle_margin + self.paddle_width + zeros,
                                     enemy_paddle_y, enemy_paddle_y + self.paddle_height)
        player_plane = rectangle_mask(self.width - self.paddle_margin - self.paddle_width + zeros,
                                      self.width - self.paddle_margin + zeros,
                                      player_paddle_y, player_paddle_y + self.paddle_height)

        res = tf.stack([enemy_plane, ball_plane, player_plane], axis=-1)
        return tf.cast(res, tf.uint8) * tf.constant(255, dtype=tf.uint8)

    def _apply_player_action(self, vels, actions):
        nops = tf.equal(actions, 0)
        ups = tf.equal(actions, 1)
        res = tf.where(nops,
                       tf.zeros_like(vels),
                       tf.where(ups, vels - 1, vels + 1))
        return tf.clip_by_value(res, tf.negative(self.velocity_cap), self.velocity_cap)

    def _clip_paddle_y(self, paddle_y):
        return tf.clip_by_value(paddle_y, 0, self.height - self.paddle_height)

    def _enemy_velocity(self, paddle_y, ball_y):
        paddle_center = paddle_y + self.paddle_height // 2
        ball_center = ball_y + self.ball_height // 2
        y_delta = ball_center - paddle_center
        return tf.clip_by_value(y_delta, tf.negative(self.enemy_speed), self.enemy_speed)

    def _ball_vert_collision(self, y, y_vel):
        top_collision = y <= 0
        bottom_collision = y + self.ball_height >= self.height
        collided = tf.logical_or(top_collision, bottom_collision)
        y = tf.clip_by_value(y, 0, self.height - self.ball_height)
        y_vel = tf.where(collided, tf.negative(y_vel), y_vel)
        return y, y_vel

    def _player_collision(self, x, y, vel_x, vel_y, player_y, player_vel):
        min_x = self.width - self.paddle_margin - self.paddle_width - self.ball_width
        max_x = self.width - self.paddle_margin
        min_y = player_y - self.ball_height
        max_y = player_y + self.paddle_height
        collided = tf.logical_and(tf.logical_and(x >= min_x, x <= max_x),
                                  tf.logical_and(y >= min_y, y <= max_y))

        x = bcast_where(collided, min_x, x)
        vel_x = tf.where(collided, tf.negative(vel_x), vel_x)
        vel_y = tf.where(collided, player_vel, vel_y)
        return x, vel_x, vel_y

    def _enemy_collision(self, x, y, vel_x, vel_y, enemy_y, enemy_vel):
        min_x = self.paddle_margin - self.ball_width
        max_x = self.paddle_margin + self.paddle_width
        min_y = enemy_y - self.ball_height
        max_y = enemy_y + self.paddle_height
        collided = tf.logical_and(tf.logical_and(x >= min_x, x <= max_x),
                                  tf.logical_and(y >= min_y, y <= max_y))

        x = bcast_where(collided, max_x, x)
        vel_x = tf.where(collided, tf.negative(vel_x), vel_x)
        vel_y = tf.where(collided, enemy_vel, vel_y)
        return x, vel_x, vel_y

    def _detect_finishes(self, ball_x):
        finished_left = ball_x + self.ball_width <= 0
        finished_right = ball_x > self.width
        dones = tf.logical_or(finished_left, finished_right)

        positive = tf.ones([tf.shape(dones)[0]], dtype=tf.float32)
        negative = tf.negative(positive)
        zeros = tf.zeros([tf.shape(dones)[0]], dtype=tf.float32)
        rews = tf.where(finished_left, positive, tf.where(finished_right, negative, zeros))

        return dones, rews
