"""
Compatibility with anyrl: https://github.com/unixpickle/anyrl-py.
"""

from anyrl.envs import BatchedEnv
import gym
import tensorflow as tf


class TFBatchedEnv(BatchedEnv):
    """
    An anyrl BatchedEnv wrapper around TFEnv.
    """

    def __init__(self, session, env, batch_size):
        """
        Create a new TFBatchedEnv.

        Args:
          session: a TensorFlow session.
          env: the TFEnv to wrap.
          batch_size: the number of environments to
            simulate.
        """
        self.session = session
        self.env = env
        self.batch_size = batch_size

        self._state = None
        self._action = None
        self._resetting = False

        self._init_state = env.reset(batch_size)
        self._state_ph = tf.placeholder(self._init_state.dtype, shape=self._init_state.get_shape())
        self._action_ph = tf.placeholder(tf.int32, shape=[batch_size])
        self._step_out = env.step(self._state_ph, self._action_ph)
        self._observation = env.observe(self._state_ph)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-10000,
            high=10000,
            shape=[x.value for x in self._observation.get_shape()[1:]],
            dtype=self._observation.dtype.as_numpy_dtype,
        )

    @property
    def num_sub_batches(self):
        return 1

    @property
    def num_envs_per_sub_batch(self):
        return self.batch_size

    def reset_start(self, sub_batch=0):
        assert sub_batch == 0
        assert not self._resetting
        assert self._action is None
        self._state = None
        self._resetting = True

    def reset_wait(self, sub_batch=0):
        assert sub_batch == 0
        assert self._resetting
        self._state = self.session.run(self._init_state)
        self._resetting = False
        return self._make_observation()

    def step_start(self, actions, sub_batch=0):
        assert sub_batch == 0
        assert actions is not None
        assert len(actions) == self.batch_size
        assert self._action is None
        assert not self._resetting
        self._action = actions

    def step_wait(self, sub_batch=0):
        assert sub_batch == 0
        assert self._action is not None
        self._state, rews, dones = self.session.run(self._step_out,
                                                    feed_dict={self._state_ph: self._state,
                                                               self._action_ph: self._action})
        self._action = None
        return self._make_observation(), rews, dones, [{} for _ in dones]

    def close(self):
        pass

    def _make_observation(self):
        return self.session.run(self._observation, feed_dict={self._state_ph: self._state})
