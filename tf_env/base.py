"""
Interface for batched TensorFlow environments.
"""

from abc import ABC, abstractmethod, abstractproperty


class TFEnv(ABC):
    """
    The base class for TensorFlow environments.
    """
    @abstractproperty
    def num_actions(self):
        """
        Get the size of the discrete action space as a
        Python integer.
        """
        pass

    @abstractmethod
    def reset(self, batch_size):
        """
        Create a batch of initialized states.

        Args:
          batch_size: an integer Tensor specifying the
            number of environments to initialize.

        Returns:
          A batch of states. This is a Tensor with outer
            dimension batch_size.
        """
        pass

    @abstractmethod
    def step(self, states, actions):
        """
        Advance the environment states by one step.

        Args:
          states: a Tensor of environment states.
          actions: a Tensor of integer action values.

        Returns:
          A tuple (states, news, rewards):
            states: a Tensor of new states.
            rewards: a Tensor of float32 rewards.
            dones: a Tensor of booleans, indicating if
              each environment was reset.
        """
        pass

    @abstractmethod
    def observe(self, states):
        """
        Turn states into observations.

        This can be used to implement partially observable
        environments.

        Args:
          states: a batch of environment states.

        Returns:
          A batch of observations. For example, for visual
          environments, this could be a uint8 Tensor of
           shape [batch_size x height x width x 3].
        """
        pass
