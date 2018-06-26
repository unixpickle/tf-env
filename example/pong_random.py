"""
Watch random agent play Pong.
"""

from tf_env import Pong

from watch_random import watch_random


def main():
    env = Pong()
    watch_random(env)


if __name__ == '__main__':
    main()
