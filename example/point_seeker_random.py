"""
Watch random agent play PointSeeker.
"""

from tf_env import PointSeeker

from watch_random import watch_random


def main():
    env = PointSeeker()
    watch_random(env, frame_rate=20.0)


if __name__ == '__main__':
    main()
