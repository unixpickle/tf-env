from setuptools import setup

setup(
    name='tf-env',
    version='0.0.3',
    description='RL environments for TensorFlow.',
    url='https://github.com/unixpickle/tf-env',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    packages=['tf_env'],
    install_requires=[
        'numpy',
        'anyrl>=0.11.0,<0.13.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.4.0"],
        "tf_gpu": ["tensorflow-gpu>=1.4.0"]
    }
)
