# Add custom local gym folder to import path
import sys
sys.path.append('gym')
import gym
from gym.envs.box2d import CarRacing

# For training
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLnLstmPolicy, FeedForwardPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

# For training resume
import os

# For custom_extractor
import numpy as np
import tensorflow as tf
from stable_baselines.common.tf_layers import conv, conv_to_fc, linear, ortho_init

# Hide TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Command line arguments
import argparse

# Global parser
parser = argparse.ArgumentParser(description='Interacts with the CarRacing-v0 agent.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--verbose', dest='verbose', action='store_const', const=True, default=False, help='add logging messages')

# Commands sub-parser
commands = parser.add_subparsers(title="command", dest="command")

# Train command
train_parser = commands.add_parser("train", help="Train the agent")
train_parser.add_argument('-e', '--envs', dest='envs', type=int, default=os.cpu_count(), help='amount of environments to run in parallel')

# Benchmark command
benchmark_parser = commands.add_parser("benchmark", help="Benchmark the agent")
benchmark_parser.add_argument('-c', '--count', dest='count', type=int, default=50, help='amount of agents to run to produce the statistics')

# Video command
video_parser = commands.add_parser("video", help="Record videos of successive agents running")
video_parser.add_argument('-c', '--count', dest='count', type=int, default=5, help='amount of agents to record')

args = parser.parse_args()

if args.command is None:
    parser.print_usage()
    exit(1)

# Environment configuration
env_kwargs = { 'verbose': 1 if args.verbose else 0 }

# Based on nature_cnn
def custom_extractor(scaled_images, **kwargs):    
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=custom_extractor, feature_extraction="cnn")

if __name__ == '__main__':
    # Define the PPO2 arguments
    ppo_default = {
        'verbose': 1 if args.verbose else 0,
        'tensorboard_log': './dl-tensorboard/'
    }

    ppo_hypertune = {
        'gamma': 0.99,
        'n_steps': 32,
        'ent_coef': 0.01,
        'learning_rate': 0.00005,
        'vf_coef': 0.5,
        'max_grad_norm': 100,
        'lam': 0.95,
        'nminibatches': 4,
        'noptepochs': 4,
    }

    ppo_args = {**ppo_default, **ppo_hypertune}

    if args.command == 'train':
        # Use SubprocVecEnv because else it just uses one thread
        env = make_vec_env(CarRacing, env_kwargs=env_kwargs, n_envs=args.envs, monitor_dir='./monitor', vec_env_cls=SubprocVecEnv)

        # Resume training if saved data is available
        if os.path.exists("car.zip"):
            model = PPO2.load('car', env, **ppo_args)
        else:
            model = PPO2(CustomPolicy, env, **ppo_args)

        # Save the model each 250k steps, and run 2.5m steps total
        for i in range(10):
            model.learn(total_timesteps=250000, reset_num_timesteps=False)
            model.save("car")

    # Create a basic environment for all commands except train
    env = make_vec_env(CarRacing, env_kwargs=env_kwargs, n_envs=1, vec_env_cls=DummyVecEnv)
    model = PPO2.load('car', env, **ppo_args)

    if args.command == 'video':
        env = gym.make('CarRacing-v0', **env_kwargs)
        env = gym.wrappers.Monitor(env, "./videos-tmp", force=True, write_upon_reset=True)

        for _ in range(args.count):
            done = False
            reward = 0.0

            obs = env.reset()

            while not done:
                action, _states = model.predict(obs)
                obs, step_reward, done, _ = env.step(action)
                env.render(mode='human')
                reward += step_reward

            print('Score: %f' % reward)

    elif args.command == "benchmark":
        rewards = []

        for count in range(args.count):
            done = False
            reward = 0.0

            obs = env.reset()

            while not done:
                action, _states = model.predict(obs)
                obs, step_reward, done, _ = env.step(action)
                env.render(mode='human')
                reward += step_reward[0]
            
            rewards.append(reward)
            print("Completed experiment %d/%d with a score of %f" % (count + 1, args.count, reward))

        print("Mean reward: %f" % np.mean(rewards))
        print("Successful runs: %d/%d" % ((np.array(rewards) > 900).sum(), args.count))
