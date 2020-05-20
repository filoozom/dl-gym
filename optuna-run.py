import numpy as np
import optuna

import sys
sys.path.append('gym')
from gym.envs.box2d import CarRacing

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy, CnnLnLstmPolicy, FeedForwardPolicy, MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

env_kwargs = { 'verbose': 0 }

def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

counter = 0

def optimize_agent(trial):
    global counter

    model_params = optimize_ppo2(trial)
    env = make_vec_env(CarRacing, env_kwargs=env_kwargs, n_envs=1, monitor_dir='./monitor', vec_env_cls=DummyVecEnv)
    model = PPO2(CnnPolicy, env, verbose=0, nminibatches=1, **model_params)
    model.learn(100000)

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    # Calculate mean performance of 5 runs
    obs = env.reset()
    while n_episodes < 5:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            rewards.append(reward_sum)
            reward_sum = 0.0
            n_episodes += 1
            obs = env.reset()

    reward = np.mean(rewards)
    trial.report(reward, counter)
    counter += 1

    # Close the environment
    env.close()

    return reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', study_name='carracing_optuna', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(optimize_agent, timeout=5*60*60, show_progress_bar=True)
    print(study.best_params)
