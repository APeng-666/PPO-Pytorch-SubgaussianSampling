import gym
import time
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np

from cfg import cfg
from utils import fix
from utils import rollout
from ppo_1 import PPO_1
from network import FeedForwardNN


env = gym.make(cfg['env_name'])
fix(env, cfg['seed'])
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print(env.observation_space)
print(env.action_space)


model = PPO_1(network=FeedForwardNN, obs_dim = obs_dim, act_dim = act_dim,
              subgaussian = cfg['subgaussian'], delta=cfg['delta'], lr = cfg['lr'], gamma= cfg['gamma'],
              clip=cfg['clip'], n_updates_per_iteration = cfg['n_updates_per_iteration'],
              max_timesteps_per_episode=cfg['max_timesteps_per_episode'],
              timesteps_per_batch=cfg['timesteps_per_batch'])
#model.load_checkpoint()
#model.critic.load_checkpoint()
model.actor.train()
model.critic.train()

print(cfg['delta'])


tmp = 0.0
avg_ep_rews_log = []

prg_bar = tqdm(range(cfg['total_iterations']))
for i in prg_bar:
    batch_obs, batch_acts, batch_rews, batch_log_probs, batch_lens = \
        rollout(model, env, cfg['timesteps_per_batch'], cfg['max_timesteps_per_episode'])

    avg_ep_lens = np.mean(batch_lens)
    avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
    avg_ep_rews_log.append(avg_ep_rews)
    
    # Round decimal places for more aesthetic logging messages
    avg_ep_lens = str(round(avg_ep_lens, 2))
    avg_ep_rews_1 = str(round(avg_ep_rews, 2))

    print(f"-------------------- Iteration --------------------", flush=True)
    print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    print(f"Average Episodic Return: {avg_ep_rews_1}", flush=True)
    
    model.learn(batch_obs, batch_acts, batch_rews, batch_log_probs, batch_lens)
    if avg_ep_rews > tmp:
        model.actor.save_checkpoint()
        model.critic.save_checkpoint()
        print("Successfully save the model!",flush=True)
    tmp = avg_ep_rews

def plot_one():
    data_name = 'data/avg_ep_rews_'+ cfg['fig_name'] + '.npy'

    plt.plot(avg_ep_rews_log[0:150], color = cfg['color'])
    fig_name = cfg['env_name']+ '_average rewards ('+ cfg['fig_name']+ ')'
    file_name = 'figures/'+ cfg['env_name'] + '_'+ cfg['fig_name'] + '.png'
    np.save(data_name, np.array(avg_ep_rews_log))
    plt.title(fig_name)
    plt.savefig(file_name, bbox_inches='tight')
    
plot_one()
