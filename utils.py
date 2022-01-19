import numpy as np
import random
import matplotlib.pyplot as plt

import torch as T

def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
  #torch.set_deterministic(True)
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.deterministic = True


# Define the rollout function to collect the on-policy trajectories
def rollout(policy, env, timesteps_per_batch, max_timesteps_per_episode):
    # Batch data.
    batch_obs = []
    batch_acts = []
    batch_log_probs = []
    batch_rews = []
    batch_lens = []
    
    epoch_rews = []
    t=0
    while t < timesteps_per_batch:
        epoch_rews = [] # rewards collected per episode
        obs = env.reset()
        done = False
        for h in range(max_timesteps_per_episode):
            batch_obs.append(obs)
            action, log_prob = policy.get_action(obs)
            obs, rew, done, _ = env.step(action)
            t +=1
            
            epoch_rews.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            
            if done:
                break
            
        batch_lens.append(h+1)
        batch_rews.append(epoch_rews)
        
    return batch_obs, batch_acts, batch_rews, batch_log_probs, batch_lens


def plot_all_figures():
    data0= np.load('data/avg_ep_rews_vanilla.npy')
    data1= np.load('data/avg_ep_rews_subgaussian_05.npy')
    data2= np.load('data/avg_ep_rews_subgaussian_02.npy')
    data3= np.load('data/avg_ep_rews_subgaussian_08.npy')
    line0, = plt.plot(data0, color = 'blue')
    line1, = plt.plot(data1, color = 'red')
    line2, = plt.plot(data2, color = 'green')
    line3, = plt.plot(data3, color = 'yellow')
    plt.legend(handles = [line0, line1, line2, line3],
               labels =['Vanilla', 'Subgaussian(0.5)','Subgaussian(0.2)','Subgaussian(0.8)'],
               loc='lower right')
    plt.title("average rewards")
    plt.savefig('figures/'+ cfg['env_name']+'_both050208', bbox_inches='tight')
