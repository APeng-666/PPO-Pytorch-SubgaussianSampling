import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import numpy as np

class PPO_1:
    def __init__(self, network, obs_dim, act_dim, subgaussian = False, delta =0.5, lr = 0.005, gamma = 0.99, clip = 0.2,n_updates_per_iteration = 5, \
                max_timesteps_per_episode = 200, timesteps_per_batch = 2048, chkpt_dir = 'model/'):
        self.chkpt_dir = chkpt_dir
        
        self.timesteps_per_batch = timesteps_per_batch                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = max_timesteps_per_episode          # Max number of timesteps per episode
        self.n_updates_per_iteration = n_updates_per_iteration                # Number of times to update actor/critic per iteration
        self.lr = lr                                 # Learning rate of actor optimizer
        self.gamma = gamma                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = clip                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.subgaussian = subgaussian
        self.delta = delta
        
        name = '.pth'
        if subgaussian:
            name = '_subgaussian.pth'
        
        self.actor = network(self.obs_dim, self.act_dim, 'ppo_actor'+name, self.chkpt_dir) 
        self.critic = network(self.obs_dim, 1, 'ppo_critic'+name, self.chkpt_dir)

        self.device = self.actor.device
        
        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = T.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = T.diag(self.cov_var).to(self.device)
        
    def learn(self, batch_obs, batch_acts, batch_rews, batch_log_probs, batch_lens, simple_times = False):
        
        device = self.actor.device
        
        batch_rtgs = self.compute_rtgs(batch_rews)
        batch_rtgs = T.tensor(np.array(batch_rtgs), dtype=T.float).to(device)
        batch_obs = T.tensor(np.array(batch_obs), dtype=T.float).to(device)
        batch_acts = T.tensor(np.array(batch_acts), dtype=T.float).to(device)
        batch_log_probs = T.tensor(np.array(batch_log_probs), dtype=T.float).to(device)
        batch_lens = T.tensor(np.array(batch_lens), dtype=T.float).to(device)
        
        # Calculate advantage at k-th iteration
        V, _ = self.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()          
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalizing the advantage
        
        for itera in range(self.n_updates_per_iteration):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            
            ratios = T.exp(curr_log_probs - batch_log_probs) # Vanilla importance sampling
            
            # Implement subgaussian importance sampling
            if itera > 0:
                var = ratios.var()
                if self.subgaussian:
                    if simple_times:
                #var = T.exp(curr_log_probs - batch_log_probs).var()
                        var_1 = var.sqrt() * self.delta # Seems to work good
                    else:
                        var_1 = 1.0/((var+ 1e-10) * batch_lens.sum())
                        var_1 = var_1.sqrt() * self.delta
                
                # Original sqrt{ frac{2 * log(1/delta)}{3 n}}, which should be used after the first iteration
                # The problem is that var could be extremely small
                    print(f"-Lambda: {var.item()} and {var_1.item()}-")
                    ratios = T.div(ratios, 1-var_1 + var_1 * ratios)
                else:
                    print(f"-Variance: {var.item()}")
                
            
            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = T.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
            
            actor_loss = (-T.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            
            # Calculate gradients and perform backward propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            
            # Calculate gradients and perform backward propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()    
            
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        
        for epoch_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(epoch_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        return batch_rtgs
        
    def get_action(self, obs):
        state = T.tensor(obs, dtype=T.float).to(self.actor.device)
        mean = self.actor.forward(state)
        
        distr = MultivariateNormal(mean, self.cov_mat)
        action = distr.sample()
        log_prob = distr.log_prob(action)
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()
    
        
    def evaluate(self, batch_obs, batch_acts):
        # Estimate the values of each observation, and the log probs of
        # each action in the most recent batch with the most recent
        # iteration of the actor network. Should be called from learn.
        
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic.forward(batch_obs).squeeze()
        
        # Calculate the log probabilities of batch actions using most recent actor network.
        mean = self.actor.forward(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # V - the predicted values of batch_obs
        # log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        return V, log_probs
    
        
