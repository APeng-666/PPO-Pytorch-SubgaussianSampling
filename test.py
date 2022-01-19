import gym
import numpy as np

from cfg import cfg
from utils import fix
from ppo_1 import PPO_1
from network import FeedForwardNN

env = gym.make(cfg['env_name'])
fix(env, cfg['seed'])
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = PPO_1(network=FeedForwardNN, obs_dim = obs_dim, act_dim = act_dim, subgaussian = True)
model.actor.load_checkpoint()
model.critic.load_checkpoint()
model.actor.eval()  # turn network to evaluation mode
model.critic.eval()
NUM_OF_TEST = 5 # Do not revise it !!!!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
    actions = []
    obs = env.reset()

  #img = plt.imshow(env.render(mode='rgb_array'))
    env.render()
  
    total_reward = 0

    done = False
    while not done:
        action, _ = model.get_action(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)

        total_reward += reward

      #img.set_data(env.render(mode='rgb_array'))
        env.render()
      #display.display(plt.gcf())
      #display.clear_output(wait=True)
    print(total_reward)
    test_total_reward.append(total_reward)


env.close()

print(f"Your final reward is : %.2f"%np.mean(test_total_reward))
