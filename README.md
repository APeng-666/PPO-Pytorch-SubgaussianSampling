# PPO-Pytorch-SubgaussianSampling
We implement a new importance weight - subgaussian sampling - for the ppo algorithm.

Empirical results show that delta >=0.5 leads to a faster convergent rate than the ppo algorithm with vanillia importance sampling.

TO DO: Test in more environments and figure out how to choose the optimal hyperparameter "delta" in different settings.

Reference: Alberto, et.al, Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning, NIPS, 2021.
