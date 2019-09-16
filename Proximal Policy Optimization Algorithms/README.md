# Proximal Policy Optimization Algorithms, OpenAI 2017
Arxiv: https://arxiv.org/abs/1707.06347  
Blog: https://openai.com/blog/openai-baselines-ppo/

## Overview
PPO repleces KL divergence by the custom function:

$$
L^{CLIP}(\theta) = \hat E_t[min(r_t(\theta) \hat A_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t )]
$$
Where:
- $\theta$ is the policy parameters vector
- $\hat{E}_t$ is the empirical expectation over timesteps
- $r_{t}$ is the following ratio:  [__Comment: not sure if this formula is correct__]
    $$
    r_t = \frac{\pi_{new}(\cdot | s_t)}{\pi_{old}(\cdot | s_t)}
    $$
- $\hat{A}_t$ is the estimated advantage at time $t$
- $\varepsilon$ is a hyperparameter, usually 0.1 or 0.2

>This objective implements a way to do a Trust Region update which is compatible with Stochastic Gradient Descent, and simplifies the algorithm by removing the KL penalty and need to make adaptive updates.


## Detailed analysis
The goal of this section is to analyse each section of the paper.

### Background: Policy Optimization

#### Equations and notation from the [TRPO paper](https://arxiv.org/abs/1502.05477)
Expected discounted reward when using policy $\pi$ ( $a_t \sim \pi(a_t|s_t)$ )
$$
\eta(\pi) = \mathop{\mathbb{E}}_{\pi} [\sum^\infin_{t=0}\gamma^t r(s_t)]
$$

Expected return of another policy $\tilde \pi$ in terms of the advantage over policy $\pi$

$$
\eta(\tilde \pi) = \eta(\pi) + \mathop{\mathbb{E}}_{\tilde \pi} [\sum^\infin_{t=0}\gamma^t A_\pi(a_t, s_t)]
$$

Local approximation to the above equation (we only compare the decision and follow the old policy $\pi$)

$$
L_\pi (\tilde \pi) = \eta(\pi) + \sum_s\rho_\pi (s) \sum_a \tilde\pi(a|s) A_\pi(a_t, s_t)
$$

where $\rho_\pi (s)$ is the (unnormalized) discounted visitation frequencies under policy $\pi$
$$
\rho_\pi (s) = P(s_0 = s) + \gamma P(s_1 = s)  + \gamma^2 P(s_2 = s)+...
$$

This approximation has the same gradient for $\theta_0$ (the policy parameters vector)
$$
L_{\pi_{\theta_0}} (\pi_{\theta_0}) = \eta(\pi_{\theta_0})
$$
$$
\left.\nabla_\theta  L_{\pi_{\theta_0}} (\pi_{\theta}) \right |_{\theta_0} = \left.\nabla_\theta  \eta(\pi_{\theta}) \right |_{\theta_0} 
$$

The rest of equation has the notation from the PPO paper.
#### Policy Gradient Methods
The most popular policy objective function
$$
L^{PG}(\theta) =  \mathop{\mathbb{\hat E}}_t [\text{log} \pi_\theta  (a_t| s_t) \hat A_t]
$$
Its gradient
$$
\hat g =  \mathop{\mathbb{\hat E}}_t [\nabla _\theta\  \text{log} \pi_\theta  (a_t| s_t) \hat A_t]
$$

#### Trust Region Methods
In the TRPO, an objective function is maximized subject to a
constraint on the size of the policy update.
$$
\underset{\theta}{\text{maximize}} \ \mathop{\mathbb{\hat E}}_t  \left[ \frac{\pi_\theta  (a_t| s_t)}{\pi_{\theta \text{old}}  (a_t| s_t)} \hat A_t \right]
$$

$$
\text{subject to} \ \mathop{\mathbb{\hat E}}_t  \left[ \text{KL}[ \pi_{\theta \text{old}} (\cdot| s_t), \pi_{\theta} (\cdot| s_t)] \right] \leq \delta
$$

KL divergence ensures that the next policy will be simillar to the previous one and we don't overshoot with the step size.

### Clipped Surrogate Objective

Let $r_t(\theta)$ denote the probability ration

$$
r_t(\theta) = \frac{\pi_\theta  (a_t| s_t)}{\pi_{\theta \text{old}}  (a_t| s_t)}
$$

So TRPO maximizes a "surrogate" objective (CPI refers to conservative policy iteration)

$$
L^{\text{CPI}} (\theta) = \mathop{\mathbb{\hat E}}_t  \left[ r_t(\theta) \hat A_t \right]
$$

Proposed in the PPO objective

$$
L^{\text{CLIP}} (\theta) = \mathop{\mathbb{\hat E}}_t  \left[ \text{min}(\  r_t(\theta) \hat A_t\ , \ \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon )  \hat A_t)\right]
$$

where epsilon is a hyperparameter,  usually 0.1 or 0.2. 
This provides us with the pesymistic evaluation of the policy comparing to the previous one (in the $r_t(\theta)$). We have to take such approach because $L^{\text{CLIP}}$ is only an approximation and we need to work in the neigbourhood of the $\pi_{\theta \text{old}}$.


### Adaptive KL Penalty Coefficient

## Related works
1. [OpenAI Algorithm implementations](https://github.com/openai/baselines)
2. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
3. [Sample Efficient Actor-Critic with Experience Replay - ACER](https://arxiv.org/abs/1611.01224)


## Questions
1. They proposed objective but what with the derivatives? How to get gradient of the clip function (make some intervals?)?
2. Fig. 2 explanation


## [Comment] Call for help at the end of blog post  
>We’re looking for people to help build and optimize our reinforcement learning algorithm codebase. If you’re excited about RL, benchmarking, thorough experimentation, and open source, please apply, and mention that you read the baselines PPO post in your application.

We are going to do it anyway and we could have track of the most popular algorithms.
