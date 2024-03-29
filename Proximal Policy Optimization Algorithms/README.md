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
- $r_{t}$ is the following ratio:
    $$
    r_t = \frac{\pi_{new}(a_t | s_t)}{\pi_{old}(a_t | s_t)}
    $$
- $\hat{A}_t$ is the estimated advantage at time $t$
- $\varepsilon$ is a hyperparameter, usually 0.1 or 0.2

>This objective implements a way to do a Trust Region update which is compatible with Stochastic Gradient Descent, and simplifies the algorithm by removing the KL penalty and need to make adaptive updates.


## Detailed analysis

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
Another approach is to use KL divergence directly in the objective function and using several epochs of minibatch SGD optimize it. For example, optimize:
$$
L^{\text{KLPEN}} (\theta) = \mathop{\mathbb{\hat E}}_t  \left[ r_t(\theta) \hat A_t  - \beta \text{KL}[ \pi_{\theta \text{old}} (\cdot| s_t), \pi_{\theta} (\cdot| s_t)]    \right]
$$

And after each iteration compute $d =  \text{KL}[ \pi_{\theta \text{old}} (\cdot| s_t), \pi_{\theta} (\cdot| s_t)]$. And compare $d$ with some arbitrary $d_{targ}$
- If $d < d_{targ} / 1.5, \beta \lArr \beta/2$ 
- If $d > d_{targ} \times  1.5, \beta \lArr \beta \times 2$ 
  
The updated $\beta$ is used for the next policy update.  

> The parameters 1.5 and 2 above are chosen heuristically, but the algorithm is not very sensitive to them.  



This approach hasn't get better results than the clipped surrogate objective.  
> In our experiments, we found that the KL penalty performed worse than the clipped surrogate objective, however, we’ve included it here because it’s an important baseline.

### Algorithm
> If using a neural network architecture that shares parameters between the policy and value function, we must use a loss function that combines the policy surrogate and a value function error term. This objective can further be augmented by adding an entropy bonus to ensure sufficient exploration.

$$
L_t^{\text{CLIP + VF + S}} (\theta) = \mathop{\mathbb{\hat E}}_t [L_t^{\text{CLIP}}(\theta) - c_1  L_t^{\text{VF}}(\theta) -c_2 S[\pi_\theta](s_t)]
$$

where $c1$, $c2$ are coefficients, and $S$ denotes an entropy bonus, and $L_t^{\text{VF}}$ is a squared-error loss $(V_\theta(s_t) − V_t^{\text{targ}}(s_t))^2$.

> One style of policy gradient implementation, popularized in and well-suited for use with recurrent neural networks, runs the policy for T timesteps (where T is much less than the episode length), and uses the collected samples for an update. This style requires an advantage estimator that does not look beyond timestep T.
> 
It can be, for example:

$$
\hat A_t = - V(s_t) + r_t +\gamma r_{t+1} + ... + \gamma^{T-t+1} r_{T-1} + \gamma^{T-t}V(S_T)
$$

where $t$ specifies the time index in $[0, T]$.


A proximal policy optimization (PPO) algorithm that uses fixed-length trajectory segments is
shown below.

> Each iteration, each of N (parallel) actors collect T timesteps of data. Then we construct the surrogate loss on these NT timesteps of data, and optimize it with minibatch SGD (or usually for better performance, Adam), for K epochs.

![Algorithm](ppo_algo.PNG)

### Experiments



## Related works
1. [OpenAI Algorithm implementations](https://github.com/openai/baselines)
2. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
3. [Sample Efficient Actor-Critic with Experience Replay - ACER](https://arxiv.org/abs/1611.01224)


## Questions
1. They proposed objective but what with the derivatives? How to get gradient of the clip function (make some intervals?)?
2. Fig. 2 explanation
3. Why KL divergence is bad?
4. Is the PPO a safe option to prevent gradient overshooting by the cost of performing more steps?
5. Purpose of using $\lambda$ in the eq. 11? 


## [Comment] Call for help at the end of blog post  
>We’re looking for people to help build and optimize our reinforcement learning algorithm codebase. If you’re excited about RL, benchmarking, thorough experimentation, and open source, please apply, and mention that you read the baselines PPO post in your application.

We are going to do it anyway and we could have track of the most popular algorithms.
