# Trust Region Policy Optimization, Berkeley, 2017
Arxiv: https://arxiv.org/abs/1502.05477  
Blog: https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9

## Overview
TRPO is constructed on:
1. Minimizing surogate function that guarantees policy improvement.
2. Approximating some of components of the surogate function (better efficiency)
3. Using two approaches: one for model free and one for simulation environments.


> [...] we first prove that minimizing a certain surrogate 
> objective function guarantees policy improvement
> with non-trivial step sizes. Then we make a series of approximations
> to the theoretically-justified algorithm, yielding
> a practical algorithm, which we call trust region policy
> optimization (TRPO). We describe two variants of this
> algorithm: first, the single-path method, which can be applied
> in the model-free setting; second, the vine method,
> which requires the system to be restored to particular states,
> which is typically only possible in simulation.


## Detailed analysis

### Preliminaries

Expected discounted reward when using policy $\pi$ ( $a_t \sim \pi(a_t|s_t)$ )
$$
\eta(\pi) = \mathop{\mathbb{E}}_{\pi} [\sum^\infin_{t=0}\gamma^t r(s_t)]
$$

Expected return of another policy $\tilde \pi$ in terms of the advantage over policy $\pi$

$$
\eta(\tilde \pi) = \eta(\pi) + \mathop{\mathbb{E}}_{\tilde \pi} [\sum^\infin_{t=0}\gamma^t A_\pi(a_t, s_t)]
$$




> [...] the update performed by exact
> policy iteration [...] improves the policy if there is
> at least one state-action pair with a positive advantage value
> and nonzero state visitation probability, otherwise the algorithm
> has converged to the optimal policy. However, in the
> approximate setting, it will typically be unavoidable, due
> to estimation and approximation error, that there will be
> some states s for which the expected advantage is negative,
> that is, $\sum_a \tilde{\pi} (a|s) A_\pi(s,a,) > 0$
> 
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

Conservative policy iteration update and its lower bound for updating the policy
$$
\pi_{\text{new}}(a|s) = (1 - \alpha) \pi_{\text{old}}(a|s) +\alpha \pi'(a|s)
$$

where $\pi'(a|s)= \text{argmax}_{\pi'}L_{\pi_{\text{old}}}(\pi')$

$$
\eta(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{2\epsilon\gamma}{(1 - \gamma) ^ 2} \alpha^2
$$
where $\epsilon = \underset{s}{\text{max}}|\mathop{\mathbb{E}}_{a \sim \pi'(a|s)}  [A_\pi (s, a)]|$



### Monotonic Improvement Guarantee for General Stochastic Policies
If we will replace $\alpha$ with distance measure between policies instead of using mixture policies,
we extend this to the __general stochastic policies__.

First example of distance measure is the total variation divergence
$$
D_{TV}(p||q) = \frac{1}{2}\sum_i|p_i-q_i|
$$
where $q$ and $p$ are discrete probability distributions.

Define $D_{TV}^{\text{max}}$ as

$$
D_{TV}^{\text{max}}(\pi, \tilde\pi) = \underset{s}{\text{max}} \ D_{TV}(\pi(\cdot | s)||\tilde \pi(\cdot | s))
$$

If $\alpha = D_{TV}^{\text{max}}(\pi_{\text{old}}, \pi_{\text{new}})$ then the lower bound is

$$
\eta(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1 - \gamma) ^ 2} \alpha^2
$$
where $\epsilon = \underset{s, a}{\text{max}}|A_\pi (s, a)|$

The relationship between total variation divergence and the KL divergence

$$
D_{TV}(p||q)^2 \leq D_{\text{KL}}(p||q)
$$

Let 
$$
D_{\text{KL}}^{\text{max}}(\pi, \tilde\pi) = \underset{s}{\text{max}} \ D_{\text{KL}}(\pi(\cdot | s)||\tilde \pi(\cdot | s))
$$

So the lower bound will be

$$
\eta(\tilde\pi) \geq L_{\pi}(\tilde\pi) - C D_{\text{KL}}^{\text{max}} (\pi, \tilde\pi)
$$
where $C = \frac{4\epsilon\gamma}{(1 - \gamma) ^ 2}$

In the paper there is a proof that making updates this way guaranteeing nondecreasing 
expected return $\eta$.

![Algorithm 1](trpo_algo_1.PNG)


### Optimization of Parameterized Policies
> In the previous section, we considered the policy optimization
> problem independently of the parameterization of $\pi$
> and under the assumption that the policy can be evaluated
> at all states.

__Therefore we will change notation to express dependency on the policy parameters $\theta$.__

By performing following maximization, we are guaranteed to imporve the true objective $\eta$

$$
\underset{\theta}{\text{maximize}}[L_{\theta{\text{old}}}(\theta) - C D_{\text{KL}}^{\text{max}}(\theta_{\text{old}}, \theta)]
$$

> In practice, if we used the penalty coefficient C recommended
> by the theory above, the step sizes would be very
> small. One way to take larger steps in a robust way is to use
> a constraint on the KL divergence between the new policy
> and the old policy, i.e., a trust region constraint:

$$
\underset{\theta}{\text{maximize }}L_{\theta{\text{old}}}(\theta)
$$
$$
\text{suject to } D_{\text{KL}}^{\text{max}}(\theta_{\text{old}}, \theta) \leq \delta
$$

Because it is bounding KL divergence at every point in the space it is impractical to solve and we use heuristic approximation to the $D_{\text{KL}}^{\text{max}}$ by:

$$
\overline{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}} (\theta_1, \theta_2) := \mathop{\mathbb{E}}_{s \sim \rho} [D_{\text{KL}}(\pi_{\theta_1}(\cdot | s)||\pi_{\theta_2}(\cdot | s))]
$$

and optimize

$$
\underset{\theta}{\text{maximize }}L_{\theta{\text{old}}}(\theta)
$$
$$
\text{suject to } \overline{D}_{\text{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta
$$



## Related works
Links to the related works
1. [A Natural Policy Gradient, UCL 2002](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)
2. [A Tutorial on MM Algorithms,  The American Statistician 2004](https://amstat.tandfonline.com/doi/abs/10.1198/0003130042836#.XYNPDigzbZk)



## Questions
1. What are general derivative-free stochastic optimization methods such as CEM and CMA?
2. Why we need to replace $\alpha$ with distance measure between policies? Section 3.  
   Is this just simplifing the $\epsilon$?
3. Why to change from the total variation divergence to the KL divergence?


## Notes

According to OpenAI blog [post](https://openai.com/blog/openai-baselines-ppo/):  
  >TRPO isn’t easily compatible with algorithms that share parameters between a policy and value function or auxiliary losses, like those used to solve problems in Atari and other domains where the visual input is significant