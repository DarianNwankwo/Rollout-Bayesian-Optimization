# Differentiating Policies for Non-Myopic Bayesian Optimization
## Abstract
Bayesian optimization (BO) methods choose sample points by optimizing an acquisition function derived from 
a statistical model of the objective.  These acquisition functions are chosen to balance sampling regions with
predicted good objective values against exploring regions where the objective is uncertain.
Standard acquisition functions are myopic, considering only the impact of the next sample,
but non-myopic acquisition functions may be more effective.
In principle, one could model the sampling by a Markov decision process, and optimally choose the next sample
by maximizing an expected reward computed by dynamic programming; however, this is infeasibly expensive.
More practical approaches, such as rollout, consider a parametric family of sampling policies.
In this paper, we show how to efficiently estimate rollout acquisition functions and their gradients,
enabling stochastic gradient-based optimization of sampling policies.

## Software Design
In order to enable our computations of interest, we expressed each core computational concept as its own
abstract type and built out our implementations from there. We first begin by enumerating the fundamental
abstract types that appear throughout our framework. The depiction below follows julia's type system, that is, the root node represents a concrete type--all other nodes are abstract:
* AbstractKernel
    * StationaryKernel
        * RadialBasisFunction
* AbstractCostFunction
    * KnownCostFunction
        * UniformCost
        * NonUniformCost
    * UnknownCostFunction (can be unknown and deterministic; fix)
        * GaussianProcessCost
* AbstractSurrogate
    * Surrogate
    * AbstractFantasySurrogate
        * FantasySurrogate
        * AbstractPerturbationSurrogate
            * SpatialPerturbationSurrogate
            * DataPerturbationSurrogate
* AbstractObservable
    * DeterministicObservable
    * StochasticObservable
* AbstractTrajectory (fix typing on concrete types)
    * ForwardTrajectory
    * AdjointTrajectory
* AbstractPolicy
    * BasePolicy
