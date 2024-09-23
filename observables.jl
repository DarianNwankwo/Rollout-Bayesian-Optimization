"""
    AbstractObservable

Abstract type to represent the mechanism for observing values along a trajectory.
"""
abstract type AbstractObservable end

function update!(o::AbstractObservable, y::Real, ∇y::AbstractVector)
    o.observations[o.step] = y
    o.gradients[:, o.step] = ∇y
end

function increment!(o::AbstractObservable)
    o.step += 1
    return nothing
end

""" 
A mutable struct `StochasticObservable` that represents an observable 
which produces stochastic observations and their gradients.
    
# Fields:
- `fs::SmartFantasyRBFsurrogate`: The surrogate model used to generate observations.
- `stdnormal::AbstractMatrix`: A matrix of standard normal variables used for generating the stochastic observations.
- `trajectory_length::Int64`: The maximum number of steps (or invocations) allowed.
- `step::Int64`: The current step or invocation count.
- `observations::Vector{Float64}`: A vector to store the observations generated.
- `gradients::Matrix{Float64}`: A matrix to store the gradients corresponding to the observations.

# Constructor:
- `StochasticObservable(fs, stdnormal, trajectory_length)`: Creates a new instance of `StochasticObservable` with the given surrogate model, standard normal variables, and trajectory length.

"""
mutable struct StochasticObservable{FS <: AbstractFantasySurrogate} <: AbstractObservable
    fs::FS
    stdnormal::Matrix{Float64}
    trajectory_length::Int
    step::Int
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function StochasticObservable(; surrogate, stdnormal, max_invocations)
        dim = size(stdnormal, 1) - 1
        invocations = size(stdnormal, 2)
        observations = zeros(invocations)
        gradients = zeros(dim, invocations)
        return new{typeof(surrogate)}(surrogate, stdnormal, max_invocations, 0, observations, gradients)
    end
end

""" 
Call operator for `StochasticObservable` which produces an observation and 
its corresponding gradient for a given input vector `x`.
    
# Arguments:
- `x::AbstractVector`: The input vector for which the observation and gradient are generated.

# Returns:
- `observation`: The generated observation for the input `x`.

"""
function (so::StochasticObservable{FS})(x::Vector{T}, θ::Vector{T})::Number where {T <: Real, FS <: FantasySurrogate}
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, θ, stdnormal=so.stdnormal[:, so.step + 1], fantasy_index=so.step - 1, with_gradient=true
    )
    increment!(so)
    update!(so, observation, gradient_)

    return observation
end

function (so::StochasticObservable{FS})(x::Vector{T})::Number where {T <: Real, FS <: FantasyRBFsurrogate}
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, stdnormal=so.stdnormal[:, so.step + 1], with_gradient=true
    )
    increment!(so)
    update!(so, observation, gradient_)

    return observation
end

""" 
A mutable struct `DeterministicObservable` that represents an observable 
which produces deterministic observations and their gradients.

# Fields:
- `testfn::TestFunction`: The test function used to generate observations.
- `trajectory_length::Int64`: The maximum number of steps (or invocations) allowed.
- `step::Int64`: The current step or invocation count.
- `observations::Vector{Float64}`: A vector to store the observations generated.
- `gradients::Matrix{Float64}`: A matrix to store the gradients corresponding to the observations.

# Constructor:
- `DeterministicObservable(testfn, trajectory_length)`: Creates a new instance of `DeterministicObservable` with the given test function and trajectory length.

"""
mutable struct DeterministicObservable <: AbstractObservable
    f
    ∇f
    trajectory_length::Int
    step::Int
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function DeterministicObservable(; func::Function, gradient::Function, max_invocations::Int)
        dim = testfn.dim
        observations = zeros(max_invocations)
        gradients = zeros(dim, max_invocations)
        return new(func, gradient, max_invocations, 0, observations, gradients)
    end
end

eval(o::DeterministicObservable) = o.f
gradient(o::DeterministicObservable) = o.∇f

""" 
Call operator for `DeterministicObservable` which produces an observation 
and its corresponding gradient for a given input vector `x`. The second argument
is unused to fit our representation for our evaluating our acquisition function
in terms of spatial coordinates and hyperparameters.

# Arguments:
- `x::AbstractVector`: The input vector for which the observation and gradient are generated.

# Returns:
- `observation`: The generated observation for the input `x`.

"""
function (deo::DeterministicObservable)(x::Vector{T}, θ::Vector{T})::Number where T <: Real
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation = eval(deo)(x)
    gradient_ = gradient(deo)(x)
    increment!(deo)
    update!(deo, observation, gradient_)
    return observation
end

function (deo::DeterministicObservable)(x::Vector{T})::Number where T <: Real
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation = eval(deo)(x)
    gradient_ = gradient(deo)(x)
    increment!(deo)
    update!(deo, observation, gradient_)
    return observation
end

get_gradient(o::AbstractObservable; at::Int) = o.gradients[:, at]