"""
    AbstractObservable

Abstract type to represent the mechanism for observing values along a trajectory.
"""
abstract type AbstractObservable end

reset!(ao::AbstractObservable) = ao.step = 0
best_observation(ao::AbstractObservable) = first(findmin(ao.observations))
best_index(ao::AbstractObservable) = findmin(ao.observations)[2]

function resolve(ao::AbstractObservable; fmini::T) where T <: Real
    return max(fmini - best_observation(ao), 0.)
end

function update!(o::AbstractObservable, y::Real, ∇y::AbstractVector)
    o.observations[o.step] = y
    o.gradients[:, o.step] = ∇y
end

function increment!(o::AbstractObservable)
    o.step += 1
    return nothing
end

"""
Once our GaussHermiteObservable is used, we're going to compute the gradient and store the result. 
The weights for our quadrature are re-used across evaluations, we just need the updated surrogate.
So evaluating the gradient needs to remember what those weights are since the quadrature rule tells us
how much that weight contributes to the acquisition.
"""
mutable struct GaussHermiteObservable{FS <: AbstractFantasySurrogate} <: AbstractObservable
    fs::FS
    nodes::Vector{Float64}
    weights::Vector{Float64}
    trajectory_length::Int
    step::Int
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function GaussHermiteObservable(; fantasy_surrogate, nodes, weights, max_invocations)
        dim = size(get_covariates(fantasy_surrogate), 1)
        observations = zeros(max_invocations)
        gradients = zeros(dim, max_invocations)
        return new{typeof(fantasy_surrogate)}(fantasy_surrogate, nodes, weights, max_invocations, 0, observations, gradients)
    end
end

get_fantasy_surrogate(ao::AbstractObservable) = ao.fs
set_weights!(gho::GaussHermiteObservable, weights::Vector{T}) where T <: Real = gho.weights .= weights
get_weight(gho::GaussHermiteObservable; at::Int) = gho.weights[at]
set_nodes!(gho::GaussHermiteObservable, nodes::Vector{T}) where T <: Real = gho.nodes .= nodes

function (gho::GaussHermiteObservable{FS})(x::Vector{T}, θ::Vector{T}) where {T <: Real, FS <: FantasySurrogate}
    @assert gho.step < gho.trajectory_length "Maximum invocations have been used"
    fsx = get_fantasy_surrogate(gho)(x, θ, fantasy_index=gho.step - 1)
    observation = fsx.μ + sqrt(2) * fsx.σ * gho.nodes[gho.step + 1]
    gradient_ = fsx.∇μ + sqrt(2) * fsx.∇σ * gho.nodes[gho.step + 1]

    increment!(gho)
    update!(gho, observation, gradient_)

    return observation
end

function resolve(gho::GaussHermiteObservable; fmini::T) where T <: Real
    normalization_constant = sqrt(pi)
    weight = get_weight(gho, at=best_index(gho))
    value = weight * max(fmini - best_observation(gho), 0.) / normalization_constant

    return value
end

function get_gradient(gho::GaussHermiteObservable; at::Int)
    normalization_constant = sqrt(pi)
    weight = get_weight(gho, at=at)
    gradient_sample = gho.gradients[:, at]
    dvalue = weight * gradient_sample / normalization_constant

    return dvalue
end

mutable struct StochasticObservable{FS <: AbstractFantasySurrogate} <: AbstractObservable
    fs::FS
    stdnormal::Matrix{Float64}
    trajectory_length::Int
    step::Int
    observations::Vector{Float64}
    gradients::Matrix{Float64}
    expected_gradients::Matrix{Float64}

    function StochasticObservable(; fantasy_surrogate, stdnormal, max_invocations)
        dim = size(stdnormal, 1) - 1
        invocations = size(stdnormal, 2)
        observations = zeros(invocations)
        gradients = zeros(dim, invocations)
        expected_gradients = zeros(dim, invocations)
        return new{typeof(fantasy_surrogate)}(
            fantasy_surrogate, stdnormal, max_invocations, 0, observations, gradients, expected_gradients
        )
    end
end

set_sequence!(so::StochasticObservable, rnstream::Matrix{T}) where T <: Real = so.stdnormal = rnstream

function (so::StochasticObservable{FS})(x::Vector{T}, θ::Vector{T})::Number where {T <: Real, FS <: FantasySurrogate}
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, θ, stdnormal=so.stdnormal[:, so.step + 1], fantasy_index=so.step - 1, with_gradient=true
    )
    # TODO: gradient_ = set the gradient to be the expected gradient here or add another field
    # to our observable that contains the gradients and expected_gradients to simplify our API
    fsx = so.fs(x, θ, fantasy_index=so.step - 1)
    so.expected_gradients[:, so.step + 1] = fsx.∇μ

    # Standard update
    increment!(so)
    update!(so, observation, gradient_)

    return observation
end

# get_gradient(so::StochasticObservable; at::Int) = so.expected_gradients[:, at]
get_gradient(so::StochasticObservable; at::Int) = so.gradients[:, at]

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

function (deo::DeterministicObservable)(x::Vector{T}, θ::Vector{T})::Number where T <: Real
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation = eval(deo)(x)
    gradient_ = gradient(deo)(x)
    increment!(deo)
    update!(deo, observation, gradient_)
    return observation
end

get_gradient(o::AbstractObservable; at::Int) = o.gradients[:, at]
get_observation(o::AbstractObservable; at::Int) = o.observations[at]
get_observations(o::AbstractObservable) = o.observations
get_gradient(gho::GaussHermiteObservable; at::Int) = gho.weights[at] * gho.gradients[:, at]