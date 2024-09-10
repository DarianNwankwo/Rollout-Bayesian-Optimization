include("radial_basis_surrogates.jl")

using SharedArrays

"""
    AbstractObservable

Abstract type to represent the mechanism for observing values along a trajectory.
"""
abstract type AbstractObservable end

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
mutable struct StochasticObservable <: AbstractObservable
    fs::AbstractFantasySurrogate
    stdnormal::AbstractMatrix
    trajectory_length::Int64
    step::Int64
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function StochasticObservable(; surrogate, stdnormal, max_invocations)
        dim = size(stdnormal, 1) - 1
        invocations = size(stdnormal, 2)
        observations = zeros(invocations)
        gradients = zeros(dim, invocations)
        return new(surrogate, stdnormal, max_invocations, 0, observations, gradients)
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
function (so::StochasticObservable)(x::AbstractVector)::Number
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, stdnormal=so.stdnormal[:, so.step + 1], fantasy_index=so.step - 1, with_gradient=true
    )
    so.step += 1
    so.observations[so.step] = observation
    so.gradients[:, so.step] = gradient_

    return observation
end

function (so::StochasticObservable)(x::AbstractVector, θ::AbstractVector)::Number
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, θ, stdnormal=so.stdnormal[:, so.step + 1], fantasy_index=so.step - 1, with_gradient=true
    )
    so.step += 1
    so.observations[so.step] = observation
    so.gradients[:, so.step] = gradient_

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
    f::Function
    ∇f::Function
    trajectory_length::Integer
    step::Integer
    observations::AbstractVector{<:Real}
    gradients::AbstractMatrix{<:Real}

    function DeterministicObservable(; func::Function, gradient::Function, max_invocations::Integer)
        dim = testfn.dim
        observations = zeros(max_invocations)
        gradients = zeros(dim, max_invocations)
        return new(func, gradient, max_invocations, 0, observations, gradients)
    end
end

eval(deo::DeterministicObservable) = deo.f
gradient(deo::DeterministicObservable) = deo.∇f

function update!(deo::DeterministicObservable, y::Real, ∇y::AbstractVector)
    deo.observations[deo.step] = y
    deo.gradients[:, deo.step] = ∇y
end

function increment!(deo::DeterministicObservable)
    deo.step += 1
    return nothing
end

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
function (deo::DeterministicObservable)(x::AbstractVector, θ::AbstractVector)::Number
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation = eval(deo)(x)
    gradient_ = gradient(deo)(x)
    increment!(deo)
    update!(deo, observation, gradient_)
    return observation
end

"""
    AbstractTrajectory

Abstract type defining a trajectory to be simulated given some base policy,
starting location, horizon and a mechanism for observing sample values along the
trajectory.
"""
abstract type AbstractTrajectory end

"""
A mutable struct `ForwardTrajectory` that represents a forward trajectory in the system.

# Fields:
- `s::RBFsurrogate`: The RBF surrogate model used in the trajectory.
- `fs::FantasyRBFsurrogate`: The fantasy RBF surrogate model used to generate the trajectory.
- `mfs::MultiOutputFantasyRBFsurrogate`: The multi-output fantasy RBF surrogate model used in the trajectory.
- `jacobians::Vector{Matrix{Float64}}`: A vector of Jacobian matrices associated with the trajectory.
- `fmin::Float64`: The minimum function value observed.
- `x0::Vector{Float64}`: The starting point of the trajectory.
- `h::Int`: The number of steps (or horizon) for the trajectory.

# Constructor:
- `ForwardTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)`: Creates a new instance of `ForwardTrajectory` by fitting the necessary surrogate models and initializing the trajectory.
"""
mutable struct ForwardTrajectory <: AbstractTrajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    mfs::MultiOutputFantasyRBFsurrogate
    jacobians::Vector{Matrix{Float64}}
    fmin::Float64
    x0::Vector{Float64}
    h::Int
end

"""
A mutable struct `AdjointTrajectory` that represents an adjoint trajectory in the system.

# Fields:
- `s::RBFsurrogate`: The RBF surrogate model used in the trajectory.
- `fs::SmartFantasyRBFsurrogate`: The smart fantasy RBF surrogate model used in the trajectory.
- `fmin::Float64`: The minimum function value observed.
- `x0::Vector{Float64}`: The starting point of the trajectory.
- `h::Int`: The number of steps (or horizon) for the trajectory.
- `observable::Union{Nothing, Observable}`: An observable associated with the trajectory, if any.

# Constructor:
- `AdjointTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)`: Creates a new instance of `AdjointTrajectory` by fitting the necessary surrogate models and initializing the trajectory.
"""
mutable struct AdjointTrajectory <: AbstractTrajectory
    s::AbstractSurrogate
    fs::AbstractFantasySurrogate
    fmin::Real
    x0::AbstractVector
    θ::AbstractVector
    h::Int
    observable::Union{Missing, AbstractObservable}
    cost::AbstractCostFunction
end

"""
Constructor for `ForwardTrajectory`.

# Arguments:
- `s::RBFsurrogate`: The RBF surrogate model to be used in the trajectory.
- `x0::Vector{Float64}`: The starting point of the trajectory.
- `h::Int`: The number of steps (or horizon) for the trajectory.

# Returns:
- `ForwardTrajectory`: A new instance of `ForwardTrajectory` with initialized fields.
"""
function ForwardTrajectory(; base_surrogate::AbstractSurrogate, start::AbstractVector, horizon::Integer)
    fmin = minimum(get_observations(base_surrogate))
    d, N = size(get_covariates(base_surrogate))

    ∇ys = [zeros(d) for i in 1:N]

    fsur = fit_fsurrogate(base_surrogate, horizon)
    mfsur = fit_multioutput_fsurrogate(base_surrogate, horizon)

    jacobians = [I(d)]

    return ForwardTrajectory(base_surrogate, fsur, mfsur, jacobians, fmin, start, horizon)
end


"""
Create an `AdjointTrajectory` object using a base surrogate model, a starting point, and a specified time horizon.

# Arguments
- `base_surrogate::Surrogate`: The base surrogate model used to generate the adjoint trajectory.
- `start::AbstractVector`: The initial starting point for the trajectory.
- `horizon::Int`: The time horizon or number of steps for the trajectory.

# Returns
- `AdjointTrajectory`: An instance of `AdjointTrajectory` initialized with the provided surrogate model, starting point, and time horizon.

# Details
- The function first computes the minimum value (`fmin`) of the observations from the `base_surrogate`.
- It then determines the dimensionality (`d`) and the number of observations (`N`) from the covariates of the `base_surrogate`.
- A smart fantasy surrogate (`fsur`) is fitted based on the `base_surrogate` and the given `horizon`.
- The `observable` is initialized as `nothing` and should be set later.
"""
function AdjointTrajectory(;
    base_surrogate::AbstractSurrogate,
    start::AbstractVector,
    hypers::AbstractVector,
    horizon::Integer,
    cost::AbstractCostFunction = UniformCost())
    fmin = minimum(get_observations(base_surrogate))
    d, N = size(get_covariates(base_surrogate))
    fsur = FantasySurrogate(base_surrogate, horizon)
    observable = missing
    return AdjointTrajectory(base_surrogate, fsur, fmin, start, hypers, horizon, observable, cost)
end

"""
Attach an observable to an `AdjointTrajectory`.

The AdjointTrajectory needs to be created first. The observable expects a mechanism for
fantasized samples, which is created once the AdjointTrajectory struct is created. We then
attach the observable after the fact.
"""
attach_observable!(AT::AdjointTrajectory, observable::AbstractObservable) = AT.observable = observable
get_observable(T::AdjointTrajectory) = T.observable
get_starting_point(T::AdjointTrajectory) = T.x0
get_base_surrogate(T::AdjointTrajectory) = T.s
get_fantasy_surrogate(T::AdjointTrajectory) = T.fs
get_cost_function(T::AdjointTrajectory) = T.cost
get_hyperparameters(T::AdjointTrajectory) = T.θ
get_horizon(T::AdjointTrajectory) = T.horizon
get_minimum(T::AdjointTrajectory) = T.fmin

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.

- TODO: Fix the logic associated with maintaining the minimum found along the sample path vs.
that of the minimum from the best value known from the known locations.
"""
Base.@kwdef mutable struct TrajectoryParameters
    x0::AbstractVector
    horizon::Integer
    mc_iters::Integer
    rnstream_sequence::AbstractArray{<:Real, 3}
    lbs::AbstractVector
    ubs::AbstractVector
    θ::AbstractVector

    function TrajectoryParameters(x0, horizon, mc_iters, rnstream, lbs, ubs, θ)
        function check_dimensions(x0, lbs, ubs)
            n = length(x0)
            @assert length(lbs) == n && length(ubs) == n "Lower and upper bounds must be the same length as the initial point"
        end

        function check_stream_dimensions(rnstream_sequence, dim, horizon, mc_iters)
            n_rows, n_cols = size(rnstream_sequence[1, :, :])
            @assert n_rows == dim + 1 && n_cols <= horizon + 1 "Random number stream must have d + 1 rows and h + 1 columns for each sample"
            @assert size(rnstream_sequence, 1) == mc_iters "Random number stream must have at least mc_iters ($mc_iters) samples"
        end

        check_dimensions(x0, lbs, ubs)
        check_stream_dimensions(rnstream, length(x0), horizon, mc_iters)
    
        return new(x0, horizon, mc_iters, rnstream, lbs, ubs, θ)
    end
end

function TrajectoryParameters(;
    start::AbstractVector,
    hypers::AbstractVector,
    horizon::Integer,
    mc_iterations::Integer,
    use_low_discrepancy_sequence::Bool,
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector)
    if use_low_discrepancy_sequence
        rns = gen_low_discrepancy_sequence(mc_iterations, length(lowerbounds), horizon + 1)
    else
        rns = randn(mc_iterations, length(lowerbounds) + 1, horizon + 1)
    end

    return TrajectoryParameters(start, horizon, mc_iterations, rns, lowerbounds, upperbounds, hypers)
end

get_bounds(tp::TrajectoryParameters) = (tp.lbs, tp.ubs)
each_trajectory(tp::TrajectoryParameters) = 1:tp.mc_iters
get_samples_rnstream(tp::TrajectoryParameters; sample_index) = tp.rnstream_sequence[sample_index, :, :]
get_starting_point(tp::TrajectoryParameters) = tp.x0
get_hyperparameters(tp::TrajectoryParameters) = tp.θ
get_horizon(tp::TrajectoryParameters) = tp.horizon
set_starting_point!(tp::TrajectoryParameters, x::AbstractVector) = tp.x0[:] = x


"""
A convenient wrapper for the expected outcome of simulating a full trajectory.
"""
struct ExpectedTrajectoryOutput
    μxθ::Real
    σ_μxθ::Real
    ∇μx::Union{AbstractVector{<:Real}, Nothing}
    σ_∇μx::Union{AbstractVector{<:Real}, Nothing}
    ∇μθ::Union{AbstractVector{<:Real}, Nothing}
    σ_∇μθ::Union{AbstractVector{<:Real}, Nothing}
end

function ExpectedTrajectoryOutput(;
    μxθ,
    σ_μxθ,
    ∇μx=nothing,
    σ_∇μx=nothing,
    ∇μθ=nothing,
    σ_∇μθ=nothing)
    return ExpectedTrajectoryOutput(μxθ, σ_μxθ, ∇μx, σ_∇μx, ∇μθ, σ_∇μθ)
end

mean(eto::ExpectedTrajectoryOutput) = eto.μxθ
Distributions.std(eto::ExpectedTrajectoryOutput) = eto.σ_μxθ
gradient(eto::ExpectedTrajectoryOutput; wrt_hypers::Bool = false) = wrt_hypers ? eto.∇μθ : eto.∇μx
std_gradient(eto::ExpectedTrajectoryOutput; wrt_hypers::Bool = false) = wrt_hypers ? eto.σ_∇μθ : eto.σ_∇μx