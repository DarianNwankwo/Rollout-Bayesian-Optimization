include("radial_basis_surrogates.jl")

using SharedArrays

""" 
Abstract type `Observable` to represent observables in the system. 
Subtypes will implement specific types of observables.
"""
abstract type Observable end

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
mutable struct StochasticObservable <: Observable
    fs::SmartFantasyRBFsurrogate
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
mutable struct DeterministicObservable <: Observable
    testfn::TestFunction
    trajectory_length::Int64
    step::Int64
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function DeterministicObservable(t::TestFunction; max_invocations)
        dim = testfn.dim
        observations = zeros(max_invocations)
        gradients = zeros(dim, max_invocations)
        return new(t, max_invocations, 0, observations, gradients)
    end
end

""" 
Call operator for `DeterministicObservable` which produces an observation 
and its corresponding gradient for a given input vector `x`.

# Arguments:
- `x::AbstractVector`: The input vector for which the observation and gradient are generated.

# Returns:
- `observation`: The generated observation for the input `x`.

"""
function (deo::DeterministicObservable)(x::AbstractVector)::Number
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation, gradient_ = testfn(x), gradient(testfn)(x)
    deo.step += 1
    deo.observations[deo.step] = observation
    deo.gradients[:, deo.step] = gradient_

    return observation
end

"""
Abstract type `Trajectory` to represent different types of trajectories in the system. 
Subtypes will implement specific types of trajectories.
"""
abstract type Trajectory end

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
mutable struct ForwardTrajectory <: Trajectory
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
mutable struct AdjointTrajectory <: Trajectory
    s::RBFsurrogate
    fs::SmartFantasyRBFsurrogate
    fmin::Float64
    x0::Vector{Float64}
    h::Int
    observable::Union{Nothing, Observable}
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
function ForwardTrajectory(; base_surrogate::RBFsurrogate, start::Vector{Float64}, horizon::Int)
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
function AdjointTrajectory(; base_surrogate::Surrogate, start::AbstractVector, horizon::Int)
    fmin = minimum(get_observations(base_surrogate))
    d, N = size(get_covariates(base_surrogate))
    fsur = fit_sfsurrogate(base_surrogate, horizon)
    observable = nothing
    return AdjointTrajectory(base_surrogate, fsur, fmin, start, horizon, observable)
end

"""
Attach an observable to an `AdjointTrajectory`.

# Arguments:
- `AT::AdjointTrajectory`: The adjoint trajectory to which the observable will be attached.
- `observable::Observable`: The observable to attach.

# Returns:
- `Observable`: The attached observable.
"""
attach_observable!(AT::AdjointTrajectory, observable::Observable) = AT.observable = observable

"""
Get the starting point of the `AdjointTrajectory`.

# Arguments:
- `T::AdjointTrajectory`: The adjoint trajectory from which to retrieve the starting point.

# Returns:
- `Vector{Float64}`: The starting point of the trajectory.
"""
starting_point(T::AdjointTrajectory) = T.x0


"""
Retrieve the observable associated with an `AdjointTrajectory`.

# Arguments:
- `T::AdjointTrajectory`: The adjoint trajectory from which to retrieve the observable.

# Returns:
- `Union{Nothing, Observable}`: The associated observable, or `nothing` if none is attached.
"""
function get_observable(T::AdjointTrajectory)
    return T.observable
end

get_base_surrogate(T::Trajectory) = T.s

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.

- TODO: Fix the logic associated with maintaining the minimum found along the sample path vs.
that of the minimum from the best value known from the known locations.
"""
Base.@kwdef mutable struct TrajectoryParameters
    x0::Vector{Float64}
    h::Int
    mc_iters::Int
    rnstream_sequence::Array{Float64, 3}
    lbs::Vector{Float64}
    ubs::Vector{Float64}

    function TrajectoryParameters(
        x0::Vector{Float64},
        h::Int,
        mc_iters::Int,
        rnstream::Array{Float64, 3},
        lbs::Vector{Float64},
        ubs::Vector{Float64}
    )
        function check_dimensions(x0::Vector{Float64}, lbs::Vector{Float64}, ubs::Vector{Float64})
            n = length(x0)
            @assert length(lbs) == n && length(ubs) == n "Lower and upper bounds must be the same length as the initial point"
        end

        function check_stream_dimensions(rnstream_sequence::Array{Float64, 3}, d::Int, h::Int, mc_iters::Int)
            n_rows, n_cols = size(rnstream_sequence[1, :, :])
            @assert n_rows == d + 1 && n_cols <= h + 1 "Random number stream must have d + 1 rows and h + 1 columns for each sample"
            @assert size(rnstream_sequence, 1) == mc_iters "Random number stream must have at least mc_iters ($mc_iters) samples"
        end

        check_dimensions(x0, lbs, ubs)
        # check_stream_dimensions(rnstream_sequence, length(x0), h, mc_iters)
    
        return new(x0, h, mc_iters, rnstream, lbs, ubs)
    end
end

function initialize_trajectory_parameters(;
    start::AbstractVector,
    horizon::Int,
    mc_iterations::Int,
    use_low_discrepancy_sequence::Bool,
    lowerbounds::AbstractVector,
    upperbounds::AbstractVector)
    if use_low_discrepancy_sequence
        rns = gen_low_discrepancy_sequence(mc_iterations, length(lowerbounds), horizon + 1)
    else
        rns = randn(mc_iterations, length(lowerbounds) + 1, horizon + 1)
    end

    return TrajectoryParameters(start, horizon, mc_iterations, rns, lowerbounds, upperbounds)
end

get_bounds(tp::TrajectoryParameters) = (tp.lbs, tp.ubs)
each_trajectory(tp::TrajectoryParameters) = 1:tp.mc_iters
get_samples_rnstream(tp::TrajectoryParameters; sample_index) = tp.rnstream_sequence[sample_index, :, :]
starting_point(tp::TrajectoryParameters) = tp.x0