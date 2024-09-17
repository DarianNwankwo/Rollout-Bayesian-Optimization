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
    jacobians::Vector{AbstractMatrix{<:Real}}
    fmin::Real
    x0::AbstractVector
    h::Integer
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

    jacobians = [Matrix{Float64}(I(d))]

    return ForwardTrajectory(base_surrogate, fsur, mfsur, jacobians, fmin, start, horizon)
end


get_starting_point(T::ForwardTrajectory) = T.x0
get_base_surrogate(T::ForwardTrajectory) = T.s
get_fantasy_surrogate(T::ForwardTrajectory) = T.fs
get_horizon(T::ForwardTrajectory) = T.h
get_minimum(T::ForwardTrajectory) = T.fmin

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
    h::Integer
    observable::Union{Missing, AbstractObservable}
    cost::AbstractCostFunction
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
    spatial_lbs::AbstractVector
    spatial_ubs::AbstractVector
    θ::AbstractVector

    function TrajectoryParameters(x0, horizon, mc_iters, rnstream, slbs, subs, θ)
        function check_dimensions(x0, lbs, ubs)
            n = length(x0)
            @assert length(lbs) == n && length(ubs) == n "Lower and upper bounds must be the same length as the initial point"
        end

        function check_stream_dimensions(rnstream_sequence, dim, horizon, mc_iters)
            n_rows, n_cols = size(rnstream_sequence[1, :, :])
            @assert n_rows == dim + 1 && n_cols <= horizon + 1 "Random number stream must have d + 1 rows and h + 1 columns for each sample"
            @assert size(rnstream_sequence, 1) == mc_iters "Random number stream must have at least mc_iters ($mc_iters) samples"
        end

        check_dimensions(x0, slbs, subs)
        check_stream_dimensions(rnstream, length(x0), horizon, mc_iters)
    
        return new(x0, horizon, mc_iters, rnstream, slbs, subs, θ)
    end
end

function TrajectoryParameters(;
    start::AbstractVector,
    hypers::AbstractVector,
    horizon::Integer,
    mc_iterations::Integer,
    use_low_discrepancy_sequence::Bool,
    spatial_lowerbounds::AbstractVector,
    spatial_upperbounds::AbstractVector)
    if use_low_discrepancy_sequence
        rns = gen_low_discrepancy_sequence(mc_iterations, length(spatial_lowerbounds), horizon + 1)
    else
        rns = randn(mc_iterations, length(spatial_lowerbounds) + 1, horizon + 1)
    end

    return TrajectoryParameters(
        start,
        horizon,
        mc_iterations,
        rns,
        spatial_lowerbounds,
        spatial_upperbounds,
        hypers
    )
end

get_spatial_bounds(tp::TrajectoryParameters) = (tp.spatial_lbs, tp.spatial_ubs)
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