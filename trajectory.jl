"""
    AbstractTrajectory

Abstract type defining a trajectory to be simulated given some base policy,
starting location, horizon and a mechanism for observing sample values along the
trajectory.
"""
abstract type AbstractTrajectory end

get_starting_point(T::AbstractTrajectory) = T.x0
get_base_surrogate(T::AbstractTrajectory) = T.s
get_fantasy_surrogate(T::AbstractTrajectory) = T.fs
get_horizon(T::AbstractTrajectory) = T.horizon
get_observable(T::AbstractTrajectory) = T.observable

mutable struct ForwardTrajectory <: AbstractTrajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    jacobians::Vector{Matrix{Float64}}
    x0::Vector{Float64}
    horizon::Int
    observable::Union{Missing, AbstractObservable}
end

function ForwardTrajectory(; base_surrogate::RBFsurrogate, start::Vector{T}, horizon::Int) where T <: Real
    d, N = size(get_covariates(base_surrogate))
    # fsur = fit_sfsurrogate(base_surrogate, horizon)
    fsur = fit_fsurrogate(base_surrogate, horizon)
    # Preallocate all space for each jacobian
    jacobians = [Matrix{Float64}(I(d))]
    for _ in 1:horizon
        push!(jacobians, zeros(d, d))
    end

    observable = missing
    
    return ForwardTrajectory(base_surrogate, fsur, jacobians, start, horizon, observable)
end

attach_observable!(FT::ForwardTrajectory, observable::AbstractObservable) = FT.observable = observable
get_observable(FT::ForwardTrajectory) = FT.observable
get_jacobian(FT::ForwardTrajectory; index::Int) = FT.jacobians[index]
set_jacobian!(FT::ForwardTrajectory; jacobian::Matrix{Float64}, index::Int) = FT.jacobians[index] = jacobian

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
mutable struct ForwardTrajectoryWithMOGP <: AbstractTrajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    mfs::MultiOutputFantasyRBFsurrogate
    jacobians::Vector{Matrix{Float64}}
    fmin::Float64
    x0::Vector{Float64}
    horizon::Int
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
function ForwardTrajectoryWithMOGP(; base_surrogate::AbstractSurrogate, start::AbstractVector, horizon::Integer)
    fmin = minimum(get_observations(base_surrogate))
    d, N = size(get_covariates(base_surrogate))

    ∇ys = [zeros(d) for i in 1:N]

    fsur = fit_fsurrogate(base_surrogate, horizon)
    mfsur = fit_multioutput_fsurrogate(base_surrogate, horizon)

    jacobians = [Matrix{Float64}(I(d))]

    return ForwardTrajectoryWithMOGP(base_surrogate, fsur, mfsur, jacobians, fmin, start, horizon)
end

get_minimum(T::ForwardTrajectoryWithMOGP) = T.fmin

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
    s::Surrogate
    fs::FantasySurrogate
    x0::Vector{Float64}
    θ::Vector{Float64}
    horizon::Int
    observable::Union{Missing, AbstractObservable}
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
    base_surrogate::Surrogate,
    start::Vector{T},
    hypers::Vector{T},
    horizon::Int) where T <: Real
    d, N = size(get_covariates(base_surrogate))
    fsur = FantasySurrogate(base_surrogate, horizon)

    observable = missing

    return AdjointTrajectory(base_surrogate, fsur, start, hypers, horizon, observable)
end

"""
Attach an observable to an `AdjointTrajectory`.

The AdjointTrajectory needs to be created first. The observable expects a mechanism for
fantasized samples, which is created once the AdjointTrajectory struct is created. We then
attach the observable after the fact.
"""
attach_observable!(AT::AdjointTrajectory, observable::AbstractObservable) = AT.observable = observable
get_observable(AT::AdjointTrajectory) = AT.observable
get_hyperparameters(T::AdjointTrajectory) = T.θ
get_minimum(T::AdjointTrajectory) = T.fmin

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.

- TODO: Fix the logic associated with maintaining the minimum found along the sample path vs.
that of the minimum from the best value known from the known locations.
"""
Base.@kwdef mutable struct TrajectoryParameters
    x0::Vector{Float64}
    horizon::Int
    mc_iters::Int
    rnstream_sequence::Array{Float64, 3}
    spatial_lbs::Vector{Float64}
    spatial_ubs::Vector{Float64}
    θ::Vector{Float64}

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
    start::Vector{T},
    hypers::Vector{T},
    horizon::Int,
    mc_iterations::Int,
    use_low_discrepancy_sequence::Bool,
    spatial_lowerbounds::Vector{T},
    spatial_upperbounds::Vector{T}) where T <: Real
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