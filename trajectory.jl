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
get_hyperparameters(T::AbstractTrajectory) = T.θ

mutable struct Trajectory <: AbstractTrajectory
    s::Surrogate
    fs::FantasySurrogate
    x0::Vector{Float64}
    θ::Vector{Float64}
    horizon::Int
    observable::Union{Missing, AbstractObservable}
end

function Trajectory(
    base_surrogate::Surrogate,
    fantasy_surrogate::FantasySurrogate;
    start::Vector{T},
    hypers::Vector{T},
    horizon::Int) where T <: Real
    d, N = size(get_covariates(base_surrogate))
    observable = missing
    return Trajectory(base_surrogate, fantasy_surrogate, start, hypers, horizon, observable)
end

"""
Attach an observable to an `AdjointTrajectory`.

The AdjointTrajectory needs to be created first. The observable expects a mechanism for
fantasized samples, which is created once the AdjointTrajectory struct is created. We then
attach the observable after the fact.
"""
attach_observable!(AT::Trajectory, observable::AbstractObservable) = AT.observable = observable
set_horizon!(T::Trajectory, h::Int) =  T.horizon = h


struct TrajectoryParameters
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
each_trajectory(tp::TrajectoryParameters; start::Int = 1) = start:tp.mc_iters
get_samples_rnstream(tp::TrajectoryParameters; sample_index) = tp.rnstream_sequence[sample_index, :, :]
get_starting_point(tp::TrajectoryParameters) = tp.x0[:]
get_hyperparameters(tp::TrajectoryParameters) = tp.θ[:]
get_horizon(tp::TrajectoryParameters) = tp.horizon
function set_starting_point!(tp::TrajectoryParameters, x::AbstractVector)
    @views begin
        tp.x0[:] .= x
    end
end


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