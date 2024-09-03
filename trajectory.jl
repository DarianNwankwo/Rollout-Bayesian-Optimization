include("radial_basis_surrogates.jl")

using SharedArrays

abstract type Observable end

mutable struct StochasticObservable <: Observable
    fs::SmartFantasyRBFsurrogate
    stdnormal::AbstractMatrix
    trajectory_length::Int64
    step::Int64
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function StochasticObservable(fs, stdnormal, trajectory_length)
        dim = size(stdnormal, 1) - 1
        invocations = size(stdnormal, 2)
        observations = zeros(invocations)
        gradients = zeros(dim, invocations)
        return new(fs, stdnormal, trajectory_length, 0, observations, gradients)
    end
end

function (so::StochasticObservable)(x::AbstractVector)
    @assert so.step < so.trajectory_length "Maximum invocations have been used"
    observation, gradient_... = gp_draw(
        so.fs, x, stdnormal=so.stdnormal[:, so.step + 1], fantasy_index=so.step - 1, with_gradient=true
    )
    so.step += 1
    so.observations[so.step] = observation
    so.gradients[:, so.step] = gradient_

    return observation
end



mutable struct DeterministicObservable <: Observable
    testfn::TestFunction
    trajectory_length::Int64
    step::Int64
    observations::Vector{Float64}
    gradients::Matrix{Float64}

    function DeterministicObservable(t::TestFunction, trajectory_length)
        dim = testfn.dim
        observations = zeros(trajectory_length)
        gradients = zeros(dim, trajectory_length)
        return new(t, trajectory_length, 0, observations, gradients)
    end
end

function (deo::DeterministicObservable)(x::AbstractVector)
    @assert deo.step < deo.trajectory_length "Maximum invocations have been used"
    observation, gradient_ = testfn(x), gradient(testfn)(x)
    deo.step += 1
    deo.observations[deo.step] = observation
    deo.gradients[:, deo.step] = gradient_

    return observation
end

mutable struct Trajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    mfs::MultiOutputFantasyRBFsurrogate
    jacobians::Vector{Matrix{Float64}}
    fmin::Float64
    x0::Vector{Float64}
    h::Int
end

mutable struct AdjointTrajectory
    s::RBFsurrogate
    fs::SmartFantasyRBFsurrogate
    fmin::Float64
    x0::Vector{Float64}
    h::Int
    observable::Union{Nothing, Observable}
end


# function ForwardTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
    fmin = minimum(get_observations(s))
    d, N = size(s.X)

    ∇ys = [zeros(d) for i in 1:N]

    fsur = fit_fsurrogate(s, h)
    mfsur = fit_multioutput_fsurrogate(s, h)

    jacobians = [I(d)]

    return Trajectory(s, fsur, mfsur, jacobians, fmin, x0, h)
end

function AdjointTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
    fmin = minimum(s.y)
    d, N = size(s.X)
    fsur = fit_sfsurrogate(s, h)
    return AdjointTrajectory(s, fsur, fmin, x0, h, nothing)
end

attach_observable!(AT::AdjointTrajectory, observable::Observable) = AT.observable = observable

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
        rnstream_sequence::Array{Float64, 3},
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
    
        return new(x0, h, mc_iters, rnstream_sequence, lbs, ubs)
    end
end


"""
A lot of computations allocate the same object and then perform some computation of interest
and overwrite the memory with the desired value. We'll maintain objects that get reused in
hopes of the compiler generated highly efficient code.
"""
Base.@kwdef mutable struct PreallocatedContainer
    candidate_locations::SharedMatrix{Float64}
    candidate_values::SharedArray{Float64}
    inner_policy_starts::Matrix{Float64}
end