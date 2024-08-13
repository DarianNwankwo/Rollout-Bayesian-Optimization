include("radial_basis_surrogates.jl")

using SharedArrays

mutable struct Trajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    ∇fs::Matrix{Float64}
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
end


# function ForwardTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
    fmin = minimum(s.y)
    d, N = size(s.X)
    fsur = fit_fsurrogate(s, h)
    ∇fs = zeros(d, h+1)
    return Trajectory(s, fsur, ∇fs, fmin, x0, h)
end

function AdjointTrajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
    fmin = minimum(s.y)
    d, N = size(s.X)
    fsur = fit_sfsurrogate(s, h)
    return AdjointTrajectory(s, fsur, fmin, x0, h)
end

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
        check_stream_dimensions(rnstream_sequence, length(x0), h, mc_iters)
    
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