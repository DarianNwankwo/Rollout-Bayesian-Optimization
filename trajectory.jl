include("radial_basis_surrogates.jl")


mutable struct Trajectory
    s::RBFsurrogate
    fs::FantasyRBFsurrogate
    mfs::MultiOutputFantasyRBFsurrogate
    jacobians::Vector{Matrix{Float64}}
    fmin::Float64
    x0::Vector{Float64}
    h::Int
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

"""
Consider giving the perturbed surrogate a zero matrix to handle computing variations
in the surrogate at the initial point.

- TODO: Fix the logic associated with maintaining the minimum found along the sample path vs.
that of the minimum from the best value known from the known locations.
"""
# function Trajectory(s::RBFsurrogate, x0::Vector{Float64}, h::Int)
#     # The ground truth surrogate is zero mean, so when we sample from our GP, we
#     # need to add the mean back in when we are updating the surrogate.
#     fmin = minimum(s.y) + s.ymean
#     d, N = size(s.X)

#     ∇ys = [zeros(d) for i in 1:N]
#     δX = zeros(d, N)

#     fsur = fit_fsurrogate(s, h)
#     δsur = fit_δsurrogate(fsur, δX, ∇ys)
#     mfsur = fit_multioutput_fsurrogate(s, h)

#     opt_HEI = Matrix{Float64}(I(d))

#     return Trajectory(s, fsur, δsur, mfsur, opt_HEI, fmin, x0, h)
# end


function update_trajectory!(
    T::Trajectory,
    x::Vector{Float64},
    δx::Vector{Float64},
    f::Float64,
    ∇f::Vector{Float64})
    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_fsurrogate!(T.fs, x, f)
    update_δsurrogate!(T.δfs, T.fs, δx, ∇f)
    update_multioutput_fsurrogate!(T.mfs, x, f,  ∇f)

    return nothing
end


function reset!(T::Trajectory)
    fmin = minimum(T.s.y) + T.s.ymean
    d, N = size(T.s.X)

    reset_fsurrogate!(T.fs, T.s)
    reset_δsurrogate!(T.δfs, T.fs)
    reset_mfsurrogate!(T.mfs, T.s)

    T.opt_HEI = zeros(d, d)
end


function check_dimensions(x0::Vector{Float64}, lbs::Vector{Float64}, ubs::Vector{Float64})
    n = length(x0)
    @assert length(lbs) == n && length(ubs) == n "Lower and upper bounds must be the same length as the initial point"
end


function check_stream_dimensions(rnstream_sequence::Array{Float64, 3}, d::Int, h::Int, mc_iters::Int)
    n_rows, n_cols = size(rnstream_sequence[1, :, :])
    @assert n_rows == d + 1 && n_cols <= h + 1 "Random number stream must have d + 1 rows and h + 1 columns for each sample"
    @assert size(rnstream_sequence, 1) == mc_iters "Random number stream must have at least mc_iters ($mc_iters) samples"
end


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
        check_dimensions(x0, lbs, ubs)
        check_stream_dimensions(rnstream_sequence, length(x0), h, mc_iters)
    
        return new(x0, h, mc_iters, rnstream_sequence, lbs, ubs)
    end
end