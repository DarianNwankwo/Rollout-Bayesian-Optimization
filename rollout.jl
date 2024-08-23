using Plots
using Sobol
using Distributions 
using LinearAlgebra
using Optim
using ForwardDiff
using Distributed
using Statistics


# Rename to rollout once refactor is complete
include("lazy_struct.jl")
include("low_discrepancy.jl")
include("optim.jl")
include("radial_basis_surrogates.jl")
include("radial_basis_functions.jl")
include("rbf_optim.jl")
include("trajectory.jl")
include("utils.jl")
include("testfns.jl")


function rollout!(
    T::Trajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    rnstream::Matrix{Float64},
    xstarts::Matrix{Float64})
    # Initial draw at predetermined location not chosen by policy
    f0 = gp_draw(T.fs, T.x0; stdnormal=rnstream[1,1])

    # Update surrogate and cache the gradient of the posterior mean field
    update_fsurrogate!(T.fs, T.x0, f0)
    T.∇fs[:, 1] = T.fs(T.x0).∇μ

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext .= multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi = gp_draw(T.fs, xnext; stdnormal=rnstream[1, j+1])
       
        # Update surrogate and cache the gradient of the posterior mean field
        update_fsurrogate!(T.fs, xnext, fi)
        T.∇fs[:, j+1] = T.fs(xnext).∇μ
        
        if fi < T.fmin
            T.fmin = fi
        end
    end
end

function function_sampler_constructor(t::TestFunction)
    return function deterministic(x::Vector{Float64})
        return t(x)
    end
end

function gp_sampler_constructor(fs::SmartFantasyRBFsurrogate, stdnormals::AbstractVector, max_fantasy_index)
    max_invocations = length(stdnormals)
    fantasy_step = 0
    
    return function gp_sampler(x::Vector{Float64})
        @assert fantasy_step < max_invocations "Maximum invocations have been used"
        fsx = fs(x, fantasy_index=fantasy_step-1)
        fantasy_step += 1
        observation = fsx.μ + fsx.σ*stdnormals[fantasy_step]

        return observation
    end
end

function adjoint_rollout!(
    T::AdjointTrajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    get_observation::Function,
    xstarts::Matrix{Float64})
    # Initial draw at predetermined location not chosen by policy
    f0::Number = get_observation(T.x0)

    # Update surrogate and cache the gradient of the posterior mean field
    update_sfsurrogate!(T.fs, T.x0, f0)

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext .= multistart_ei_solve(T.fs, lbs, ubs, xstarts, fantasy_index=j-1)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi::Number = get_observation(xnext)
       
        # Update surrogate and cache the gradient of the posterior mean field
        update_sfsurrogate!(T.fs, xnext, fi)
    end
end


function get_minimum_index(T)
    N = length(T.s.y)
    return minimum(T.fs.y[N+1:end])
end


function sample(T::Trajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Cannot sample from a trajectory that has not been rolled out"
    fantasy_slice = T.fs.known_observed + 1 : T.fs.known_observed + T.fs.fantasies_observed
    ∇f_offset = T.fs.known_observed
    return [
        (
            x=T.fs.X[:,i],
            y=T.fs.y[i],
            ∇f=T.∇fs[:, i - ∇f_offset]
        ) for i in fantasy_slice
    ]
end

function sample(T::AdjointTrajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Cannot sample from a trajectory that has not been rolled out"
    fantasy_slice = T.fs.known_observed + 1 : T.fs.known_observed + T.fs.fantasies_observed
    return [
        (
            x=T.fs.X[:, i],
            y=T.fs.y[i]
        ) for i in fantasy_slice
    ]
end


function best(T::Union{Trajectory, AdjointTrajectory})
    # Filter function to remove elements which have close x-values to their preceding element
    function find_min_index(path; epsilon=.01)
        # Initial values set to the first tuple in the path
        min_y = path[1].y
        min_x = path[1].x
        min_idx = 1
    
        # Start from the second tuple
        for i in 2:length(path)
            # If a new potential minimum y is found and its x value is not epsilon close to the current minimum's x
            if path[i].y < min_y && norm(path[i].x - min_x) > epsilon
                min_y = path[i].y
                min_x = path[i].x
                min_idx = i
            end
        end
    
        return min_idx
    end

    # step[2] corresponds to the function value
    path = sample(T)
    minndx = find_min_index(path)
    return minndx, path[minndx]
end


function α(T::Trajectory)
    path = sample(T)
    fmini = minimum(T.s.y)
    best_ndx, best_step = best(T)
    fb = best_step.y
    return max(fmini - fb, 0.)
end

function resolve(T::AdjointTrajectory)
    path = sample(T)
    fmini = minimum(T.s.y)
    best_ndx, best_step = best(T)
    fb = best_step.y
    return max(fmini - fb, 0.)
end

function get_fantasy_observations(T::AdjointTrajectory)
    N = T.fs.known_observed
    return T.fs.y[N+1:end]
end

function gradient(T::AdjointTrajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Rollout the trajectory before differentiating"
    _, max_fantasy_index = findmin(get_fantasy_observations(T))
    vbar = solve_adjoint_system(T, max_fantasy_index)
end


function simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    αxs::Vector{Float64})
    deepcopy_s = Base.deepcopy(s)

    for sample_ndx in 1:tp.mc_iters
        # Rollout trajectory
        T = Trajectory(deepcopy_s, tp.x0, tp.h)
        rollout!(T, tp.lbs, tp.ubs;
            rnstream=tp.rnstream_sequence[sample_ndx, :, :],
            xstarts=xstarts)

        # Evaluate rolled out trajectory
        αxs[sample_ndx] = α(T)
    end

    # Average across trajectories
    μx = mean(αxs)
    std_μx = std(αxs, mean=μx)

    return μx, std_μx
end

function deterministic_simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    testfn::TestFunction,
    resolutions::AbstractVector)
    deepcopy_s = Base.deepcopy(s)

    for sample_index in 1:tp.mc_iters
        # Rollout trajectory
        T = AdjointTrajectory(deepcopy_s, tp.x0, tp.h)
        sampler = function_sampler_constructor(testfn)
        adjoint_rollout!(T, tp.lbs, tp.ubs;
            get_observation=sampler,
            xstarts=xstarts,
        )

        # Evaluate rolled out trajectory
        resolutions[sample_index] = resolve(T)
    end

    return  mean(resolutions)
end

function simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    resolutions::AbstractVector)
    deepcopy_s = Base.deepcopy(s)

    for sample_index in 1:tp.mc_iters
        # Rollout trajectory
        T = AdjointTrajectory(deepcopy_s, tp.x0, tp.h)
        sampler = gp_sampler_constructor(
            T.fs, vec(tp.rnstream_sequence[sample_index, :, :]), tp.h
        )
        adjoint_rollout!(T, tp.lbs, tp.ubs;
            get_observation=sampler,
            xstarts=xstarts,
        )

        # Evaluate rolled out trajectory
        resolutions[sample_index] = resolve(T)
    end

    μ = mean(resolutions)
    σ = std(resolutions, mean=μ)
    return  μ, σ
end