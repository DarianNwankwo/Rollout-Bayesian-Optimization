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


"""
Where j represents the jth column of all of our observations we want to compute the perturbation of.
"""
function compute_policy_perturbation(
    # T::ForwardTrajectory,
    T::Trajectory,
    xnext::Vector{Float64},
    jacobian_matrix::Matrix{Float64},
    total_observations::Int,
    j::Int)::Matrix{Float64}
    # Setup perturbation matrix where all columns, except 1, are zero.
    # println("Apply Perturbation to Column $(T.mfs.known_observed + j)")
    δXi = zeros(size(jacobian_matrix, 1), total_observations)

    # Evaluate surrogate at location from policy solve
    sxi = T.fs(xnext)

    # Placeholder for collecting all perturbations
    fantasy_begin_ndx = T.mfs.known_observed + 1
    # println("Fantasy Begin Index: $(fantasy_begin_ndx)")
    ∇ys = [zeros(length(xnext)) for i in 1:total_observations]
    ∇ys[fantasy_begin_ndx + j - 1] = T.mfs.∇y[:, j]

    # Intermediate storage of each column computed in the current perturbation
    P = zeros(length(xnext), length(xnext))
    
    # Perturb the gradient of the policy according to the perturbations in the jacobian matrix
    for (column_ndx, direction) in enumerate(eachcol(jacobian_matrix))
        δXi[:, fantasy_begin_ndx + j - 1] = direction
        # Compute perturbation
        δs = fit_δsurrogate(T.fs, δXi, ∇ys)
        δsxi = δs(sxi)
        # println("Perturbation Matrix: $(δXi)")
        # println("δsxi.μ: $(δsxi.μ) -- δsxi.∇μ: $(δsxi.∇μ) -- δsxi.kx: $(δsxi.kx) -- δsxi.∇kx: $(δsxi.∇kx) -- δsxi.∇EI: $(δsxi.∇EI) -- δsxi.EI: $(δsxi.EI)")
        P[:, column_ndx] = δsxi.∇EI
    end

    return P
end


function rollout!(
    T::Trajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    rnstream::Matrix{Float64},
    xstarts::Matrix{Float64})
    # Initial draw at predetermined location not chosen by policy
    f0, ∇f0 = gp_draw(T.mfs, T.x0; stdnormal=(@view rnstream[:,1]))

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_fsurrogate!(T.fs, T.x0, f0)
    update_multioutput_fsurrogate!(T.mfs, T.x0, f0, ∇f0)

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext .= multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.mfs, xnext; stdnormal=rnstream[:, j+1])
       
        # Placeholder for jacobian matrix
        δxi_jacobian::Matrix{Float64} = zeros(length(xnext), length(xnext))
        # Intermediate matrices before summing placeholder
        δxi_intermediates = Array{Matrix{Float64}}(undef, 0)

        total_observations = T.mfs.known_observed + T.mfs.fantasies_observed
        for (j, jacobian) in enumerate(T.jacobians)
            # Compute perturbation to each spatial location
            P = compute_policy_perturbation(T, xnext, jacobian, total_observations, j)
            push!(δxi_intermediates, P)
        end

        # Sum all perturbations
        δxi_intermediates = reduce(+, δxi_intermediates)
        # δxi_jacobian .= -T.fs(xnext).HEI \ δxi_intermediates
        if det(T.fs(xnext).HEI) < 1e-16
            δxi_jacobian .= zeros(length(xnext), length(xnext))
            # δxi_jacobian .= 0.
        else
            δxi_jacobian .= -T.fs(xnext).HEI \ δxi_intermediates
        end

        # Update surrogate, perturbed surrogate, and multioutput surrogate
        update_fsurrogate!(T.fs, xnext, fi)
        update_multioutput_fsurrogate!(T.mfs, xnext, fi, ∇fi)

        # Update jacobian matrix
        push!(T.jacobians, δxi_jacobian)

        if fi < T.fmin
            T.fmin = fi
        end
    end

    return nothing
end


function rollout2!(
    T::Trajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    rnstream::Vector{Float64},
    xstarts::Matrix{Float64})
    # Initial draw at predetermined location not chosen by policy
    f0 = gp_draw(T.fs, T.x0; stdnormal=rnstream[1])

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_fsurrogate!(T.fs, T.x0, f0)

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext .= multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi = gp_draw(T.fs, xnext; stdnormal=rnstream[j+1])
        # fi, ∇fi = gp_draw(T.mfs, xnext; stdnormal=rnstream[:, j+1])

        # Update surrogate, perturbed surrogate, and multioutput surrogate
        update_fsurrogate!(T.fs, xnext, fi)

        if fi < T.fmin
            T.fmin = fi
        end
    end

    return nothing
end






function adjoint_rollout!(
    T::AdjointTrajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    get_observation::Observable,
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
    mini, j_mini = findmin(T.fs.y[N+1:end])
    return j_mini
end


function sample(T::Trajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Cannot sample from a trajectory that has not been rolled out"
    fantasy_slice = T.fs.known_observed + 1 : T.fs.known_observed + T.fs.fantasies_observed
    ∇f_offset = T.fs.known_observed
    return [
        (
            x=T.mfs.X[:,i],
            y=T.mfs.y[i],
            ∇f=T.mfs.∇y[:, i - ∇f_offset]
        ) for i in fantasy_slice
    ]
    # return [
    #     (
    #         x=T.fs.X[:,i],
    #         y=T.fs.y[i]
    #     ) for i in fantasy_slice
    # ]
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

function ∇α(T::Trajectory)
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    xb, fb, ∇fb = best_step

    if fmini <= fb
        return zeros(length(xb))
    end

    if best_ndx == 1
        return -∇fb
    end
    
    opt_jacobian = T.jacobians[best_ndx]
    return transpose(-∇fb'*opt_jacobian)
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

function recover_policy_solve(T::AdjointTrajectory; solve_index::Int64)
    # @assert solve_index > 0 "The first step isn't chosen via an optimization problem"
    @assert solve_index <= T.h "Can only recover policy solves up to, and including, step $(T.h)"

    N = T.fs.known_observed
    xopt = T.fs.X[:, N + solve_index + 1]
    sx = T.fs(xopt, fantasy_index=solve_index-1)
    
    return sx
end

function solve_dual_y(T::AdjointTrajectory, x_duals::Vector{Vector{Float64}}; optimal_index::Int64, solve_index::Int64)
    y_dual = 0.
    dim = length(x_duals[1])
    
    for policy_solve_step in solve_index+1:optimal_index
        dp_sur = fit_data_perturbation_surrogate(T.fs, policy_solve_step - 1)
        sxi = recover_policy_solve(T, solve_index=policy_solve_step)

        δx = rand(dim)
        dp_sx = eval(dp_sur, sxi, δx=δx, current_step=solve_index)
        
        y_dual += dot(dp_sx.∇EI, x_duals[policy_solve_step])
    end

    return y_dual
end

function solve_dual_x(T::AdjointTrajectory, y_duals::Vector{Float64}, x_duals::Vector{Vector{Float64}}; optimal_index::Int64, solve_index::Int64)
    sx = recover_policy_solve(T, solve_index=solve_index)
    x_dual = -sx.∇μ * y_duals[solve_index + 1]
    dim = length(x_duals[1])
    I_d = Matrix{Float64}(I(dim))
    
    for policy_solve_step in solve_index+1:optimal_index
        sp_sur = fit_spatial_perturbation_surrogate(T.fs, policy_solve_step - 1)
        sxi = recover_policy_solve(T, solve_index=policy_solve_step)

        dri_dxj = zeros(size(I_d))
        for j in 1:dim
            δx = I_d[:, j]
            sp_sx = eval(sp_sur, sxi, δx=δx, current_step=solve_index)
            dri_dxj[:, j] = sp_sx.∇EI
        end
        x_dual -= dri_dxj' * x_duals[policy_solve_step]
    end

    x_dual = sx.HEI' \ x_dual
    
    return x_dual
end

function gather_g(T::AdjointTrajectory; optimal_index::Int64)
    sx = recover_policy_solve(T, solve_index=1)
    dim = length(sx.∇μ)
    g::Vector{AbstractMatrix} = [sx.∇μ']
    I_d = Matrix{Float64}(I(dim))

    for policy_solve_step in 1:optimal_index
        # Preallocate policy perturbation
        drj_dx0 = zeros(size(I_d))
        # Fit the spatial perturbation surrogate wrt to x0 only
        sp_sur = fit_spatial_perturbation_surrogate(T.fs, 0)
        # Recover the `solve_index` policy and perturb wrt to x0
        sxj = recover_policy_solve(T, solve_index=policy_solve_step)
        for j in 1:dim
            δx = I_d[:, j]
            sp_sx = eval(sp_sur, sxj, δx=δx, current_step=0)
            drj_dx0[:, j] = sp_sx.∇EI
        end
        
        push!(g, drj_dx0)
        # push!(g, zeros(1, dim))
    end

    return g
end

function gradient(T::AdjointTrajectory)
    fmini = minimum(T.s.y)
    best_ndx, best_step = best(T)
    fb = best_step.y

    if fmini <= fb
        return zeros(length(best_step.x))
    end

    t = get_minimum_index(T) - 1 # first sample isn't solved via an optimization routine
    if t == 0
        sx = recover_policy_solve(T, solve_index=0)
        # return T.s(T.x0).∇μ
        # return sx.∇μ
        return -T.observable.gradients[:, t+1]
    end
    dim = length(T.x0)
    xbars = [zeros(dim) for _ in 1:t]
    ybars = zeros(t+1)

    # Initialize backsubstitution procedure
    ybars[end] = 1
    # Solve linear system
    for j in t:-1:1
        xbars[j] = solve_dual_x(T, ybars, xbars, optimal_index=t, solve_index=j)
        ybars[j] = solve_dual_y(T, xbars, optimal_index=t, solve_index=j-1)
    end

    g = gather_g(T, optimal_index=t);

    grad = g[1]' * ybars[1]
    for j in 1:length(xbars)
        grad += g[j+1]' * xbars[j]
    end

    return -grad
end


function simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    αxs::Vector{Float64},
    ∇αxs::Matrix{Float64})
    deepcopy_s = Base.deepcopy(s)

    for sample_ndx in 1:tp.mc_iters
        # Rollout trajectory
        T = Trajectory(deepcopy_s, tp.x0, tp.h)
        rollout!(T, tp.lbs, tp.ubs;
            rnstream=tp.rnstream_sequence[sample_ndx, :, :],
            xstarts=xstarts)

        # Evaluate rolled out trajectory
        αxs[sample_ndx] = α(T)
        ∇αxs[:, sample_ndx] = ∇α(T)
    end

    # Average across trajectories
    μx = mean(αxs)
    ∇μx = mean(∇αxs, dims=2)
    std_μx = std(αxs, mean=μx)

    return μx, std_μx, ∇μx
end

function deterministic_simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    testfn::TestFunction,
    resolutions::AbstractVector,
    gradient_resolutions::AbstractMatrix)
    deepcopy_s = Base.deepcopy(s)

    for sample_index in 1:tp.mc_iters
        # Rollout trajectory
        T = AdjointTrajectory(deepcopy_s, tp.x0, tp.h)
        sampler = DeterministicObservable(testfn, tp.h + 1)
        attach_observable!(T, sampler)
        adjoint_rollout!(T, tp.lbs, tp.ubs;
            get_observation=T.observable,
            xstarts=xstarts,
        )

        # Evaluate rolled out trajectory
        resolutions[sample_index] = resolve(T)
        gradient_resolutions[:, sample_index] = gradient(T)
    end

    μ = mean(resolutions)
    σ = std(resolutions, mean=μ)
    ∇μ = mean(gradient_resolutions, dims=2)
    return  μ, σ, ∇μ
end

function simulate_adjoint_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    resolutions::AbstractVector,
    gradient_resolutions::AbstractMatrix)
    deepcopy_s = Base.deepcopy(s)

    for sample_index in 1:tp.mc_iters
        # Rollout trajectory
        T = AdjointTrajectory(deepcopy_s, tp.x0, tp.h)
        sampler = StochasticObservable(T.fs, tp.rnstream_sequence[sample_index, :, :], tp.h + 1)
        attach_observable!(T, sampler)
        adjoint_rollout!(T, tp.lbs, tp.ubs;
            get_observation=T.observable,
            xstarts=xstarts,
        )

        # Evaluate rolled out trajectory
        resolutions[sample_index] = resolve(T)
        gradient_resolutions[:, sample_index] = gradient(T)
    end

    μ = mean(resolutions)
    σ = std(resolutions, mean=μ)
    ∇μ = mean(gradient_resolutions, dims=2)
    return  μ, σ, ∇μ
end