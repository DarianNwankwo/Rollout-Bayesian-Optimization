using Plots
using Sobol
using Distributions 
using LinearAlgebra
using Optim
using ForwardDiff
using Distributed


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


function not_near(x::Vector{Float64}, X::Matrix{Float64}; tol::Float64=1e-6)
    return all([norm(x - X[:, i]) > tol for i in 1:size(X, 2)])
end


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

"""
I THINK MY COVARIANCE MEASURES FOR GRADIENTS IS WRONG

If we happen to be near a known location, we shouldn't sample from our multioutput surrogate, but rather our
ground truth single output surrogate, I suspect. Here is what is going on:

Our first sample on our fantasized trajectory isn't driven by our policy explicitly. It says that if EI told us to sample
at x0, what is the anticipated behavior, after using EI, for the next h samples. So, our trajectory contains h+1 samples,
where the first sample is at x0, and the next h samples are driven by EI. So h+1 fantasized samples, h of which are driven
by EI. When x0 happens to be near a known location, we observe a tendency of the rollout acquisition function to blowup.
I think this occurs when x0 is a known location that is close to the best thing we've seen thus far.

This means that if EI told us to sample at a known location and then we rollout our trajectory and there is some
expected reduction in objective, our rollout acquisition function is going to say x0 is a decent location to sample our
original process at.

When we sample gradient information at x0, when x0 is near a known point, we have a tendency to learn an approximation
of the underlying function that is not accurate.
"""
function rollout!(
    T::Trajectory,
    lbs::Vector{Float64},
    ubs::Vector{Float64};
    rnstream::Matrix{Float64},
    xstarts::Matrix{Float64},
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64}
    )
    # Initial draw at predetermined location not chosen by policy
    f0, ∇f0 = gp_draw(T.mfs, T.x0; stdnormal=rnstream[:,1])

    # Update surrogate, perturbed surrogate, and multioutput surrogate
    update_fsurrogate!(T.fs, T.x0, f0)
    update_multioutput_fsurrogate!(T.mfs, T.x0, f0, ∇f0)

    # Preallocate for newton solves
    xnext = zeros(length(T.x0))

    # Perform rollout for fantasized trajectories
    for j in 1:T.h
        # Solve base acquisition function to determine next sample location
        xnext .= distributed_multistart_ei_solve(
            T.fs, lbs, ubs, xstarts,
            candidate_locations=candidate_locations, candidate_values=candidate_values
        )
        # xnext = multistart_ei_solve(T.fs, lbs, ubs, xstarts)

        # Draw fantasized sample at proposed location after base acquisition solve
        fi, ∇fi = gp_draw(T.mfs, xnext; stdnormal=rnstream[:, j+1])
       
        # Placeholder for jacobian matrix
        δxi_jacobian::Matrix{Float64} = zeros(length(xnext), length(xnext))
        # Intermediate matrices before summing placeholder
        # Create a type for an array of matrices
        δxi_intermediates = Array{Matrix{Float64}}(undef, 0)

        total_observations = T.mfs.known_observed + T.mfs.fantasies_observed
        for (j, jacobian) in enumerate(T.jacobians)
            # Compute perturbation to each spatial location
            P = compute_policy_perturbation(T, xnext, jacobian, total_observations, j)
            push!(δxi_intermediates, P)
        end

        # Sum all perturbations
        δxi_intermediates = reduce(+, δxi_intermediates)
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


function sample(T::Trajectory)
    @assert T.fs.fantasies_observed == T.h + 1 "Cannot sample from a trajectory that has not been rolled out"
    fantasy_slice = T.fs.known_observed + 1 : T.fs.known_observed + T.fs.fantasies_observed
    M = T.fs.known_observed
    return [
        (
            x=T.mfs.X[:,i],
            y=T.mfs.y[i] .+ T.mfs.ymean,
            ∇y=T.mfs.∇y[:,i-M] .+ T.mfs.∇ymean,
        ) for i in fantasy_slice
    ]
end


function best(T::Trajectory)
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
    fmini = minimum(get_observations(T.s))
    best_ndx, best_step = best(T)
    fb = best_step.y
    # println("Best Index: $best_ndx")
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


function simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    variance_reduction::Bool=false,
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64}
    )
    αxs, ∇αxs = zeros(tp.mc_iters), zeros(length(tp.x0), tp.mc_iters)
    deepcopy_s = Base.deepcopy(s)

    for sample_ndx in 1:tp.mc_iters
        # Rollout trajectory
        T = Trajectory(deepcopy_s, tp.x0, tp.h)
        rollout!(T, tp.lbs, tp.ubs;
            rnstream=tp.rnstream_sequence[sample_ndx, :, :],
            xstarts=xstarts,
            candidate_locations=candidate_locations,
            candidate_values=candidate_values
        )
        
        # Evaluate rolled out trajectory
        αxs[sample_ndx] = α(T)
        ∇αxs[:, sample_ndx] .= ∇α(T)
    end

    # Average trajectories
    μx::Float64 = sum(αxs) / tp.mc_iters
    ∇μx::Vector{Float64} = vec(sum(∇αxs, dims=2) / tp.mc_iters)
    stderr_μx = sqrt(sum((αxs .- μx) .^ 2) / (tp.mc_iters - 1))
    stderr_∇μx = sqrt(sum((∇αxs .- ∇μx) .^ 2) / (tp.mc_iters - 1))

    if variance_reduction
        sx = s(tp.x0)
        μx += sx.EI
        ∇μx .+= sx.∇EI
    end

    return μx, ∇μx, stderr_μx, stderr_∇μx
end


function distributed_simulate_trajectory(
    s::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64};
    variance_reduction::Bool=false,
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64},
    αxs::SharedArray{Float64},
    ∇αxs::SharedMatrix{Float64}
    )
    deepcopy_s = Base.deepcopy(s)

    @sync @distributed for sample_ndx in 1:tp.mc_iters
        # Rollout trajectory
        T = Trajectory(deepcopy_s, tp.x0, tp.h)
        rollout!(T, tp.lbs, tp.ubs;
            rnstream=tp.rnstream_sequence[sample_ndx, :, :],
            xstarts=xstarts,
            candidate_locations=candidate_locations,
            candidate_values=candidate_values
        )
        
        # Evaluate rolled out trajectory
        αxs[sample_ndx] = α(T)
        ∇αxs[:, sample_ndx] .= ∇α(T)
    end

    # Average trajectories
    μx::Float64 = sum(αxs) / tp.mc_iters
    ∇μx::Vector{Float64} = vec(sum(∇αxs, dims=2) / tp.mc_iters)
    stderr_μx = sqrt(sum((αxs .- μx) .^ 2) / (tp.mc_iters - 1))
    stderr_∇μx = sqrt(sum((∇αxs .- ∇μx) .^ 2) / (tp.mc_iters - 1))

    if variance_reduction
        sx = s(tp.x0)
        μx += sx.EI
        ∇μx .+= sx.∇EI
    end

    return μx, ∇μx, stderr_μx, stderr_∇μx
end


function rollout_solver(;
    sur::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64},
    batch::Matrix{Float64},
    max_iterations::Int = 25,
    varred::Bool = true,
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64}
    )
    batch_results = Array{Any, 1}(undef, size(batch, 2))

    for i in 1:size(batch, 2)
        # Update start of trajectory for each point in the batch
        tp.x0 .= batch[:, i]

        # Perform stochastic gradient ascent on the point in the batch
        batch_results[i] = stochastic_gradient_ascent_adam1(
            sur=sur,
            tp=tp,
            max_sgd_iters=max_iterations,
            varred=varred,
            xstarts=xstarts,
            candidate_locations=candidate_locations,
            candidate_values=candidate_values
        )
    end

    # Find the point in the batch that maximizes the rollout acquisition function
    best_tuple = first(batch_results)
    for result in batch_results[2:end]
        if result.final_obj > best_tuple.final_obj
            best_tuple = result
        end
    end

    return best_tuple.finish, best_tuple.final_obj
end

function distributed_rollout_solver(;
    sur::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64},
    batch::Matrix{Float64},
    max_iterations::Int = 100,
    varred::Bool = true,
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64},
    αxs::SharedArray{Float64},
    ∇αxs::SharedMatrix{Float64},
    final_locations::SharedMatrix{Float64},
    final_evaluations::SharedArray{Float64}
    )
    for i in 1:size(batch, 2)
        # Update start of trajectory for each point in the batch
        tp.x0 = batch[:, i]

        # Perform stochastic gradient ascent on the point in the batch
        result = stochastic_gradient_ascent_adam(
            sur=sur,
            tp=tp,
            max_sgd_iters=max_iterations,
            varred=varred,
            xstarts=xstarts,
            candidate_locations=candidate_locations,
            candidate_values=candidate_values,
            αxs=αxs,
            ∇αxs=∇αxs
        )
        final_locations[:, i] = result.finish
        final_evaluations[i] = result.final_obj
    end

    # Find the point in the batch that maximizes the rollout acquisition function
    best_ndx, best_evaluation, best_location = 1, first(final_evaluations), final_locations[:, 1]
    for i in 1:size(batch, 2)
        if final_evaluations[i] > best_evaluation
            best_ndx = i
            best_evaluation = final_evaluations[i]
            best_location = final_locations[:, i]
        end
    end

    return (best_location, best_evaluation)
end


function rollout_solver_saa(;
    sur::RBFsurrogate,
    tp::TrajectoryParameters,
    xstarts::Matrix{Float64},
    batch::Matrix{Float64},
    max_iterations::Int = 100,
    varred::Bool = true,
    candidate_locations::SharedMatrix{Float64},
    candidate_values::SharedArray{Float64},
    αxs::SharedArray{Float64},
    ∇αxs::SharedMatrix{Float64},
    final_locations::SharedMatrix{Float64},
    final_evaluations::SharedArray{Float64}
    )
    for i in 1:size(batch, 2)
        # Update start of trajectory for each point in the batch
        tp.x0 = batch[:, i]

        # Perform stochastic gradient ascent on the point in the batch
        result = stochastic_gradient_ascent_adam(
            sur=sur,
            tp=tp,
            max_sgd_iters=max_iterations,
            varred=varred,
            xstarts=xstarts,
            candidate_locations=candidate_locations,
            candidate_values=candidate_values,
            αxs=αxs,
            ∇αxs=∇αxs
        )
        final_locations[:, i] = result.finish
        final_evaluations[i] = result.final_obj
    end

    # Find the point in the batch that maximizes the rollout acquisition function
    best_ndx, best_evaluation, best_location = 1, first(final_evaluations), final_locations[:, 1]
    for i in 1:size(batch, 2)
        if final_evaluations[i] > best_evaluation
            best_ndx = i
            best_evaluation = final_evaluations[i]
            best_location = final_locations[:, i]
        end
    end

    return (best_location, best_evaluation)
end