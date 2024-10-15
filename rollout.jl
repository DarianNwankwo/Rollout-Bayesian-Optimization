"""
Where j represents the jth column of all of our observations we want to compute the perturbation of.
"""
# function compute_policy_perturbation(
#     T::ForwardTrajectory,
#     xnext::Vector{Float64},
#     jacobian_matrix::Matrix{Float64},
#     total_observations::Int,
#     j::Int)::Matrix{Float64}
#     δXi = zeros(size(jacobian_matrix, 1), total_observations)

#     # Evaluate surrogate at location from policy solve
#     fs = get_fantasy_surrogate(T)
#     sxi = fs(xnext)

#     # Placeholder for collecting all perturbations
#     fantasy_begin_ndx = get_known_observations(fs) + 1
#     ∇ys = [zeros(length(xnext)) for i in 1:total_observations]
#     ∇ys[fantasy_begin_ndx + j - 1] = get_gradient(get_observable(T), at=j)

#     # Intermediate storage of each column computed in the current perturbation
#     P = zeros(length(xnext), length(xnext))

#     # Perturb the gradient of the policy according to the perturbations in the jacobian matrix
#     for (column_ndx, direction) in enumerate(eachcol(jacobian_matrix))
#         δXi[:, fantasy_begin_ndx + j - 1] = direction
        
#         # Compute perturbation
#         δs = fit_δsurrogate(get_fantasy_surrogate(T), δXi, ∇ys)
#         δsxi = δs(sxi)

#         P[:, column_ndx] = δsxi.∇EI
#     end

#     return P
# end


function rollout!(
    T::Trajectory;
    lowerbounds::Vector{T1},
    upperbounds::Vector{T1},
    get_observation::AO,
    xstarts::Matrix{T1}) where {T1 <: Real, AO <: AbstractObservable}
    # Initial draw at predetermined location not chosen by policy
    f0 = get_observation(get_starting_point(T), get_hyperparameters(T))

    # Update surrogate and cache the gradient of the posterior mean field
    condition!(get_fantasy_surrogate(T), get_starting_point(T), f0)

    # Preallocate for newton solves
    xlen = length(get_starting_point(T))
    xnext = zeros(xlen)

    # Perform rollout for fantasized trajectories
    for j in 1:get_horizon(T)
        # Solve base acquisition function to determine next sample location
        multistart_base_solve!(
            get_fantasy_surrogate(T),
            xnext,
            spatial_lbs=lowerbounds,
            spatial_ubs=upperbounds,
            θfixed=get_hyperparameters(T),
            guesses=xstarts,
            fantasy_index=j-1,
        )

        # Draw fantasized sample at proposed location after base acquisition solve
        fi = get_observation(xnext, get_hyperparameters(T))
       
        # Update surrogate and cache the gradient of the posterior mean field
        condition!(get_fantasy_surrogate(T), xnext, fi)
    end
end


function get_minimum_index(T::AbstractTrajectory)
    y = get_observations(get_base_surrogate(T))
    y_fantasies = get_fantasy_observations(get_fantasy_surrogate(T))
    mini, j_mini = findmin(y_fantasies)
    return j_mini
end


function sample(T::AbstractTrajectory)
    fs = get_fantasy_surrogate(T)
    m = get_fantasies_observed(fs)
    n = get_known_observations(fs)
    @assert m == get_horizon(T) + 1 "Cannot sample from a trajectory that has not been rolled out"
    
    fantasy_slice = n + 1 : n + m
    return [
        (
            x=T.fs.X[:, i],
            y=T.fs.y[i]
        ) for i in fantasy_slice
    ]
end


function best(T::AbstractTrajectory)
    path = sample(T)
    trajectory_index = get_minimum_index(T) - 1
    return trajectory_index, path[trajectory_index + 1]
end


function resolve(T::AbstractTrajectory)
    fmini = minimum(get_observations(get_base_surrogate(T)))
    best_ndx, best_step = best(T)
    fb = best_step.y
    return max(fmini - fb, 0.)
end



function recover_policy_solve(T::Trajectory; solve_index::Int64)
    # @assert solve_index > 0 "The first step isn't chosen via an optimization problem"
    @assert solve_index <= get_horizon(T) + 1 "Can only recover policy solves up to, and including, step $(T.h + 1)"

    fs = get_fantasy_surrogate(T)
    N = get_known_observations(fs)
    xopt = get_covariates(fs)[:, N + solve_index + 1]
    sx = fs(xopt, T.θ, fantasy_index=solve_index-1)
    
    return sx
end

function solve_dual_y(
    T::Trajectory,
    x_duals::Vector{Vector{Float64}};
    optimal_index::Int,
    solve_index::Int)
    y_dual = 0.
    dim = length(x_duals[1])
    δx = rand(dim)
    
    for policy_solve_step in solve_index+1:optimal_index
        dp_sur = DataPerturbationSurrogate(
            reference_surrogate=get_fantasy_surrogate(T),
            fantasy_step=policy_solve_step - 1
        )
        sxi = recover_policy_solve(T, solve_index=policy_solve_step)
        ∇y = get_gradient(get_observable(T), at=optimal_index)
        dp_sx = dp_sur(sxi, variation=δx, ∇y=∇y, sample_index=solve_index)
        
        y_dual += dot(gradient(dp_sx), x_duals[policy_solve_step])
    end

    return y_dual
end

function solve_dual_x(
    T::Trajectory,
    y_duals::Vector{Float64},
    x_duals::Vector{Vector{Float64}};
    optimal_index::Int, 
    solve_index::Int,
    htol::Float64 = 1e-16)
    sx = recover_policy_solve(T, solve_index=solve_index)

    if det(hessian(sx)) < htol
        return zeros(length(get_starting_point(T)))
    end

    x_dual = -sx.∇μ * y_duals[solve_index + 1]
    
    dim = length(get_starting_point(T))
    I_d = Matrix{Float64}(I(dim))
    # Preallocate once and reuse
    dri_dxj = zeros(size(I_d))
    
    for policy_solve_step in solve_index+1:optimal_index
        sp_sur = SpatialPerturbationSurrogate(
            reference_surrogate=get_fantasy_surrogate(T),
            fantasy_step=policy_solve_step - 1
        )
        sxi = recover_policy_solve(T, solve_index=policy_solve_step)
        
        for j in 1:dim
            sp_sx = sp_sur(sxi, variation=I_d[:, j], sample_index=solve_index)
            dri_dxj[:, j] = gradient(sp_sx)
        end

        x_dual -= dri_dxj' * x_duals[policy_solve_step]
    end

    x_dual = hessian(sx)' \ x_dual
    
    return x_dual
end

function gather_g(T::Trajectory; optimal_index::Int)
    sx = recover_policy_solve(T, solve_index=0)
    dim = length(sx.∇μ)
    g::Vector{Matrix{Float64}} = [sx.∇μ']
    I_d = Matrix{Float64}(I(dim))

    for policy_solve_step in 1:optimal_index
        # Preallocate policy perturbation
        drj_dx0 = zeros(size(I_d))
        # Fit the spatial perturbation surrogate wrt to x0 only
        sp_sur = SpatialPerturbationSurrogate(
            reference_surrogate=get_fantasy_surrogate(T),
            fantasy_step=policy_solve_step - 1
        )
        # Recover the `solve_index` policy and perturb wrt to x0, i.e. sample_index = 0
        sxj = recover_policy_solve(T, solve_index=policy_solve_step)
        for j in 1:dim
            sp_sx = sp_sur(sxj, variation=I_d[:, j], sample_index=0)
            drj_dx0[:, j] = gradient(sp_sx)
        end
        
        push!(g, drj_dx0)
    end

    return g
end

function gather_q(T::Trajectory; optimal_index::Int)
    q::Vector{AbstractMatrix} = []

    for solve_index in 1:optimal_index
        sxj = recover_policy_solve(T, solve_index=solve_index)
        push!(q, hessian(sxj, wrt_hypers=true))
    end
    
    return q
end

function gradient(T::Trajectory)
    fmini = minimum(get_observations(get_base_surrogate(T)))
    t, best_step = best(T)
    fb = best_step.y
    xdim, θdim = length(T.x0), length(T.θ)

    # Case #1: Nothing along the trajectory was found to be better than what has been observed,
    # hence we return the zero vector
    if fmini <= fb
        return (∇x=zeros(xdim), ∇θ=zeros(θdim))
    end

    # Case #2: The initial sample was the best step along the trajectory so we return the exact gradient
    # in the determinisitc case. In the stochastic case, we return the gradient of the mean field since in
    # adjoint mode, the gradient depends linearly on the function gradients. Therefore, we replace the random
    # variables with their expectations.
    if t == 0
        observable = get_observable(T)
        if observable isa DeterministicObservable
            return (∇x=-get_gradient(observable, at=t+1), ∇θ=zeros(θdim))
        elseif observable isa StochasticObservable
            # sx = recover_policy_solve(T, solve_index=1)
            # return (∇x=-sx.∇μ, ∇θ=zeros(θdim))
            return (∇x=-get_gradient(observable, at=t+1), ∇θ=zeros(θdim))
        else
            error("Unsupported observable type")
        end
    end
    
    # Case #3: The best step along the trajectory was at a non-trivial step (>= 1)
    xbars = [zeros(xdim) for _ in 1:t]
    ybars = zeros(t+1)

    # Initialize backsubstitution procedure
    ybars[end] = 1.
    # Solve linear system via backsubstitution. The entries of ybars and xbars have been
    # preallocated, so the solution vectors are updated at the correct indices
    for j in t:-1:1
        xbars[j] = solve_dual_x(T, ybars, xbars, optimal_index=t, solve_index=j)
        ybars[j] = solve_dual_y(T, xbars, optimal_index=t, solve_index=j-1)
    end

    g = gather_g(T, optimal_index=t)
    q = gather_q(T, optimal_index=t)

    grad_x = g[1]' * ybars[1]
    grad_θ = zeros(length(T.θ))

    for j in 1:length(xbars)
        grad_x += g[j+1]' * xbars[j]
        grad_θ += q[j]' * xbars[j]
    end
    grad_x = vec(grad_x)

    return (∇x=-grad_x, ∇θ=-grad_θ)
end

function simulate_adjoint_trajectory(
    T::Trajectory,
    tp::TrajectoryParameters;
    inner_solve_xstarts::Matrix{T1},
    resolutions::Vector{T1},
    spatial_gradients_container::Union{Nothing, Matrix{T1}} = nothing,
    hyperparameter_gradients_container::Union{Nothing, Matrix{T1}} = nothing) where T1 <: Real
    slbs, subs = get_spatial_bounds(tp)

    for sample_index in each_trajectory(tp)
        # Rollout trajectory
        sampler = StochasticObservable(
            surrogate=get_fantasy_surrogate(T), 
            stdnormal=get_samples_rnstream(tp, sample_index=sample_index),
            max_invocations=get_horizon(tp) + 1
        )
        attach_observable!(T, sampler)

        rollout!(
            T,
            lowerbounds=slbs,
            upperbounds=subs,
            get_observation=get_observable(T),
            xstarts=inner_solve_xstarts
        )

        # Evaluate rolled out trajectory
        resolutions[sample_index] = resolve(T)
        if !isnothing(spatial_gradients_container) && !isnothing(hyperparameter_gradients_container)
            ∇x, ∇θ = gradient(T)
            spatial_gradients_container[:, sample_index] = ∇x
            hyperparameter_gradients_container[:, sample_index] = ∇θ
        end

        reset!(get_fantasy_surrogate(T))
    end

    μxθ = Distributions.mean(resolutions)
    σ_μxθ = Distributions.std(resolutions, mean=μxθ)
    
    if isnothing(spatial_gradients_container) && isnothing(hyperparameter_gradients_container)
        return ExpectedTrajectoryOutput(μxθ=μxθ, σ_μxθ=σ_μxθ)
    else
        ∇μx = vec(Distributions.mean(spatial_gradients_container, dims=2))
        σ_∇μx = vec(Distributions.std(spatial_gradients_container, dims=2, mean=∇μx))
        ∇μθ = vec(Distributions.mean(hyperparameter_gradients_container, dims=2))
        σ_∇μθ = vec(Distributions.std(hyperparameter_gradients_container, dims=2, mean=∇μθ))
        return  ExpectedTrajectoryOutput(μxθ=μxθ, σ_μxθ=σ_μxθ, ∇μx=∇μx, σ_∇μx=σ_∇μx, ∇μθ=∇μθ, σ_∇μθ=σ_∇μθ)
    end
end