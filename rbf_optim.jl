function ei_solve(s::FantasyRBFsurrogate, lbs::Vector{Float64}, ubs::Vector{Float64}
    , xstart::Vector{Float64})
    fun(x) = -s(x).EI
    function fun_grad!(g, x)
        g[:] = -s(x).∇EI
    end
    function fun_hess!(h, x)
        h[:, :] = -s(x).HEI
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(lbs, ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3)
    )

    return Optim.minimizer(res), res
end

function ei_solve(s::SmartFantasyRBFsurrogate, lbs::Vector{Float64}, ubs::Vector{Float64}
    , xstart::Vector{Float64}; fantasy_index::Int64)
    fun(x) = -s(x, fantasy_index=fantasy_index).EI
    function fun_grad!(g, x)
        g[:] = -s(x, fantasy_index=fantasy_index).∇EI
    end
    function fun_hess!(h, x)
        h[:, :] = -s(x, fantasy_index=fantasy_index).HEI
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(lbs, ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3)
        # Optim.Options(f_tol=1e-16)
    )

    return Optim.minimizer(res), res
end

function base_solve(
    fantasy_surrogate::FantasySurrogate;
    spatial_lbs::AbstractVector,
    spatial_ubs::AbstractVector,
    hyperparameters_lbs::AbstractVector,
    hyperparameters_ubs::AbstractVector,
    xstart::AbstractVector,
    θstart::AbstractVector,
    fantasy_index::Int,
    cost::AbstractCostFunction = UniformCost())
    start = vcat(xstart, θstart)

    function fun(xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]
        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θ, fantasy_index=fantasy_index, cost=cost)
        return -eval(fantasy_surrogate_at_xθ)
    end
    
    function fun_grad!(g, xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]
        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θ, fantasy_index=fantasy_index, cost=cost)
        g[1:length(xstart)] = -gradient(fantasy_surrogate_at_xθ)
        g[length(x)+1:end] = -gradient(fantasy_surrogate_at_xθ, wrt_hypers=true)
    end
    
    function fun_hess!(h, xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]

        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θ, fantasy_index=fantasy_index, cost=cost)
        spatial_slice = 1:length(x)
        hyperparameters_slice = length(x)+1:length(θ)+1

        h[spatial_slice, spatial_slice] .= -hessian(fantasy_surrogate_at_xθ)
        h[hyperparameters_slice, hyperparameters_slice] .= -hessian(fantasy_surrogate_at_xθ, wrt_hypers=true)
        h[hyperparameters_slice, spatial_slice] .= mixed_partials(fantasy_surrogate_at_xθ)
        h[spatial_slice, hyperparameters_slice] .= h[hyperparameters_slice, spatial_slice]'
    end

    lbs = vcat(spatial_lbs, hyperparameters_lbs)
    ubs = vcat(spatial_ubs, hyperparameters_ubs)
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, start)
    dfc = TwiceDifferentiableConstraints(lbs, ubs)
    res = optimize(
        df, dfc, start, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3)

    )
    
    return Optim.minimizer(res), res
end

function base_solve(
    surrogate::Surrogate;
    spatial_lbs::AbstractVector,
    spatial_ubs::AbstractVector,
    hyperparameters_lbs::AbstractVector,
    hyperparameters_ubs::AbstractVector,
    xstart::AbstractVector,
    θstart::AbstractVector,
    cost::AbstractCostFunction = UniformCost())
    start = vcat(xstart, θstart)

    function fun(xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]
        fantasy_surrogate_at_xθ = surrogate(x, θ, cost=cost)
        return -eval(fantasy_surrogate_at_xθ)
    end
    
    function fun_grad!(g, xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]
        fantasy_surrogate_at_xθ = surrogate(x, θ, cost=cost)
        g[1:length(xstart)] = -gradient(fantasy_surrogate_at_xθ)
        g[length(x)+1:end] = -gradient(fantasy_surrogate_at_xθ, wrt_hypers=true)
    end
    
    function fun_hess!(h, xθ)
        x = xθ[1:length(xstart)]
        θ = xθ[length(x)+1:end]

        fantasy_surrogate_at_xθ = surrogate(x, θ, cost=cost)
        spatial_slice = 1:length(x)
        hyperparameters_slice = length(x)+1:length(θ)+1

        h[spatial_slice, spatial_slice] .= -hessian(fantasy_surrogate_at_xθ)
        h[hyperparameters_slice, hyperparameters_slice] .= -hessian(fantasy_surrogate_at_xθ, wrt_hypers=true)
        h[hyperparameters_slice, spatial_slice] .= mixed_partials(fantasy_surrogate_at_xθ)
        h[spatial_slice, hyperparameters_slice] .= h[hyperparameters_slice, spatial_slice]'
    end

    lbs = vcat(spatial_lbs, hyperparameters_lbs)
    ubs = vcat(spatial_ubs, hyperparameters_ubs)
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, start)
    dfc = TwiceDifferentiableConstraints(lbs, ubs)
    res = optimize(
        df, dfc, start, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3)

    )
    
    return Optim.minimizer(res), res
end

function multistart_ei_solve(s::FantasyRBFsurrogate, lbs::Vector{Float64},
    ubs::Vector{Float64}, xstarts::Matrix{Float64})::Vector{Float64}
    candidates = []
    
    for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        try
            minimizer, res = ei_solve(s, lbs, ubs, xi)
            push!(candidates, (minimizer, minimum(res)))
        catch e
            println(e)
        end
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    minimizer = candidates[j_mini][1]

    return minimizer
end

function multistart_ei_solve(s::SmartFantasyRBFsurrogate, lbs::Vector{Float64},
    ubs::Vector{Float64}, xstarts::Matrix{Float64}; fantasy_index::Int64)::Vector{Float64}
    candidates = []
    
    for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        try
            minimizer, res = ei_solve(s, lbs, ubs, xi, fantasy_index=fantasy_index)
            push!(candidates, (minimizer, minimum(res)))
        catch e
            println(e)
        end
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    minimizer = candidates[j_mini][1]

    return minimizer
end

function multistart_base_solve(
    fantasy_surrogate::FantasySurrogate;
    spatial_lbs::AbstractVector,
    spatial_ubs::AbstractVector,
    hyperparameters_lbs::AbstractVector,
    hyperparameters_ubs::AbstractVector,
    xstarts::AbstractMatrix,
    θstarts::AbstractMatrix,
    fantasy_index::Int,
    cost::AbstractCostFunction = UniformCost())::AbstractVector
    candidates = []
    
    for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        θi = θstarts[:, i]

        minimizer, res = base_solve(
            fantasy_surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            hyperparameters_lbs=hyperparameters_lbs,
            hyperparameters_ubs=hyperparameters_ubs,
            xstart=xi,
            θstart=θi,
            fantasy_index=fantasy_index,
            cost=cost
        )
        push!(candidates, (minimizer, minimum(res)))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    minimizer = candidates[j_mini][1]

    return minimizer
end

function multistart_base_solve(
    surrogate::Surrogate;
    spatial_lbs::AbstractVector,
    spatial_ubs::AbstractVector,
    hyperparameters_lbs::AbstractVector,
    hyperparameters_ubs::AbstractVector,
    xstarts::AbstractMatrix,
    θstarts::AbstractMatrix,
    cost::AbstractCostFunction = UniformCost())::AbstractVector
    candidates = []
    
    for i in 1:size(xstarts, 2)
        xi = xstarts[:,i]
        θi = θstarts[:, i]

        minimizer, res = base_solve(
            surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            hyperparameters_lbs=hyperparameters_lbs,
            hyperparameters_ubs=hyperparameters_ubs,
            xstart=xi,
            θstart=θi,
            cost=cost
        )
        push!(candidates, (minimizer, minimum(res)))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    minimizer = candidates[j_mini][1]

    return minimizer
end