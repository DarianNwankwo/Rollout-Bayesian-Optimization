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

function base_solve(
    fantasy_surrogate::FantasySurrogate;
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    xstart::Vector{T},
    θfixed::Vector{T},
    fantasy_index::Int) where T <: Real

    function fun(x)
        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θfixed, fantasy_index=fantasy_index)
        return -eval(fantasy_surrogate_at_xθ)
    end
    
    function fun_grad!(g, x)
        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θfixed, fantasy_index=fantasy_index)
        g[:] = -gradient(fantasy_surrogate_at_xθ)
    end
    
    function fun_hess!(h, x)
        fantasy_surrogate_at_xθ = fantasy_surrogate(x, θfixed, fantasy_index=fantasy_index)
        h[:, :] .= -hessian(fantasy_surrogate_at_xθ)
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(spatial_lbs, spatial_ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
        Optim.Options(x_tol=1e-3, f_tol=1e-3)

    )
    
    return Optim.minimizer(res), res
end

function base_solve(
    surrogate::Surrogate;
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    xstart::Vector{T},
    θfixed::Vector{T}) where T <: Real

    function fun(x)
        fantasy_surrogate_at_xθ = surrogate(x, θfixed)
        return -eval(fantasy_surrogate_at_xθ)
    end
    
    function fun_grad!(g, x)
        fantasy_surrogate_at_xθ = surrogate(x, θfixed)
        g[:] = -gradient(fantasy_surrogate_at_xθ)
    end
    
    function fun_hess!(h, x)
        fantasy_surrogate_at_xθ = surrogate(x, θfixed)

        h[:, :] .= -hessian(fantasy_surrogate_at_xθ)
    end

    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, xstart)
    dfc = TwiceDifferentiableConstraints(spatial_lbs, spatial_ubs)
    res = optimize(
        df, dfc, xstart, IPNewton(),
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


function multistart_base_solve!(
    fantasy_surrogate::FantasySurrogate,
    xfinal::Vector{T};
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    guesses::Matrix{T},
    θfixed::Vector{T},
    fantasy_index::Int) where T <: Real
    if get_name(get_decision_rule(fantasy_surrogate)) == "Random"
        xfinal[:] = spatial_lbs .+ (spatial_ubs .- spatial_lbs) .* rand(length(spatial_lbs))
        return nothing
    end
    candidates = []
    
    for i in 1:size(guesses, 2)
        xi = guesses[:, i]

        minimizer, res = base_solve(
            fantasy_surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            xstart=xi,
            θfixed=θfixed,
            fantasy_index=fantasy_index,
        )
        push!(candidates, (minimizer, minimum(res)))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xfinal .= candidates[j_mini][1]

    return nothing
end

function multistart_base_solve!(
    surrogate::Surrogate,
    xfinal::Vector{T};
    spatial_lbs::Vector{T},
    spatial_ubs::Vector{T},
    guesses::Matrix{T},
    θfixed::Vector{T}) where T <: Real
    if get_name(get_decision_rule(surrogate)) == "Random"
        xfinal[:] = spatial_lbs .+ (spatial_ubs .- spatial_lbs) .* rand(length(spatial_lbs))
        return nothing
    end
    candidates = []
    
    for i in 1:size(guesses, 2)
        xi = guesses[:, i]

        minimizer, res = base_solve(
            surrogate,
            spatial_lbs=spatial_lbs,
            spatial_ubs=spatial_ubs,
            xstart=xi,
            θfixed=θfixed
        )
        push!(candidates, (minimizer, minimum(res)))
    end
    
    candidates = filter(pair -> !any(isnan.(pair[1])), candidates)
    mini, j_mini = findmin(pair -> pair[2], candidates)
    xfinal .= candidates[j_mini][1]

    return nothing
end