using SharedArrays
using Distributed

include("radial_basis_surrogates.jl")


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
        # Optim.Options(x_tol=1e-3, f_tol=1e-3)
        # Optim.Options(f_tol=1e-16)
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