using ForwardDiff

abstract type Policy end

"""TODO:
The idea here is to get an arbitrary function from the user in terms of μ, σ, and θ
and store all the derivatives we care about. If the user doesn't provide these things,
we compute them via automatic differentiation for them.
"""
struct BasePolicy <: Policy
    g::Function             # The function g(μ, σ, θ)
    dg_dμ::Function         # Gradient of g with respect to μ
    d2g_dμ::Function        # Hessian of g with respect to μ
    dg_dσ::Function         # Gradient of g with respect to σ
    d2g_dσ::Function        # Hessian of g with respect to σ
    dg_dθ::Function         # Gradient of g with respect to θ
    d2g_dθ::Function         # Hessian of g with respect to θ

    BasePolicy(g, dg_dμ, d2g_dμ, dg_dσ, d2g_dσ, dg_dθ, d2g_dθ) = new(g, dg_dμ, d2g_dμ, dg_dσ, d2g_dσ, dg_dθ, d2g_dθ)
end


function BasePolicy(g::Function)
    dg_dμ(μ, σ, θ) = ForwardDiff.derivative(μ -> g(μ, σ, θ), μ)
    d2g_dμ(μ, σ, θ) = ForwardDiff.derivative(μ -> dg_dμ(μ, σ, θ), μ)
    dg_dσ(μ, σ, θ) = ForwardDiff.derivative(σ -> g(μ, σ, θ), σ)
    d2g_dσ(μ, σ, θ) = ForwardDiff.derivative(σ -> dg_dμ(μ, σ, θ), σ)
    dg_dθ(μ, σ, θ) = ForwardDiff.gradient(θ -> g(μ, σ, θ), θ)
    d2g_dθ(μ, σ, θ) = ForwardDiff.hessian(θ -> dg_dμ(μ, σ, θ), θ)

    return BasePolicy(g, dg_dμ, d2g_dμ, dg_dσ, d2g_dσ, dg_dθ, d2g_dθ)
end

(bp::BasePolicy)(μ::Number, σ::Number, θ::AbstractVector) = bp.g(μ, σ, θ)


function first_partial(p::Policy; symbol::Symbol)
    if symbol == :μ
        return p.dg_dμ
    elseif symbol == :σ
        return p.dg_dσ
    elseif symbol == :θ
        return p.dg_dθ
    else
        error("Unknown symbol. Use :μ, :σ, or :θ")
    end
end
first_partials(p::Policy) = (
    μ=first_partial(p, symbol=:μ),
    σ=first_partial(p, symbol=:σ),
    θ=first_partial(p, symbol=:θ)
)

function second_partial(p::Policy; symbol::Symbol)
    if symbol == :μ
        return p.d2g_dμ
    elseif symbol == :σ
        return p.d2g_dσ
    elseif symbol == :θ
        return p.d2g_dθ
    else
        error("Unknown symbol. Use :μ, :σ, or :θ")
    end
end
second_partials(p::Policy) = (
    μ=second_partial(p, symbol=:μ),
    σ=second_partial(p, symbol=:σ),
    θ=second_partial(p, symbol=:θ)
)