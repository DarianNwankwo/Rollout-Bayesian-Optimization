abstract type AbstractPolicy end

# TODO: Add support for computing mixed derivatives wrt to x and θ
struct BasePolicy{
    F<:Function,
    DFμ<:Function,
    HFμ<:Function,
    DFσ<:Function,
    HFσ<:Function,
    DFθ<:Function,
    HFθ<:Function,
    MDμθ<:Function,
    MDσθ<:Function
    } <: AbstractPolicy
    g::F             # The function g(μ, σ, θ)
    dg_dμ::DFμ       # Gradient of g with respect to μ
    d2g_dμ::HFμ      # Hessian of g with respect to μ
    dg_dσ::DFσ       # Gradient of g with respect to σ
    d2g_dσ::HFσ      # Hessian of g with respect to σ
    dg_dθ::DFθ       # Gradient of g with respect to θ
    d2g_dθ::HFθ      # Hessian of g with respect to θ
    d2g_dμdθ::MDμθ   # Mixed derivative of g with respect to μ and θ
    d2g_dσdθ::MDσθ   # Mixed derivative of g with respect to σ and θ
    name::String
end

function BasePolicy(g::Function, name::String)
    dg_dμ(μ, σ, θ, sx) = ForwardDiff.derivative(μ -> g(μ, σ, θ, sx), μ)
    d2g_dμ(μ, σ, θ, sx) = ForwardDiff.derivative(μ -> dg_dμ(μ, σ, θ, sx), μ)
    dg_dσ(μ, σ, θ, sx) = ForwardDiff.derivative(σ -> g(μ, σ, θ, sx), σ)
    d2g_dσ(μ, σ, θ, sx) = ForwardDiff.derivative(σ -> dg_dσ(μ, σ, θ, sx), σ)
    dg_dθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> g(μ, σ, θ, sx), θ)
    d2g_dθ(μ, σ, θ, sx) = ForwardDiff.hessian(θ -> g(μ, σ, θ, sx), θ)
    d2g_dμdθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> dg_dμ(μ, σ, θ, sx), θ)
    d2g_dσdθ(μ, σ, θ, sx) = ForwardDiff.gradient(θ -> dg_dσ(μ, σ, θ, sx), θ)

    return BasePolicy(g, dg_dμ, d2g_dμ, dg_dσ, d2g_dσ, dg_dθ, d2g_dθ, d2g_dμdθ, d2g_dσdθ, name)
end

(bp::BasePolicy)(μ::Number, σ::Number, θ::AbstractVector, sx) = bp.g(μ, σ, θ, sx)


function first_partial(p::AbstractPolicy; symbol::Symbol)
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
first_partials(p::AbstractPolicy) = (
    μ=first_partial(p, symbol=:μ),
    σ=first_partial(p, symbol=:σ),
    θ=first_partial(p, symbol=:θ)
)

function second_partial(p::AbstractPolicy; symbol::Symbol)
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
second_partials(p::AbstractPolicy) = (
    μ=second_partial(p, symbol=:μ),
    σ=second_partial(p, symbol=:σ),
    θ=second_partial(p, symbol=:θ)
)

function mixed_partial(p::AbstractPolicy; symbol::Symbol)
    if symbol == :μθ
        p.d2g_dμdθ 
    elseif symbol == :σθ
        p.d2g_dσdθ
    else
        error("Unknown symbol. Use :μθ or :σθ")
    end
end

# Some Common Acquisition Functions
function EI(; σtol=1e-8)
    # TODO: Create a lazy evaluation version of this function. I want to return ei(μ, σ, θ, minimum(y))
    function ei(μ, σ, θ, sx)
        if σ < σtol
            return 0.
        end
        fmini = minimum(sx.y)
        improvement = fmini - μ - θ[1]
        z = improvement / σ
        standard_normal = Distributions.Normal(0, 1)
        
        expected_improvement = improvement*Distributions.cdf(standard_normal, z) + σ*Distributions.pdf(standard_normal, z)
        return expected_improvement
    end

    return BasePolicy(ei, "Expected Improvement")
end


function POI(; σtol=1e-8)
    function poi(μ, σ, θ, sx)
        if σ < σtol
            return 0.0
        end
        fmini = minimum(sx.y)
        improvement = fmini - μ - θ[1]
        z = improvement / σ
        standard_normal = Distributions.Normal(0, 1)

        probability_improvement = Distributions.cdf(standard_normal, z)
        return probability_improvement
    end

    return BasePolicy(poi, "Probability of Improvement")
end

function UCB()
    function ucb(μ, σ, θ, sx)
        return μ + θ[1] * σ
    end

    return BasePolicy(ucb, "Upper Confidence Bound")
end


# Custom string method for BasePolicy
Base.string(bp::AbstractPolicy) = bp.name