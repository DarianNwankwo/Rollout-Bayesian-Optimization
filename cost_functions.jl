abstract type AbstractCostFunction end
abstract type KnownCostFunction <: AbstractCostFunction end
abstract type UnknownCostFunction <: AbstractCostFunction end

struct NonUniformCost{F <: Function, DF <: Function, HF <: Function} <: KnownCostFunction
    f::F
    ∇f::DF
    Hf::HF
end

function NonUniformCost(f::Function)
    ∇f(x) = ForwardDiff.gradient(x -> f(x), x)
    Hf(x) = ForwardDiff.hessian(x -> f(x), x)

    return NonUniformCost(f, ∇f, Hf)
end

(nuc::NonUniformCost)(x::AbstractVector) = nuc.f(x)
gradient(nuc::NonUniformCost) = nuc.∇f
hessian(nuc::NonUniformCost) = nuc.Hf

struct UniformCost{F <: Function, DF <: Function, HF <: Function} <: KnownCostFunction
    f::F
    ∇f::DF
    Hf::HF
end

function UniformCost(f::Function)
    ∇f(x) = zeros(length(x))
    Hf(x) = zeros(length(x), length(x))

    return UniformCost(f, ∇f, Hf)
end
UniformCost(n::Real = 1) = UniformCost(x -> n)

(uc::UniformCost)(x::AbstractVector) = uc.f(x)
gradient(uc::UniformCost) = uc.∇f
hessian(uc::UniformCost) = uc.Hf

UnitCost() = UniformCost(x -> 1)


"""
We can use the mechanics for our GP here to model situations where the cost is unknown
"""
struct GaussianProcessCost <: UnknownCostFunction
end