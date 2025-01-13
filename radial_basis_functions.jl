abstract type AbstractKernel end
abstract type StationaryKernel <: AbstractKernel end
abstract type NonStationaryKernel <: AbstractKernel end

get_hyperparameters(sk::AbstractKernel) = sk.θ

struct RadialBasisFunction{T <: Real} <: StationaryKernel
    θ::Vector{T}
    ψ
    Dρ_ψ
    Dρρ_ψ
    ∇θ_ψ
    constructor
end

function Base.show(io::IO, r::RadialBasisFunction{T}) where T
    print(io, "RadialBasisFunction{", T, "}")
end

(rbf::RadialBasisFunction)(ρ) = rbf.ψ(ρ)
derivative(rbf::RadialBasisFunction) = rbf.Dρ_ψ
second_derivative(rbf::RadialBasisFunction) = rbf.Dρρ_ψ
hypersgradient(rbf::RadialBasisFunction) = rbf.∇θ_ψ

function set_hyperparameters!(rbf::RadialBasisFunction, θ::Vector{T}) where T <: Real
    @views begin
        rbf.θ .= θ
    end

    return rbf
end

"""
A generic way of constructing a radial basis functions with a given kernel
using automatic differentiation. The kernel should be a function of the normed
distance ρ and the hyperparameter vector θ. The kernel should be
differentiable with respect to ρ and θ. The kernel should be positive
definite.
"""
# Function barrier to handle ForwardDiff derivative calculations
function compute_derivatives(k::Function, θ::Vector{T}, ψ::Function) where T
    Dρ_ψ(ρ) = ForwardDiff.derivative(ψ, ρ)        # Derivative wrt ρ
    Dρρ_ψ(ρ) = ForwardDiff.derivative(Dρ_ψ, ρ)    # Second derivative wrt ρ
    ∇θ_ψ(ρ) = ForwardDiff.gradient(θ -> k(ρ, θ), θ)  # Gradient wrt θ
    return Dρ_ψ, Dρρ_ψ, ∇θ_ψ  # Return all derivatives
end

# Main constructor with function barrier for derivative calculations
function RadialBasisFunctionGeneric(k::Function, θ::Vector{T}, constructor::Function) where T <: Real
    # Define the radial basis function ψ(ρ)
    ψ(ρ) = k(ρ, θ)
    
    # Use the function barrier to compute the derivatives
    Dρ_ψ, Dρρ_ψ, ∇θ_ψ = compute_derivatives(k, θ, ψ)
    
    # Return the constructed RadialBasisFunction with concrete types
    return RadialBasisFunction(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ, constructor)
end

function Matern52(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(5.0) / l
        s = c*ρ
        return (1+s*(1+s/3.0))*exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ, Matern52)
end

function Matern32(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(3.0) / l
        s = c*ρ
        return (1+s)*exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ, Matern32)
end

function Matern12(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = 1.0 / l
        s = c*ρ
        return exp(-s)
    end
    return RadialBasisFunctionGeneric(k, θ, Matern12)
end

function SquaredExponential(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        return exp(-ρ^2/(2*l^2))
    end
    return RadialBasisFunctionGeneric(k, θ, SquaredExponential)
end

function Periodic(θ=[1., 1.])
    function k(ρ, θ)
        return exp(-2 * sin(pi * ρ / θ[2]) ^ 2 / θ[1] ^ 2)
    end
    return RadialBasisFunctionGeneric(k, θ, Periodic)
end

# struct DotProductFunction{T <: Real} <: NonStationaryKernel
# end

"""
This technically isn't a radial basis function, so some care needs to be had
when writing support for this.
θ[1] ==> offset, θ[2] ==> exponent
"""
# function PolynomialKernel(θ=[0., 1.])
#     function k(ρ, θ)
#         return (ρ^2 + θ[1]) ^ θ[2]
#     end
#     return RadialBasisFunctionGeneric(k, θ, PolynomialKernel)
# end

eval_k(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real = rbf(norm(r))

"""
Given a radial basis function and the distances between two points, evaluate
the gradient of the kernel.
"""

function eval_∇k(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real
    ρ = norm(r)
    if ρ == 0
        return 0*r
    end
    ∇ρ = r/ρ
    return derivative(rbf)(ρ)*∇ρ
end

"""
Given a radial basis function and the distances between two points, evaluate
the Hessian of the kernel.
"""

function eval_Hk(rbf::RadialBasisFunction, r::Vector{T}) where T <: Real
    p = norm(r)
    if p > 0
        ∇p = r/p
        Dψr = derivative(rbf)(p)/p
        D2ψ = second_derivative(rbf)(p)
        return (D2ψ-Dψr)*∇p*∇p' + Dψr*I
    end
    return second_derivative(rbf)(p) * Matrix(I, length(r), length(r))
end

function eval_Dk(rbf::RadialBasisFunction, r::AbstractVector{T}) where T <: Real
    K = eval_k(rbf, r)
    ∇K = eval_∇k(rbf, r)
    HK = eval_Hk(rbf, r)
    
    return [K   -∇K'
            ∇K -HK]
end

function eval_KXX(rbf::RadialBasisFunction, X::AbstractMatrix{T}; σn2::T = 1e-6) where T <: Real
    d, N = size(X)
    KXX = zeros(N, N)
    ψ0 = rbf(0.0)

    @views begin
        for j = 1:N
            KXX[j,j] = ψ0
            for i = j+1:N
                Kij = rbf(norm(X[:,i]-X[:,j]))
                KXX[i,j] = Kij
                KXX[j,i] = Kij
            end
        end
    end

    return KXX + σn2*I
end

function eval_KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    KxX = zeros(N)
    
    @views begin
        for i = 1:N
            KxX[i] = rbf(norm(x-X[:,i]))
        end
    end

    return KxX
end

function eval_∇KxX(rbf::RadialBasisFunction, x::AbstractVector{T}, X::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    ∇KxX = zeros(d, N)
    
    @views begin
        for j = 1:N
            r = x-X[:,j]
            ρ = norm(r)
            if ρ > 0
                ∇KxX[:,j] = rbf.Dρ_ψ(ρ)*r/ρ
            end
        end
    end

    return ∇KxX
end

function eval_δKXX(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δKXX = zeros(N, N)

    @views begin
        for j = 1:N
            for i = j+1:N
                δKij = eval_∇k(rbf, X[:,i]-X[:,j])' * (δX[:,i]-δX[:,j])
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end

function eval_δKxX(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δKxX = zeros(N)

    @views begin
        for j = 1:N
            δKxX[j] = eval_∇k(rbf, x-X[:,j])' * (-δX[:,j])
        end
    end

    return δKxX
end

function eval_δ∇KxX(
    rbf::RadialBasisFunction,
    x::AbstractVector{T},
    X::AbstractMatrix{T},
    δX::AbstractMatrix{T}) where T <: Real
    d, N = size(X)
    δ∇KxX = zeros(d, N)

    @views begin
        for j = 1:N
            δ∇KxX[:,j] = eval_Hk(rbf, x-X[:,j]) * (-δX[:,j])
        end
    end

    return δ∇KxX
end

function eval_Dθ_KXX(
    rbf::RadialBasisFunction,
    X::AbstractMatrix{T},
    δθ::AbstractVector{T}) where T <: Real
    d, N = size(X)
    δKXX = zeros(N, N)
    δψ0 = rbf.∇θ_ψ(0.0)' * δθ

    @views begin
        for j = 1:N
            δKXX[j,j] = δψ0
            for i = j+1:N
                δKij = rbf.∇θ_ψ(norm(X[:,i]-X[:,j]))' * δθ
                δKXX[i,j] = δKij
                δKXX[j,i] = δKij
            end
        end
    end

    return δKXX
end