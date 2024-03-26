#----------------------------------------------------
# TODOS:
# - Add support for kernel transformations
#----------------------------------------------------
# using ForwardDiff
# using LinearAlgebra

"""
A struct representing a radial basis function. The struct contains the
hyperparameter vector, the kernel function, the derivative of the kernel
function with respect to the distance, the second derivative of the kernel
function with respect to the distance, and the gradient of the kernel function
with respect to the hyperparameter vector.
"""
struct RBFfun
    θ::Vector{Float64}         # Hyperparameter vector
    ψ::Function                # Radial basis function
    Dρ_ψ::Function             # Derivative of the RBF wrt ρ
    Dρρ_ψ::Function            # Second derivative
    ∇θ_ψ::Function             # Gradient with respect to hypers
end


(rbf::RBFfun)(ρ) = rbf.ψ(ρ)

"""
A generic way of constructing a radial basis functions with a given kernel
using automatic differentiation. The kernel should be a function of the
distance ρ and the hyperparameter vector θ. The kernel should be
differentiable with respect to ρ and θ. The kernel should be positive
definite.
"""

function kernel_generic(k, θ)
    ψ(ρ) = k(ρ, θ)
    Dρ_ψ(ρ) = ForwardDiff.derivative(ψ, ρ)
    Dρρ_ψ(ρ) = ForwardDiff.derivative(Dρ_ψ, ρ)
    ∇θ_ψ(ρ) = ForwardDiff.gradient(θ -> k(ρ, θ), θ)
    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end

function kernel_scale(kfun, θ; kwargs...)
    s = θ[1]
    base_rbf = kfun(θ[2:end]; kwargs...)
    ψ(ρ)     = s * base_rbf.ψ(ρ)
    Dρ_ψ(ρ)  = s * base_rbf.Dρ_ψ(ρ)
    Dρρ_ψ(ρ) = s * base_rbf.Dρρ_ψ(ρ)
    ∇θ_ψ(ρ)  = vcat([base_rbf.ψ(ρ)], s * base_rbf.∇θ_ψ(ρ))
    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end

function kernel_matern52(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(5.0) / l
        s = c*ρ
        return (1+s*(1+s/3.0))*exp(-s)
    end
    return kernel_generic(k, θ)
end

function kernel_matern32(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = sqrt(3.0) / l
        s = c*ρ
        return (1+s)*exp(-s)
    end
    return kernel_generic(k, θ)
end

function kernel_matern12(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        c = 1.0 / l
        s = c*ρ
        return exp(-s)
    end
    return kernel_generic(k, θ)
end

function kernel_SE(θ=[1.])
    function k(ρ, θ)
        l = θ[1]
        return exp(-ρ^2/(2*l^2))
    end
    return kernel_generic(k, θ)
end

"""
Given a radial basis function and the distances between two points, evaluate
the kernel.
"""

function eval_k(rbf::RBFfun, r)
    ρ = norm(r)
    return rbf(ρ)
end

"""
Given a radial basis function and the distances between two points, evaluate
the gradient of the kernel.
"""

function eval_∇k(rbf::RBFfun, r)
    ρ = norm(r)
    if ρ == 0
        return 0*r
    end
    ∇ρ = r/ρ
    return rbf.Dρ_ψ(ρ)*∇ρ
end

"""
Given a radial basis function and the distances between two points, evaluate
the Hessian of the kernel.
"""

function eval_Hk(rbf::RBFfun, r)
    p = norm(r)
    if p > 0
        ∇p = r/p
        Dψr = rbf.Dρ_ψ(p)/p
        D2ψ = rbf.Dρρ_ψ(p)
        return (D2ψ-Dψr)*∇p*∇p' + Dψr*I
    end
    return rbf.Dρρ_ψ(p) * Matrix(I, length(r), length(r))
end


"""
Given a radial basis function and vector representing the pairwise difference
between two sets of observations, evaluate the matrix containing the kernel
evaluation, gradient of the kernel evaluation, and hessian of the kernel
evaluation.
"""

function eval_Dk(rbf::RBFfun, r::Vector{Float64}; D::Int64)
    K = eval_k(rbf, r)
    ∇K = eval_∇k(rbf, r)
    HK = eval_Hk(rbf, r)
    
    return [K   -∇K'
            ∇K -HK]
end

function eval_DKxX(rbf :: RBFfun, x::Vector{Float64},
    X::Matrix{Float64}; D::Int)
    M, N = size(X)
    KxX = eval_Dk(rbf, x-X[:,1], D=D)
    for j = 2:N
        KxX = hcat(
            KxX,
            eval_Dk(rbf, x-X[:,j], D=D)
        )
    end

    return KxX
end

"""
    eval_DKXX(rbf, X, D=D)

Constructs a covariance matrix between observations and gradients
where the block entries are the covariances between observations
and gradient observations

Kij = [k(xi,xj) ∇k(xi,xj)
       ∇k(xi,xj) Hk(xi,xj)]
KXX = [K11 ... K1N
       .   ...  .
       .   ...  .
       KN1 ... KNN]
"""

function eval_DKXX(rbf :: RBFfun, X::Matrix{Float64}; D::Int, σn2=1e-6)
    M, N = size(X)
    nd1 = N*(D+1)
    K = zeros(nd1, nd1)
    r0 = zeros(M)
    ψ0 = eval_Dk(rbf, r0, D=D)
    s(i) = (i-1)*(D+1)+1
    e(i) = s(i)+D

    for i = 1:N
        # Starting indices
        si, ei = s(i), e(i)
        K[si:ei, si:ei] = ψ0
        # Reduce computations by leveraging symmetric structure of
        # covariance matrix
        for j = i+1:N
            # Row remains stationary as columns (j=i+1) vary as a function
            # of the row index (i)
            sj, ej = s(j), e(j)
            Kij = eval_Dk(rbf, X[:,i]-X[:,j], D=D)
            K[si:ei, sj:ej] = Kij
            K[sj:ej, si:ei] = Kij'
        end
    end

    return K + σn2*I
end


"""
Given a radial basis function and a matrix of observations, evaluate the
kernel matrix.
"""

function eval_KXX(rbf::RBFfun, X::Matrix{Float64}; σn2=1e-6)
    d, N = size(X)
    KXX = zeros(N, N)
    ψ0 = rbf(0.0)

    for j = 1:N
        KXX[j,j] = ψ0
        for i = j+1:N
            Kij = rbf(norm(X[:,i]-X[:,j]))
            KXX[i,j] = Kij
            KXX[j,i] = Kij
        end
    end

    return KXX + σn2*I
end

"""
Given a radial basis function, a matrix X and a matrix Y, evaluate the
covariance matrix between the two sets of observations.
"""

function eval_KXY(rbf::RBFfun, X::Matrix{Float64}, Y::Matrix{Float64})
    d, M = size(X)
    d, N = size(Y)
    KXY = zeros(M, N)

    for j = 1:N
        for i = 1:M
            KXY[i,j] = rbf(norm(X[:,i]-Y[:,j]))
        end
    end

    return KXY
end

"""
Given a radial basis function and a matrix of observations, evaluate the
covariance vector between obersations and a test point.
"""

function eval_KxX(rbf::RBFfun, x::Vector{Float64}, X::Matrix{Float64})
    d, N = size(X)
    KxX = zeros(N)
    
    for i = 1:N
        KxX[i] = rbf(norm(x-X[:,i]))
    end

    return KxX
end

"""
Given a radial basis function, a matrix of observations (function evaluations
and gradient evaluations), and a start index of the gradient evaluations, we
computed the mixed covariance matrix. The mixed covariance matrix can be broken
into 4 routines as follows:
MK = [eval_KXX(...)  covmat_gat(...)
      covmat_ga(...) covmat_gg(...)]

"""

function eval_mixed_KXX(rbf::RBFfun, X::Matrix{Float64}; j_∇::Int64, σn2=1e-6)
    Xnograd = X[:, 1:j_∇]
    Xgrad = X[:, j_∇:end]

    """
    Evaluates the covariance between gradient observations and
    all observations. X is expected to contain gradient
    observations; whereas Y is expected to contain all
    observations.
    """
    function covmat_ga(ψ, X, Y)
        m = num_grad_observations = size(X, 2)
        n = num_all_observations = size(Y, 2)
        covmats = []
        
        for i = 1:m
            K = eval_∇k(ψ, X[:, i] - Y[:, 1])
            for j = 2:n
                K = hcat(K, eval_∇k(ψ, X[:, i] - Y[:, j]))
            end
            push!(covmats, K)
        end
        
        
        covmat = covmats[1]
        for i = 2:length(covmats)
            covmat = vcat(covmat, covmats[i])
        end
        return covmat
    end

    """
    Evaluates the covariance between gradient observations and
    all observations. X is expected to contain gradient
    observations; whereas Y is expected to contain all
    observations. Then transposes it.
    """
    function covmat_gat(ψ, X, Y)
        m = num_grad_observations = size(X, 2)
        n = num_all_observations = size(Y, 2)
        covmats = []
        
        for i = 1:m
            K = -eval_∇k(ψ, X[:, i] - Y[:, 1])'
            for j = 2:n
                K = vcat(K, -eval_∇k(ψ, X[:, i] - Y[:, j])')
            end
            push!(covmats, K)
        end
        
        
        covmat = covmats[1]
        for i = 2:length(covmats)
            covmat = hcat(covmat, covmats[i])
        end
        return covmat
    end

    """
    Evaluates the covariance between gradient observations and
    all observations. X is expected to contain gradient
    observations; whereas Y is expected to contain all
    observations. Then transposes it.
    """
    function covmat_gg(ψ, X)
        m = num_grad_observations = size(X, 2)
        covmats = []
        
        for i = 1:m
            K = eval_Hk(ψ, X[:, i] - X[:, 1])
            for j = 2:m
                K = hcat(K, eval_Hk(ψ, X[:, i] - X[:, j]))
            end
            push!(covmats, K)
        end
        
        
        covmat = covmats[1]
        for i = 2:length(covmats)
            covmat = vcat(covmat, covmats[i])
        end
        return covmat
    end

    # Our surrogate is getting the gradient computations wrong. I suspect
    # it is something to do with covmat_gat().
    K = [eval_KXX(rbf, X)           -covmat_gat(rbf, Xgrad, X);
         covmat_ga(rbf, Xgrad, X)   -covmat_gg(rbf, Xgrad)]

    return K + σn2*I
end

"""
Given a radial basis function, an arbitrary point, and a matrix of observations,
evaluate the gradient of the covariance vector between the point and the
observations.
"""
# TODO: If the sign of r is negative, our covariance measure is wrong by a factor of -1.
function eval_∇KxX(rbf::RBFfun, x::Vector{Float64}, X::Matrix{Float64})
    d, N = size(X)
    ∇KxX = zeros(d, N)
    
    for j = 1:N
        r = x-X[:,j]
        ρ = norm(r)
        if ρ > 0
            ∇KxX[:,j] = rbf.Dρ_ψ(ρ)*r/ρ
        end
    end

    return ∇KxX
end

"""
Given a radial basis function, a matrix of observations, the index of the first
gradient observation and a new test location, compute the mixed covariance matrix
containing the covariances of the test locations against known locations
and their gradient covariances.
"""

function eval_mixed_KxX(rbf::RBFfun, X::Matrix{Float64}, x::Vector{Float64};
    j_∇::Int64)
    d, N = size(X)
    Xgrad = X[:, j_∇:end]
    d, M = size(Xgrad)

    Kx∇X = zeros(1, d*M)
    for i = 1:M
        startj, endj = (i-1)*d + 1, i*d
        Kx∇X[1, startj:endj] = eval_∇k(rbf, Xgrad[:,i] - x)'
    end

    K∇xX = zeros(d, N)
    for i = 1:N
        K∇xX[:, i] = eval_∇k(rbf, x - X[:, i])
    end

    K∇x∇X = zeros(d, d*M)
    for i = 1:M
        startj, endj = (i-1)*d + 1, i*d
        K∇x∇X[:, startj:endj] = -eval_Hk(rbf, x - Xgrad[:,i])
    end

    KXX = eval_KxX(rbf, x, X)
    KXX = reshape(KXX, 1, length(KXX))
    K = [KXX   Kx∇X
         K∇xX K∇x∇X]

    return K
end

function eval_mixed_Kxx(rbf::RBFfun, x::Vector{Float64}; σn2=1e-6)
    d = length(x)
    K = zeros(d+1, d+1)

    # Kxx = eval_KXX(rbf, reshape(x, length(x), 1))
    Kxx = eval_k(rbf, 0*x)
    ∇Kx = eval_∇k(rbf, 0*x)
    HKx = eval_Hk(rbf, 0*x)

    K = [Kxx -∇Kx'
         ∇Kx -HKx]

    return K + σn2*I
end

"""
Given a radial basis function, a matrix of observations, and a matrix
of perturbations to the observations, evaluate the covariance matrix.
"""

function eval_δKXX(rbf::RBFfun, X::Matrix{Float64}, δX::Matrix{Float64})
    d, N = size(X)
    δKXX = zeros(N, N)

    for j = 1:N
        for i = j+1:N
            δKij = eval_∇k(rbf, X[:,i]-X[:,j])' * (δX[:,i]-δX[:,j])
            δKXX[i,j] = δKij
            δKXX[j,i] = δKij
        end
    end

    return δKXX
end

"""
Given a radial basis function, a matrix of observations X, a matrix
of observations Y, and matrices representing their respective perturbations,
evaluate the covariance matrix.
"""

function eval_δKXY(rbf::RBFfun, X::Matrix{Float64}, Y::Matrix{Float64},
    δX::Matrix{Float64}, δY::Matrix{Float64})
    d, N = size(X)
    d, M = size(Y)
    δKXY = zeros(N, M)

    for i = 1:N
        for j = 1:M
            δKXY[i,j] = eval_∇k(rbf, X[:,i]-Y[:,j])' * (δX[:,i]-δY[:,j])
        end
    end

    return δKXY
end

"""
Given a radial basis function, a vector representing some arbitrary location,
a matrix of observations, and a matrix of perturbations to the observations,
evaluate the covariance vector formed by perturbations of the kernel
hyperparameters.
"""

function eval_δKxX(rbf::RBFfun, x::Vector{Float64}, X::Matrix{Float64},
    δX::Matrix{Float64})
    d, N = size(X)
    δKxX = zeros(N)

    for j = 1:N
        δKxX[j] = eval_∇k(rbf, x-X[:,j])' * (δX[:,j])
    end

    return δKxX
end

"""
Given a radial basis function, a vector representing some arbitrary location,
a matrix of observations, and a matrix of perturbations to the observations,
evaluate the gradient of the covariance vector formed by perturbations of the
kernel hyperparameters.
"""

function eval_δ∇KxX(rbf::RBFfun, x::Vector{Float64}, X::Matrix{Float64},
    δX::Matrix{Float64})
    d, N = size(X)
    δ∇KxX = zeros(d, N)

    for j = 1:N
        δ∇KxX[:,j] = eval_Hk(rbf, x-X[:,j]) * (-δX[:,j])
    end

    return δ∇KxX
end

"""
Given a radial basis function, a vector representing some arbitrary location,
and a matrix of observations, evaluate the covariance matrix containing the
pairwise covariances (including pairwise covariance gradients) between the
arbitrary location and known observations.
"""

function eval_DKxX(rbf::RBFfun, x::Vector{Float64}, X::Matrix{Float64}; D::Int64)
    M, N = size(X)
    
    KxX = eval_Dk(rbf, x-X[:,1], D=D)
    for j = 2:N
        KxX = hcat(KxX, eval_Dk(rbf, x-X[:,j], D=D))
    end

    return KxX
end

function eval_Dθ_KXX(rbf::RBFfun, X::Matrix{Float64}, δθ::Vector{Float64})
    d, N = size(X)
    δKXX = zeros(N, N)
    δψ0 = rbf.∇θ_ψ(0.0)' * δθ

    for j = 1:N
        δKXX[j,j] = δψ0
        for i = j+1:N
            δKij = rbf.∇θ_ψ(norm(X[:,i]-X[:,j]))' * δθ
            δKXX[i,j] = δKij
            δKXX[j,i] = δKij
        end
    end

    return δKXX
end

