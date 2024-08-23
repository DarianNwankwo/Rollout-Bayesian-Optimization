using LinearAlgebra

include("lazy_struct.jl")
include("radial_basis_functions.jl")
"""
Here we distinguish between four possible surrogates that can be used in the
optimization process:
    1. RBFsurrogate: a standard RBF surrogate
    2. FantasyRBFsurrogate: a fantasized RBF surrogate
    3. δFRBFsurrogate: a perturbed fantasized RBF surrogate
    4. MultiOutputFantasyRBFsurrogate: a multi-output RBF surrogate
"""

# ------------------------------------------------------------------
# 1. Operations on GP/RBF surrogates
# ------------------------------------------------------------------
struct RBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    c::Vector{Float64}
    σn2::Float64
end

function fit_surrogate(ψ::RBFfun, X::Matrix{Float64}, y::Vector{Float64}; σn2=1e-6)
    d, N = size(X)
    K = eval_KXX(ψ, X, σn2=σn2)
    L = cholesky(Hermitian(K)).L
    c = L'\(L\y)
    return RBFsurrogate(ψ, X, K, L, y, c, σn2)
end

# TODO: Change to a function that updates the object in place
function update_surrogate(s::RBFsurrogate, xnew::Vector{Float64}, ynew::Float64)
    X = hcat(s.X, xnew)
    y = vcat(s.y, ynew)

    # Update covariance matrix and it's cholesky factorization
    KxX = eval_KxX(s.ψ, xnew, s.X)
    K = [s.K  KxX
         KxX' eval_KXX(s.ψ, reshape(xnew, length(xnew), 1), σn2=s.σn2)]
    
    function update_cholesky(K::Matrix{Float64}, L::LowerTriangular{Float64, Matrix{Float64}})
        # Grab entries from update covariance matrix
        n = size(K, 1)
        B = @view K[n:n, 1:n-1]
        C = K[n, n]
        
        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = sqrt(C - first(L21*L21'))
    
        # Update the full factorization
        ufK = zeros(n, n)
        ufK[1:n-1, 1:n-1] .= L
        ufK[n:n, 1:n-1] .= L21
        ufK[n, n] = L22
    
        return LowerTriangular(ufK)
    end

    L = update_cholesky(K, s.L)
    c = L'\(L\y)

    return RBFsurrogate(s.ψ, X, K, L, y, c, s.σn2)
end


function plot1D(s::RBFsurrogate; xmin=-1, xmax=1, npts=100)
    x = range(xmin, stop=xmax, length=npts)
    μ, σ = zeros(npts), zeros(npts)

    for i = 1:npts
        sx = s([x[i]])
        μ[i] = sx.μ
        σ[i] = sx.σ
    end

    p = plot(x, μ, ribbons=2σ, label="μ±2σ (Ground Truth)", grid=false)
    scatter!(s.X[1,:], s.y, label="Observations")
    return p
end

function eval(s::RBFsurrogate, x::Vector{Float64}, ymin::Real)
    sx = LazyStruct()
    set(sx, :s, s)
    set(sx, :x, x)
    set(sx, :ymin, ymin)

    d, N = size(s.X)

    sx.kx = () -> eval_KxX(s.ψ, x, s.X)
    sx.∇kx = () -> eval_∇KxX(s.ψ, x, s.X)

    sx.μ = () -> dot(sx.kx, s.c)
    sx.∇μ = () -> sx.∇kx * s.c
    sx.Hμ = function()
        H = zeros(d, d)
        for j = 1:N
            H += s.c[j] * eval_Hk(s.ψ, x-s.X[:,j])
        end
        return H
    end

    sx.w = () -> s.L'\(s.L\sx.kx)
    sx.Dw = () -> s.L'\(s.L\(sx.∇kx'))
    sx.∇w = () -> sx.Dw'
    sx.σ = () -> sqrt(s.ψ(0) - dot(sx.kx', sx.w))
    sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
    sx.Hσ = function()
        H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
        w = sx.w
        for j = 1:N
            H -= w[j] * eval_Hk(s.ψ, x-s.X[:,j])
        end
        H /= sx.σ
        return H
    end

    sx.z = () -> (ymin - sx.μ) / sx.σ
    sx.∇z = () -> (-sx.∇μ - sx.z*sx.∇σ) / sx.σ
    sx.Hz = () -> Hermitian((-sx.Hμ + (sx.∇μ*sx.∇σ' + sx.∇σ*sx.∇μ')/sx.σ -
        sx.z*(sx.Hσ - 2/sx.σ*sx.∇σ*sx.∇σ')) / sx.σ)

    sx.Φz = () -> Distributions.normcdf(sx.z)
    sx.ϕz = () -> Distributions.normpdf(sx.z)
    sx.g = () -> sx.z * sx.Φz + sx.ϕz

    sx.EI = () -> sx.σ*sx.g
    sx.∇EI = () -> sx.g*sx.∇σ + sx.σ*sx.Φz*sx.∇z
    sx.HEI = () -> Hermitian(sx.Hσ*sx.g +
        sx.Φz*(sx.∇σ*sx.∇z' + sx.∇z*sx.∇σ' + sx.σ*sx.Hz) +
        sx.σ*sx.ϕz*sx.∇z*sx.∇z')

    # Optimizing expected improvement is tricky in regions where EI is
    # exponentially small -- we have to have a reasonable starting
    # point to get going.  For negative z values, we rewrite g(z) = G(-z)
    # in terms of the Mills ratio R(z) = Q(z)/ϕ(z) where Q(z) is the
    # complementary CDF.  Then G(z) = H(z) ϕ(z) where H(z) = 1-zR(z).
    # For sufficiently large R, the Mills ratio can be computed by a
    # generalized continued fraction due to Laplace:
    #   R(z) = 1/z+ 1/z+ 2/z+ 3/z+ ...
    # We rewrite this as
    #   R(z) = W(z)/(z W(z)+1) where W(z) = z + 2/z+ 3/z+ ...
    # Using this definition, we have
    #   H(z) = 1/(1+z W(z))
    #   log G(z) = -log(w+zW(z)) + normlogpdf(z)
    #   [log G(z)]' = -Q(z)/G(z) = -W(z)
    #   [log G(z)]'' = 1 + zW(z) - W(z)^2
    # The continued fraction doesn't converge super-fast, but that is
    # almost surely fine for what we're doing here.  If needed, we could
    # do a similar manipulation to get an optimized rational approximation
    # to W from Cody's 1969 rational approximations to erfc.  Or we could
    # use a less accurate approximation -- the point of getting the tails
    # right is really to give us enough inormation to climb out of the flat
    # regions for EI.

    sx.WQint = function()
        z = -sx.z
        u = z
        for k = 500:-1:2
            u = k/(z+u)
        end
        z + u
    end

    sx.logEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            log(sx.σ) - log(1-z*W) + Distributions.normlogpdf(z)
        else
            log(sx.σ) + log(sx.g)
        end
    end

    sx.∇logEI = function()
        z = sx.z
        if z < -1.0
            sx.∇σ/sx.σ + sx.WQint*sx.∇z
        else
            sx.∇σ/sx.σ + sx.Φz/sx.g*sx.∇z
        end
    end

    sx.HlogEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            HlogG = 1.0-(z+W)*W
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        else
            W = sx.Φz/sx.g
            HlogG = (sx.ϕz-sx.Φz*sx.Φz/sx.g)/sx.g
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        end
    end
    
    return sx
end

eval(s::RBFsurrogate, x::Vector{Float64}) = eval(s, x, minimum(s.y))
(s::RBFsurrogate)(x::Vector{Float64}) = eval(s, x)

# ------------------------------------------------------------------
# 2. Operations on Fantasized GP/RBF surrogates
# ------------------------------------------------------------------
mutable struct FantasyRBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    c::Vector{Float64}
    σn2::Float64
    h::Int64
    known_observed::Int64
    fantasies_observed::Int64
end

mutable struct SmartFantasyRBFsurrogate
    ψ::RBFfun
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    cs::Vector{Vector{Float64}}
    σn2::Float64
    h::Int64
    known_observed::Int64
    fantasies_observed::Int64 
end


function gp_draw(s::Union{RBFsurrogate, FantasyRBFsurrogate}, xloc; stdnormal)
    sx = s(xloc)

    return sx.μ + sx.σ*stdnormal
end

function gp_draw(s::SmartFantasyRBFsurrogate, xloc; stdnormal, fantasy_index)
    sx = s(xloc, fantasy_index=fantasy_index)

    return sx.μ + sx.σ*stdnormal
end

"""
Fitting the fantasy surrogate consist of using the previous surrogate's covariance
factorization and preallocating space for the remaining factorization when fantasy
samples are observed.
"""
function fit_fsurrogate(s::RBFsurrogate, h::Int64)
    d, N = size(s.X)
    K = zeros(N+h+1, N+h+1)
    K[1:N, 1:N] = @view s.K[:,:]
    L = LowerTriangular(zeros(N+h+1, N+h+1))
    L[1:N, 1:N] = @view s.L[:,:]
    X = zeros(d, N+h+1)
    slice = 1:N
    X[:, slice] = @view s.X[:,:] 
    return FantasyRBFsurrogate(
        s.ψ, X, K, L, deepcopy(s.y), deepcopy(s.c), deepcopy(s.σn2), h, N, 0
    )
end

function fit_sfsurrogate(s::RBFsurrogate, h::Int64)
    d, N = size(s.X)
    slice = 1:N
    K = zeros(N+h+1, N+h+1)
    K[slice, slice] = @view s.K[:, :]
    L = LowerTriangular(zeros(N+h+1, N+h+1))
    L[slice, slice] = @view s.L[:, :]
    X = zeros(d, N+h+1)
    X[:, slice] = @view s.X[:, :]
    return SmartFantasyRBFsurrogate(
        s.ψ, X, K, L, deepcopy(s.y), [deepcopy(s.c)], deepcopy(s.σn2), h, N, 0
    )
end

function update_fsurrogate!(fs::FantasyRBFsurrogate, xnew::Vector{Float64}, ynew::Float64)
    @assert fs.fantasies_observed < fs.h + 1 "All fantasies have been observed!"
    update_ndx = fs.known_observed + fs.fantasies_observed + 1
    # We can use the same logic here for preallocating space for X
    fs.X[:, update_ndx] = xnew
    fs.y = vcat(fs.y, ynew)

    # Update covariance matrix and it's cholesky factorization
    KxX = eval_KxX(fs.ψ, xnew, fs.X[:, 1:update_ndx-1])
    fs.K[update_ndx, 1:update_ndx-1] = KxX
    fs.K[1:update_ndx-1, update_ndx] = KxX'
    fs.K[update_ndx, update_ndx] = first(eval_KXX(fs.ψ, reshape(xnew, length(xnew), 1), σn2=fs.σn2))
    
    function update_cholesky(K::Matrix{Float64}, L::Matrix{Float64})
        # Grab entries from update covariance matrix
        n = size(K, 1)
        B = @view K[n:n, 1:n-1]
        C = K[n, n]
        
        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = sqrt(C - first(L21*L21'))
    
        return L21, L22
    end

    L21, L22 = update_cholesky(
        fs.K[1:update_ndx, 1:update_ndx],
        fs.L[1:update_ndx-1, 1:update_ndx-1]
    )
    fs.L[update_ndx, 1:update_ndx-1] = L21
    fs.L[update_ndx, update_ndx] = L22
    
    L = @view fs.L[1:update_ndx, 1:update_ndx] 
    fs.c = L'\(L\fs.y)
    fs.fantasies_observed += 1    

    return nothing
end

function update_sfsurrogate!(fs::SmartFantasyRBFsurrogate, xnew::Vector{Float64}, ynew::Float64)
    @assert fs.fantasies_observed < fs.h + 1 "All fantasies have been observed!"
    update_ndx = fs.known_observed + fs.fantasies_observed + 1
    # We can use the same logic here for preallocating space for X
    fs.X[:, update_ndx] = xnew
    fs.y = vcat(fs.y, ynew)

    # Update covariance matrix and it's cholesky factorization
    KxX = eval_KxX(fs.ψ, xnew, fs.X[:, 1:update_ndx-1])
    fs.K[update_ndx, 1:update_ndx-1] = KxX
    fs.K[1:update_ndx-1, update_ndx] = KxX'
    fs.K[update_ndx, update_ndx] = first(eval_KXX(fs.ψ, reshape(xnew, length(xnew), 1), σn2=fs.σn2))
    
    function update_cholesky(K::Matrix{Float64}, L::Matrix{Float64})
        # Grab entries from update covariance matrix
        n = size(K, 1)
        B = @view K[n:n, 1:n-1]
        C = K[n, n]
        
        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = sqrt(C - first(L21*L21'))
    
        return L21, L22
    end

    L21, L22 = update_cholesky(
        fs.K[1:update_ndx, 1:update_ndx],
        fs.L[1:update_ndx-1, 1:update_ndx-1]
    )
    fs.L[update_ndx, 1:update_ndx-1] = L21
    fs.L[update_ndx, update_ndx] = L22
    
    L = @view fs.L[1:update_ndx, 1:update_ndx]
    c = L'\(L\fs.y)
    push!(fs.cs, c)
    fs.fantasies_observed += 1    

    return nothing
end

function eval(fs::FantasyRBFsurrogate, x::Vector{Float64}, ymin::Real)
    sx = LazyStruct()
    set(sx, :fs, fs)
    set(sx, :x, x)
    set(sx, :ymin, ymin)

    d, N = size(fs.X)
    slice = 1:fs.known_observed + fs.fantasies_observed

    sx.kx = () -> eval_KxX(fs.ψ, x, fs.X[:, slice])
    sx.∇kx = () -> eval_∇KxX(fs.ψ, x, fs.X[:, slice])

    sx.μ = () -> dot(sx.kx, fs.c)
    sx.∇μ = () -> sx.∇kx * fs.c
    sx.Hμ = function()
        H = zeros(d, d)
        # for j = 1:N
        for j = slice
            H += fs.c[j] * eval_Hk(fs.ψ, x-fs.X[:,j])
        end
        return H
    end

    sx.w = () -> fs.L[slice, slice]'\(fs.L[slice, slice]\sx.kx)
    sx.Dw = () -> fs.L[slice, slice]'\(fs.L[slice, slice]\(sx.∇kx'))
    sx.∇w = () -> sx.Dw'
    sx.σ = () -> sqrt(fs.ψ(0) - dot(sx.kx', sx.w))
    sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
    sx.Hσ = function()
        H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
        w = sx.w
        # for j = 1:N
        for j = slice
            H -= w[j] * eval_Hk(fs.ψ, x-fs.X[:,j])
        end
        H /= sx.σ
        return H
    end

    sx.z = () -> (ymin - sx.μ) / sx.σ
    sx.∇z = () -> (-sx.∇μ - sx.z*sx.∇σ) / sx.σ
    sx.Hz = () -> Hermitian((-sx.Hμ + (sx.∇μ*sx.∇σ' + sx.∇σ*sx.∇μ')/sx.σ -
        sx.z*(sx.Hσ - 2/sx.σ*sx.∇σ*sx.∇σ')) / sx.σ)

    sx.Φz = () -> Distributions.normcdf(sx.z)
    sx.ϕz = () -> Distributions.normpdf(sx.z)
    sx.g = () -> sx.z * sx.Φz + sx.ϕz

    sx.α = () -> sx.μ  + sx.θ * sx.σ
    sx.EI = () -> sx.σ*sx.g
    sx.∇EI = () -> sx.g*sx.∇σ + sx.σ*sx.Φz*sx.∇z
    sx.HEI = () -> Hermitian(sx.Hσ*sx.g +
        sx.Φz*(sx.∇σ*sx.∇z' + sx.∇z*sx.∇σ' + sx.σ*sx.Hz) +
        sx.σ*sx.ϕz*sx.∇z*sx.∇z')

    # Optimizing expected improvement is tricky in regions where EI is
    # exponentially small -- we have to have a reasonable starting
    # point to get going.  For negative z values, we rewrite g(z) = G(-z)
    # in terms of the Mills ratio R(z) = Q(z)/ϕ(z) where Q(z) is the
    # complementary CDF.  Then G(z) = H(z) ϕ(z) where H(z) = 1-zR(z).
    # For sufficiently large R, the Mills ratio can be computed by a
    # generalized continued fraction due to Laplace:
    #   R(z) = 1/z+ 1/z+ 2/z+ 3/z+ ...
    # We rewrite this as
    #   R(z) = W(z)/(z W(z)+1) where W(z) = z + 2/z+ 3/z+ ...
    # Using this definition, we have
    #   H(z) = 1/(1+z W(z))
    #   log G(z) = -log(w+zW(z)) + normlogpdf(z)
    #   [log G(z)]' = -Q(z)/G(z) = -W(z)
    #   [log G(z)]'' = 1 + zW(z) - W(z)^2
    # The continued fraction doesn't converge super-fast, but that is
    # almost surely fine for what we're doing here.  If needed, we could
    # do a similar manipulation to get an optimized rational approximation
    # to W from Cody's 1969 rational approximations to erfc.  Or we could
    # use a less accurate approximation -- the point of getting the tails
    # right is really to give us enough inormation to climb out of the flat
    # regions for EI.

    sx.WQint = function()
        z = -sx.z
        u = z
        for k = 500:-1:2
            u = k/(z+u)
        end
        z + u
    end

    sx.logEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            log(sx.σ) - log(1-z*W) + Distributions.normlogpdf(z)
        else
            log(sx.σ) + log(sx.g)
        end
    end

    sx.∇logEI = function()
        z = sx.z
        if z < -1.0
            sx.∇σ/sx.σ + sx.WQint*sx.∇z
        else
            sx.∇σ/sx.σ + sx.Φz/sx.g*sx.∇z
        end
    end

    sx.HlogEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            HlogG = 1.0-(z+W)*W
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        else
            W = sx.Φz/sx.g
            HlogG = (sx.ϕz-sx.Φz*sx.Φz/sx.g)/sx.g
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        end
    end
    
    return sx
end

eval(fs::FantasyRBFsurrogate, x::Vector{Float64}) = eval(fs, x, minimum(fs.y))
(fs::FantasyRBFsurrogate)(x::Vector{Float64}) = eval(fs, x)

function eval(fs::SmartFantasyRBFsurrogate, x::Vector{Float64}, ymin::Real, fantasy_index::Int64)
    sx = LazyStruct()
    set(sx, :fs, fs)
    set(sx, :x, x)
    set(sx, :ymin, ymin)
    set(sx, :fantasy_index, fantasy_index)

    d, N = size(fs.X)
    ZERO_BASED_OFFSET = 1
    FANTASY_BASED_OFFSET = 1
    TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET
    slice = 1:fs.known_observed + fantasy_index + FANTASY_BASED_OFFSET

    sx.kx = () -> eval_KxX(fs.ψ, x, (@view fs.X[:, slice]))
    sx.∇kx = () -> eval_∇KxX(fs.ψ, x, (@view fs.X[:, slice]))

    sx.μ = () -> dot(sx.kx, fs.cs[fantasy_index + TOTAL_OFFSET])
    sx.∇μ = () -> sx.∇kx * fs.cs[fantasy_index + TOTAL_OFFSET]
    sx.Hμ = function()
        H = zeros(d, d)
        # for j = 1:N
        for j = slice
            H += fs.cs[fantasy_index + TOTAL_OFFSET][j] * eval_Hk(fs.ψ, x-fs.X[:,j])
        end
        return H
    end

    sx.w = () -> fs.L[slice, slice]'\(fs.L[slice, slice]\sx.kx)
    sx.Dw = () -> fs.L[slice, slice]'\(fs.L[slice, slice]\(sx.∇kx'))
    sx.∇w = () -> sx.Dw'
    sx.σ = () -> sqrt(fs.ψ(0) - dot(sx.kx', sx.w))
    sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
    sx.Hσ = function()
        H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
        w = sx.w
        # for j = 1:N
        for j = slice
            H -= w[j] * eval_Hk(fs.ψ, x-fs.X[:,j])
        end
        H /= sx.σ
        return H
    end

    sx.z = () -> (ymin - sx.μ) / sx.σ
    sx.∇z = () -> (-sx.∇μ - sx.z*sx.∇σ) / sx.σ
    sx.Hz = () -> Hermitian((-sx.Hμ + (sx.∇μ*sx.∇σ' + sx.∇σ*sx.∇μ')/sx.σ -
        sx.z*(sx.Hσ - 2/sx.σ*sx.∇σ*sx.∇σ')) / sx.σ)

    sx.Φz = () -> Distributions.normcdf(sx.z)
    sx.ϕz = () -> Distributions.normpdf(sx.z)
    sx.g = () -> sx.z * sx.Φz + sx.ϕz

    sx.EI = () -> sx.σ*sx.g
    sx.∇EI = () -> sx.g*sx.∇σ + sx.σ*sx.Φz*sx.∇z
    sx.HEI = () -> Hermitian(sx.Hσ*sx.g +
        sx.Φz*(sx.∇σ*sx.∇z' + sx.∇z*sx.∇σ' + sx.σ*sx.Hz) +
        sx.σ*sx.ϕz*sx.∇z*sx.∇z')

    return sx
end

eval(fs::SmartFantasyRBFsurrogate, x::Vector{Float64}, fantasy_index::Int64) = eval(fs, x, minimum(fs.y), fantasy_index)
(fs::SmartFantasyRBFsurrogate)(x::Vector{Float64}; fantasy_index::Int64) = eval(fs, x, fantasy_index)

function plot1D(s::FantasyRBFsurrogate; xmin=-1, xmax=1, npts=100)
    x = range(xmin, stop=xmax, length=npts)
    μ, σ = zeros(npts), zeros(npts)

    for i = 1:npts
        sx = s([x[i]])
        μ[i] = sx.μ
        σ[i] = sx.σ
    end

    p = plot(x, μ, ribbons=2σ, label="μ±2σ (Fantasy RBF surrogate)")
    scatter!(s.X[1,:], s.y, label="Observations")
    return p
end

function get_known_and_fantasy_counts(fs::SmartFantasyRBFsurrogate)
    return (fs.known_observed, fs.fantasies_observed)
end

function get_active_locations(fs::SmartFantasyRBFsurrogate, fantasy_index::Int64)
    return fs.X[:, 1:fs.known_observed + fantasy_index + 1]
end

function get_active_cholesky_factor(fs::SmartFantasyRBFsurrogate, fantasy_index::Int64)
    slice = 1:fs.known_observed + fantasy_index + 1
    return fs.L[slice, slice]
end

mutable struct SpatialPerturbationSurrogate
    fs::SmartFantasyRBFsurrogate
    X::Matrix{Float64}
    max_fantasized_step::Int64
end

function construct_perturbation_matrix(fs::SmartFantasyRBFsurrogate, fantasy_index)
    known_observed, _ = get_known_and_fantasy_counts(fs)
    d, N = size(fs.X)
    δX = zeros(d, known_observed + fantasy_index + 1)
    return δX
end

# Fantasy index is up to, but not including. We also start from 0.
# We should fit once and perturb an arbitrary amount of times
function fit_spatial_perturbation_surrogate(fs::SmartFantasyRBFsurrogate, max_fantasized_step::Int64)
    δX = construct_perturbation_matrix(fs, max_fantasized_step)
    return SpatialPerturbationSurrogate(fs, δX, max_fantasized_step)
end

function eval(δs::SpatialPerturbationSurrogate, sx; δx::Vector{Float64}, current_step::Int64)
    # @assert 0 <= current_step "Can only perturb fantasized locations"
    # @assert current_step <= δs.max_fantasized_step "Attempting to perturb an observation beyong our trajectory"
    δsx = LazyStruct()
    set(δsx, :sx, sx)

    fs = δs.fs
    x = sx.x
    X = get_active_locations(fs, δs.max_fantasized_step)
    # Set's the desired location's perturbation while keeping every other location constant
    δs.X[:,:] .= 0.
    δs.X[:, fs.known_observed + current_step + 1] .= δx
    # Since we maintain the first set of learned coefficients, we need to offset by 1 to grab
    # the first fantasized pair
    FANTASY_BASED_OFFSET = 1
    # Our fantasy logic assumes the current step can take on the value 0, so we need to offset by
    # 1 again to account for this
    ZERO_BASED_OFFSET = 1
    TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET

    δsx.K = () -> eval_δKXX(fs.ψ, get_active_locations(fs, δs.max_fantasized_step), δs.X)
    δsx.L = () -> get_active_cholesky_factor(fs, δs.max_fantasized_step)
    δsx.c = () -> -(δsx.L' \ (δsx.L \ (δsx.K*fs.cs[δs.max_fantasized_step + TOTAL_OFFSET])))

    δsx.kx = () -> eval_δKxX(fs.ψ, x, X, δs.X)
    δsx.∇kx = () -> eval_δ∇KxX(fs.ψ, x, X, δs.X)

    δsx.μ = () -> δsx.kx'*fs.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.kx'*δsx.c
    δsx.∇μ = () -> δsx.∇kx*fs.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.∇kx*δsx.c

    δsx.σ = () -> (-2*δsx.kx'*sx.w + sx.w'*(δsx.K*sx.w)) / (2*sx.σ)
    δsx.∇σ = () -> (sx.∇w*(δsx.K*sx.w) - δsx.∇kx*sx.w - sx.∇w*δsx.kx - δsx.σ*sx.∇σ) / sx.σ

    δsx.z = () -> (-δsx.μ - δsx.σ*sx.z) / sx.σ
    δsx.∇z = () -> (δsx.∇μ - sx.∇σ*δsx.z - δsx.∇σ*sx.z - δsx.σ*sx.∇z) / sx.σ

    δsx.EI = () -> δsx.σ*sx.g + sx.σ*sx.Φz*δsx.z
    δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(sx.∇σ*δsx.z + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.σ*sx.ϕz*δsx.z*sx.∇z

    return δsx
end

mutable struct DataPerturbationSurrogate
    fs::SmartFantasyRBFsurrogate
    X::Matrix{Float64}
    max_fantasized_step::Int64
end

function fit_data_perturbation_surrogate(fs::SmartFantasyRBFsurrogate, max_fantasized_step::Int64)
    δX = construct_perturbation_matrix(fs, max_fantasized_step)
    return DataPerturbationSurrogate(fs, δX, max_fantasized_step)
end

function fit_perturbation_surrogates(fs::SmartFantasyRBFsurrogate, max_fantasized_step::Int64)
    dp_sur = fit_data_perturbation_surrogate(fs, max_fantasized_step)
    sp_sur = fit_spatial_perturbation_surrogate(fs, max_fantasized_step)

    return (dp_sur, sp_sur)
end

function eval(δs::DataPerturbationSurrogate, sx; δx::Vector{Float64}, current_step::Int64)
    # @assert 0 <= current_step "Can only perturb fantasized locations"
    # @assert current_step <= δs.max_fantasized_step "Attempting to perturb an observation beyong our trajectory"
    δsx = LazyStruct()
    set(δsx, :sx, sx)

    fs = δs.fs
    x = sx.x
    X = get_active_locations(fs, δs.max_fantasized_step)
    # Set's the desired location's perturbation while keeping every other location constant
    δs.X[:,:] .= 0.
    δs.X[:, fs.known_observed + current_step + 1] .= δx
    # Since we maintain the first set of learned coefficients, we need to offset by 1 to grab
    # the first fantasized pair
    FANTASY_BASED_OFFSET = 1
    # Our fantasy logic assumes the current step can take on the value 0, so we need to offset by
    # 1 again to account for this
    ZERO_BASED_OFFSET = 1
    TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET

    δsx.L = () -> get_active_cholesky_factor(fs, δs.max_fantasized_step)
    δsx.y = function()
        ys = zeros(fs.known_observed + δs.max_fantasized_step + 1)
        # Grab the gradient of the mean field at the current_step location
        # current_∇y = δs.fs(X[:, fs.known_observed + current_step + 1], fantasy_index=current_step).∇μ
        current_∇y = sx.∇μ
        ys[fs.known_observed + current_step + 1] = current_∇y' * δs.X[:, fs.known_observed + current_step + 1]
        # ys[fs.known_observed + current_step + 1] = ∇y' * δs.X[:, fs.known_observed + current_step + 1]

        return ys
    end
    δsx.ymin = function()
        maximum_consideration = fs.known_observed + δs.max_fantasized_step + 1
        ymin, j_ymin = findmin(fs.y[1:maximum_consideration])
        δymin = δsx.y[j_ymin]

        return δymin
    end

    δsx.c = () -> δsx.L' \ (δsx.L \ δsx.y)

    δsx.μ = () -> sx.kx' * δsx.c
    δsx.∇μ = () -> sx.∇kx * δsx.c

    δsx.σ = () -> 0.
    δsx.∇σ = () -> zeros(size(fs.X, 1))

    δsx.z = () -> (δsx.ymin - δsx.μ) / sx.σ
    δsx.∇z = () -> -(sx.∇σ*δsx.z - δsx.∇μ) / sx.σ

    δsx.EI = () -> sx.σ*sx.Φz*δsx.z
    δsx.∇EI = () -> sx.∇σ*sx.Φz*δsx.z + sx.σ*(sx.ϕz*δsx.z*sx.∇z + sx.Φz*δsx.∇z)

    return δsx
end

# ------------------------------------------------------------------
# 3. Operations on GP/RBF surrogate derivatives wrt node positions
# ------------------------------------------------------------------
mutable struct δRBFsurrogate
    fs::FantasyRBFsurrogate
    X::Matrix{Float64}
    K::Matrix{Float64}
    y::Vector{Float64}
    c::Vector{Float64}
end

function fit_δsurrogate(fs::FantasyRBFsurrogate, δX::Matrix{Float64}, ∇ys::Vector{Vector{Float64}})
    slice = 1:fs.known_observed + fs.fantasies_observed # fs.known_observed == N
    d, N = size(fs.X[:, slice])
    δK = zeros(N+fs.h+1, N+fs.h+1)
    δK[1:N, 1:N] = eval_δKXX(fs.ψ, fs.X[:, slice], δX)
    δy = [dot(∇ys[j], δX[:,j]) for j=1:N]
    δc = fs.L[slice, slice]' \ (fs.L[slice, slice] \ (δy - δK[slice, slice]*fs.c))
    # δXpreallocate = zeros(d, N+fs.h+1)
    # δXpreallocate[:, slice] = δX
    # return δRBFsurrogate(fs, δXpreallocate, δK, δy, δc)
    return δRBFsurrogate(fs, δX, δK, δy, δc)
end


function update_δsurrogate!(δs::δRBFsurrogate, ufs::FantasyRBFsurrogate, δx::Vector{Float64}, ∇y::Vector{Float64})
    update_ndx = ufs.known_observed + ufs.fantasies_observed
    d, N = size(ufs.X, 1), ufs.known_observed + ufs.fantasies_observed
    # Recover the original perturbation vector and add new perturbation
    δs.y = vcat(δs.y, dot(∇y, δx))

    # Update the perturbation to the covariance matrix
    δs.X[:, update_ndx] = δx
    δKxX = eval_δKxX(ufs.ψ, ufs.X[:, update_ndx], ufs.X[:, 1:update_ndx], δs.X[:, 1:update_ndx])
    δKxX, δKxx = δKxX[1:end-1], δKxX[end]

    # Update the corresponding slice of the preallocated covariance matrix
    δs.K[update_ndx, 1:update_ndx-1] = δKxX
    δs.K[1:update_ndx-1, update_ndx] = δKxX'
    δs.K[update_ndx, update_ndx] = δKxx

    slice = 1:update_ndx
    # δs.K[slice, slice]*ufs.c
    # Print the size of of the left hand side of the linear solve and the right hand side using
    # string interpolation
    δs.c = ufs.L[slice, slice]\(δs.y - δs.K[slice, slice]*ufs.c)

    return nothing
end

function eval(δs :: δRBFsurrogate, sx, δymin)
    δsx = LazyStruct()
    set(δsx, :sx, sx)
    set(δsx, :δymin, δymin)

    fs = δs.fs
    x = sx.x
    d, N = size(fs.X, 1), fs.known_observed + fs.fantasies_observed
    slice = 1:N

    δsx.kx  = () -> eval_δKxX(fs.ψ, x, fs.X[:, slice], δs.X[:, slice])
    δsx.∇kx = () -> eval_δ∇KxX(fs.ψ, x, fs.X[:, slice], δs.X[:, slice])

    δsx.μ  = () -> δsx.kx'*fs.c + sx.kx'*δs.c
    δsx.∇μ = () -> δsx.∇kx*fs.c + sx.∇kx*δs.c

    δsx.σ  = () -> (-2*δsx.kx'*sx.w + sx.w'*(δs.K[slice, slice]*sx.w)) / (2*sx.σ)
    δsx.∇σ = () -> (-δsx.∇kx*sx.w - sx.∇w*δsx.kx + sx.∇w*(δs.K[slice, slice]*sx.w)-δsx.σ*sx.∇σ)/sx.σ

    δsx.z  = () -> (δymin-δsx.μ-sx.z*δsx.σ)/sx.σ
    δsx.∇z = () -> (δsx.∇μ - sx.∇z*δsx.σ - sx.z*δsx.∇σ - δsx.z*sx.∇σ)/sx.σ

    δsx.EI  = () -> sx.g*δsx.σ + sx.σ*sx.Φz*δsx.z
    # δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(δsx.z*sx.∇σ + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.ϕz*δsx.z*sx.∇z
    δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(δsx.z*sx.∇σ + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.ϕz*δsx.z*sx.∇z*sx.σ

    δsx
end


function eval(δs :: δRBFsurrogate, sx)
    ymin, j_ymin = findmin(δs.fs.y)
    δymin = δs.y[j_ymin]
    return eval(δs, sx, δymin)
end

(δs :: δRBFsurrogate)(sx) = eval(δs, sx)

# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function log_likelihood(s :: RBFsurrogate)
    n = size(s.X)[2]
    # -s.y'*s.c/2 - sum(log.(diag(s.fK.L))) - n*log(2π)/2
    -s.y'*s.c/2 - sum(log.(diag(s.L))) - n*log(2π)/2
end


function δlog_likelihood(s :: RBFsurrogate, δθ)
    δK = eval_Dθ_KXX(s.ψ, s.X, δθ)
    (s.c'*δK*s.c - tr(s.fK\δK))/2
end


function ∇log_likelihood(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood(s, δθ)
    end
    ∇L
end

function log_likelihood_v(s :: RBFsurrogate)
    n = size(s.X)[2]
    α = s.y'*s.c/n
    -n/2*(1.0 + log(α) + log(2π)) - sum(log.(diag(s.fK.L)))
end


function δlog_likelihood_v(s :: RBFsurrogate, δθ)
    n = size(s.X)[2]
    c = s.c
    y = s.y
    δK = eval_Dθ_KXX(s.ψ, s.X, δθ)
    n/2*(c'*δK*c)/(c'*y) - tr(s.fK\δK)/2
end


function ∇log_likelihood_v(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood_v(s, δθ)
    end
    ∇L
end

"""
This only optimizes for lengthscale hyperparameter where the lengthscale is the
same in each respective dimension.
"""
function optimize_hypers_optim(s::RBFsurrogate, ψconstructor; max_iterations=100)
    function f(θ)
        ψ = ψconstructor(θ)
        lsur = fit_surrogate(ψ, s.X, s.y; σn2=s.σn2)
        return -log_likelihood(lsur)
    end

    θinit = s.ψ.θ
    lowerbounds = [1e-1]
    upperbounds = [10.]
    res = optimize(f, lowerbounds, upperbounds, θinit, Fminbox(LBFGS()), Optim.Options(iterations=max_iterations))
    lengthscale = Optim.minimizer(res)
    kernel = ψconstructor(lengthscale)
    opt_sur = fit_surrogate(kernel, s.X, s.y; σn2=s.σn2)

    return opt_sur
end

function optimize_hypers(θ, kernel_constructor, X, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end

function optimize_hypers(θ, kernel_constructor, X, sur::RBFsurrogate, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, sur, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, sur, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end


function optimize_hypers_v(θ, kernel_constructor, X, f;
                           Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                           monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)

    Lref = log_likelihood_v(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood_v(s)/Lref
    ∇g(s) = ∇log_likelihood_v(s)/Lref

    θ0, s = tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                   Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                   monitor=monitor)
    α = s.c'*s.y/(size(s.X)[2])
    θ = vcat([α], θ0)
    rbf = kernel_scale(kernel_constructor, θ)
    θ, fit_surrogate(rbf, X, f)
end


function get_minimum(s, lbs, ubs; guesses)
    function predictive_mean(x)
        return s(x).μ
    end

    function grad_predictive_mean!(g, x)
        g[:] = s(x).∇μ
    end
    
    function hessian_predictive_mean!(h, x)
        h .= s(x).Hμ
    end

    final_minimizer = (guesses[:, 1], Inf)
    for j in 1:size(guesses, 2)
        guess = guesses[:, j]
        df = TwiceDifferentiable(predictive_mean, grad_predictive_mean!, hessian_predictive_mean!, guess)
        dfc = TwiceDifferentiableConstraints(lbs, ubs)
        result = optimize(
            df, dfc, guess, IPNewton(),
            Optim.Options(x_tol=1e-3, f_tol=1e-3, time_limit=10, iterations=100)
        )
        # result = optimize(
        #     predictive_mean, grad_predictive_mean!,
        #     lbs, ubs, guess, Fminbox(LBFGS()), Optim.Options(x_tol=1e-3, f_tol=1e-3)
        # )
        cur_minimizer, cur_minimum = Optim.minimizer(result), Optim.minimum(result)

        if cur_minimum < final_minimizer[2]
            final_minimizer = (cur_minimizer, cur_minimum)
        end
    end

    return final_minimizer
end