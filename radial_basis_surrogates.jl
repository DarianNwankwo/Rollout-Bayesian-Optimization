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
abstract type AbstractSurrogate end
abstract type AbstractFantasySurrogate <: AbstractSurrogate end
abstract type AbstractPerturbationSurrogate <: AbstractSurrogate end

get_known_observations(afs::AbstractFantasySurrogate) = afs.observed
get_total_observations(afs::AbstractFantasySurrogate) = afs.observed + afs.fantasies_observed
get_fantasies_observed(afs::AbstractFantasySurrogate) = afs.fantasies_observed
get_kernel(afs::AbstractSurrogate) = afs.ψ
get_covariates(afs::AbstractSurrogate) = afs.X
get_observations(afs::AbstractSurrogate) = afs.y
get_decision_rule(afs::AbstractSurrogate) = afs.g
get_coefficients(afs::AbstractSurrogate) = afs.c
get_cholesky(as::AbstractSurrogate) = as.L
get_covariance(as::AbstractSurrogate) = as.K


# mutable struct Surrogate{RBF <: StationaryKernel, P <: AbstractDecisionRule} <: AbstractSurrogate
mutable struct Surrogate{RBF <: StationaryKernel} <: AbstractSurrogate
    ψ::RBF
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    c::Vector{Float64}
    σn2::Float64
    g::DecisionRule
    observed::Int
    capacity::Int
end

get_capacity(s::AbstractSurrogate) = s.capacity
increment!(s::AbstractSurrogate) = s.observed += 1
get_observed(s::AbstractSurrogate) = s.observed
is_full(s::AbstractSurrogate) = get_observed(s) == get_capacity(s)
get_active_covariates(s::AbstractSurrogate) = @view get_covariates(s)[:, 1:get_observed(s)]
get_active_cholesky(s::AbstractSurrogate) = @view get_cholesky(s)[1:get_observed(s), 1:get_observed(s)]
get_active_covariance(s::AbstractSurrogate) = @view get_covariance(s)[1:get_observed(s), 1:get_observed(s)]
get_active_observations(s::AbstractSurrogate) = @view get_observations(s)[1:get_observed(s)]
get_active_coefficients(s::AbstractSurrogate) = @view get_coefficients(s)[1:get_observed(s)]

get_active_covariates(s::AbstractFantasySurrogate) = @view get_covariates(s)[:, 1:get_total_observations(s)]
get_active_cholesky(s::AbstractFantasySurrogate) = @view get_cholesky(s)[1:get_total_observations(s), 1:get_total_observations(s)]
get_active_covariance(s::AbstractFantasySurrogate) = @view get_covariance(s)[1:get_total_observations(s), 1:get_total_observations(s)]
get_active_observations(s::AbstractFantasySurrogate) = @view get_observations(s)[1:get_total_observations(s)]
get_active_coefficients(s::AbstractFantasySurrogate) = @view get_coefficients(s)[1:get_total_observations(s)]

function get_fantasy_observations(fs::AbstractFantasySurrogate)
    y = get_observations(fs)
    N = get_known_observations(fs)
    M = get_total_observations(fs)
    return y[N+1:M]
end

# Define the custom show method for Surrogate
# function Base.show(io::IO, s::Surrogate{RBF, P}) where {RBF, P}
function Base.show(io::IO, s::Surrogate{RBF}) where {RBF}
    print(io, "Surrogate{RBF = ")
    show(io, s.ψ)    # Use the show method for RBF
    print(io, "}")
end

get_decision_rule(s::AbstractSurrogate) = s.g
set_decision_rule!(s::AbstractSurrogate, g::DecisionRule) = s.g = g

function Surrogate(
    ψ::RadialBasisFunction,
    X::Matrix{T},
    y::Vector{T};
    capacity::Int = DEFAULT_CAPACITY,
    decision_rule::AbstractDecisionRule = EI(),
    σn2::T = 1e-6) where T <: Real
    @assert length(y) <= capacity "Capacity must be >= number of observations."
    d, N = size(X)

    preallocated_X = zeros(d, capacity)
    preallocated_X[:, 1:N] = X

    preallocated_K = zeros(capacity, capacity)
    preallocated_K[1:N, 1:N] = eval_KXX(ψ, X, σn2=σn2)

    preallocated_L = LowerTriangular(zeros(capacity, capacity))
    preallocated_L[1:N, 1:N] = cholesky(
        Hermitian(
            preallocated_K[1:N, 1:N]
        )
    ).L

    preallocated_c = zeros(capacity)
    preallocated_c[1:N] = preallocated_L[1:N, 1:N]' \ (preallocated_L[1:N, 1:N] \ y)

    preallocated_y = zeros(capacity)
    preallocated_y[1:N] = y

    return Surrogate(
        ψ,
        preallocated_X,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        preallocated_c,
        σn2,
        decision_rule,
        length(y),
        capacity
    )
end

"""
When the kernel is changed, we need to update c, K, and L
"""
function set_kernel!(s::Surrogate, kernel::RadialBasisFunction)
    @views begin
        N = get_observed(s)
        s.ψ = kernel
        s.K[1:N, 1:N] .= eval_KXX(get_kernel(s), get_active_covariates(s), σn2=s.σn2)
        s.L[1:N, 1:N] .= LowerTriangular(
            cholesky(
                Hermitian(s.K[1:N, 1:N])
            ).L
        )
        s.c[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ get_active_observations(s))
    end
end

function resize(s::Surrogate)
    return Surrogate(
        get_kernel(s),
        get_covariates(s),
        get_observations(s),
        capacity=get_capacity(s) * 2,
        decision_rule=get_decision_rule(s)
    )
end

function reset!(s::Surrogate, X::Matrix{T}, y::Vector{T}) where T <: Real
    @views begin
        d, N = size(X)

        s.X[:, 1:N] = X
        s.K[1:N, 1:N] = eval_KXX(get_kernel(s), s.X[:, 1:N], σn2=s.σn2)
        s.L[1:N, 1:N] = LowerTriangular(
            cholesky(
                Hermitian(
                    s.K[1:N, 1:N]
                )
            ).L
        )
        s.c[1:N] = s.L[1:N, 1:N]' \ (s.L[1:N, 1:N] \ y)
        s.y[1:N] = y
        s.observed = length(y)
    end
end

function insert!(s::Surrogate, x::Vector{T}, y::T) where T <: Real
    insert_index = get_observed(s) + 1
    s.X[:, insert_index] = x
    s.y[insert_index] = y
end

function update_covariance!(s::Surrogate, x::Vector{T}, y::T) where T <: Real
    @views begin
        update_index = get_observed(s)
        active_X = get_covariates(s)[:, 1:update_index - 1]
        kernel = get_kernel(s)

        # Update the main diagonal
        s.K[update_index, update_index] = kernel(0.) + s.σn2
        # Update the rows and columns with covariance vector formed from k(x, X)
        s.K[update_index, 1:update_index - 1] = eval_KxX(kernel, x, active_X)'
        s.K[1:update_index - 1, update_index] = s.K[update_index, 1:update_index - 1] 
    end
end

function update_cholesky!(s::Surrogate)
    # Grab entries from update covariance matrix
    @views begin
        n = get_observed(s)
        B = s.K[n:n, 1:n-1]
        C = s.K[n:n, n:n]
        L = s.L[1:n-1, 1:n-1]
        
        # Compute the updated factorizations using schur complements
        L21 = B / L'
        L22 = cholesky(C - L21*L21').L

        # Update the full factorization
        for j in 1:n-1
            s.L[n, j] = L21[1, j]
        end
        s.L[n, n] = L22[1, 1]
    end
end

function update_coefficients!(s::Surrogate)
    update_index = get_observed(s)
    @views begin
        L = s.L[1:update_index, 1:update_index]
        s.c[1:update_index] = L' \ (L \ s.y[1:update_index])
    end
end

function condition!(s::Surrogate, xnew::Vector{T}, ynew::T) where T <: Real
    if is_full(s) s = resize(s) end
    insert!(s, xnew, ynew)
    increment!(s)
    update_covariance!(s, xnew, ynew)
    update_cholesky!(s)
    update_coefficients!(s)
    return s
end

function eval(
    s::Surrogate,
    x::Vector{T},
    θ::Vector{T}) where T<: Real
    @views begin
        sx = LazyStruct()
        set(sx, :s, s)
        set(sx, :x, x)
        set(sx, :θ, θ)

        active_index = get_observed(s)
        X = get_active_covariates(s)
        L = get_active_cholesky(s)
        c = get_active_coefficients(s)
        y = get_active_observations(s)
        kernel = get_kernel(s)

        d, N = size(X)

        sx.kx = () -> eval_KxX(kernel, x, X)
        sx.∇kx = () -> eval_∇KxX(kernel, x, X)

        sx.μ = () -> dot(sx.kx, c)
        sx.∇μ = () -> sx.∇kx * c
        sx.dμ = () -> vcat(sx.μ, sx.∇μ)
        sx.Hμ = function()
            H = zeros(d, d)
            for j = 1:N
                H += c[j] * eval_Hk(kernel, x-X[:,j])
            end
            return H
        end

        sx.w = () -> L'\(L\sx.kx)
        sx.Dw = () -> L'\(L\(sx.∇kx'))
        sx.∇w = () -> sx.Dw'
        sx.σ = () -> sqrt(kernel(0) - dot(sx.kx', sx.w))
        sx.dσ = function()
            kxx = eval_Dk(kernel, zeros(d))
            kxX = [eval_KxX(kernel, x, X)'; eval_∇KxX(kernel, x, X)]
            σx = Symmetric(kxx - kxX * (L' \ (L \ kxX')))
            σx = cholesky(σx).L
            return σx
        end
        sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
        sx.Hσ = function()
            H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
            w = sx.w
            for j = 1:N
                H -= w[j] * eval_Hk(kernel, x-X[:,j])
            end
            H /= sx.σ
            return H
        end

        sx.y = () -> y
        sx.g = () -> get_decision_rule(s)

        sx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dθ = () -> first_partial(sx.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμ = () -> second_partial(sx.s.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσ = () -> second_partial(sx.s.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dθ = () -> second_partial(sx.s.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμdθ = () -> mixed_partial(sx.s.g, symbol=:μθ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσdθ = () -> mixed_partial(sx.s.g, symbol=:σθ)(sx.μ, sx.σ, sx.θ, sx)

        sx.αxθ = () -> s.g(sx.μ, sx.σ, sx.θ, sx)

        # Spatial derivatives
        sx.∇αx = () -> sx.dg_dμ * sx.∇μ + sx.dg_dσ * sx.∇σ
        sx.Hαx = () -> sx.d2g_dμ*sx.∇μ*sx.∇μ' + sx.dg_dμ*sx.Hμ + sx.d2g_dσ*sx.∇σ*sx.∇σ' + sx.dg_dσ*sx.Hσ
       
        # Hyperparameter derivatives
        sx.∇αθ = () -> sx.dg_dθ
        sx.Hαθ = () -> sx.d2g_dθ

        # Mixed partials
        sx.d2α_dσdθ = () -> sx.∇σ * sx.d2g_dσdθ'
        sx.d2α_dμdθ = () -> sx.∇μ * sx.d2g_dμdθ'
        sx.d2α_dxdθ = () -> sx.d2α_dμdθ + sx.d2α_dσdθ
    end

    return sx
end


(s::Surrogate)(x::T, θ::T) where T <: AbstractVector = eval(s, x, θ)
eval(sx) = sx.αxθ
gradient(sx; wrt_hypers=false) = wrt_hypers ? sx.∇αθ : sx.∇αx
hessian(sx; wrt_hypers=false) = wrt_hypers ? sx.Hαθ : sx.Hαx
mixed_partials(sx) = sx.d2α_dxdθ

# mutable struct FantasySurrogate{RBF <: StationaryKernel, P <: AbstractDecisionRule} <: AbstractFantasySurrogate
mutable struct FantasySurrogate{RBF <: StationaryKernel} <: AbstractFantasySurrogate
    ψ::RBF
    X::Matrix{Float64}
    K::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    y::Vector{Float64}
    cs::Vector{Vector{Float64}}
    σn2::Float64
    g::DecisionRule
    h::Int
    observed::Int
    fantasies_observed::Int
    capacity::Int
end

increment!(fs::FantasySurrogate) = fs.fantasies_observed += 1


function Base.show(io::IO, s::FantasySurrogate{RBF}) where {RBF}
    print(io, "FantasySurrogate{RBF = ")
    show(io, s.ψ)    # Use the show method for RBF
    print(io, "}")
end


function FantasySurrogate(s::Surrogate, horizon::Int)
    @views begin
        N = get_observed(s)
        capacity = get_capacity(s)
        d = size(s.X, 1)
        # Preallocate covariance matrix for full BO loop and fantasized trajectories
        preallocated_K = zeros(capacity + horizon + 1, capacity + horizon + 1)
        preallocated_K[1:N, 1:N] = get_covariance(s)[1:N, 1:N]

        # Preallocate cholesky matrix for full BO loop and fantasized trajectories
        preallocated_L = LowerTriangular(zeros(capacity + horizon + 1, capacity + horizon + 1))
        preallocated_L[1:N, 1:N] = get_cholesky(s)[1:N, 1:N]

        # Preallocate covariate matrix
        preallocated_X = zeros(d, capacity + horizon + 1)
        preallocated_X[:, 1:N] = get_covariates(s)[:, 1:N]

        # Preallocated observation vector
        preallocated_y = zeros(capacity + horizon + 1)
        preallocated_y[1:N] = get_observations(s)[1:N]
    end

    return FantasySurrogate(
        get_kernel(s),
        preallocated_X,
        preallocated_K,
        preallocated_L,
        preallocated_y,
        [get_coefficients(s)[1:N]],
        s.σn2,
        get_decision_rule(s),
        horizon,
        N,
        0,
        get_capacity(s)
    )
end

function insert!(fs::FantasySurrogate, x::Vector{T}, y::T) where T <: Real
    insert_index = get_total_observations(fs) + 1
    fs.X[:, insert_index] = x
    fs.y[insert_index] = y
end

function update_covariance!(fs::FantasySurrogate, x::Vector{T}, y::T) where T <: Real
    @views begin
        update_index = get_total_observations(fs)
        active_X = get_covariates(fs)[:, 1:update_index - 1]
        kernel = get_kernel(fs)

        # Update the main diagonal enry
        fs.K[update_index, update_index] = kernel(0.) + fs.σn2
        # Update the rows and columns with covariance vector formed from k(x, X)
        fs.K[update_index, 1:update_index - 1] = eval_KxX(kernel, x, active_X)'
        fs.K[1:update_index - 1, update_index] = fs.K[update_index, 1:update_index - 1]
    end
end

function update_cholesky!(fs::FantasySurrogate)
    @views begin
        n = get_total_observations(fs)
        B = fs.K[n:n, 1:n-1]
        C = fs.K[n:n, n:n]
        L = fs.L[1:n-1, 1:n-1]

        # Compute the updated factorizations using Schur complements
        L21 = B / L'
        L22 = cholesky(C - L21*L21').L

        # Update the full factorization
        for j in 1:n-1
            fs.L[n, j] = L21[1, j]
        end
        fs.L[n, n] = L22[1, 1]
    end
end

function update_coefficients!(fs::FantasySurrogate)
    @views begin
        update_index = get_total_observations(fs)
        L = fs.L[1:update_index, 1:update_index]
        y = fs.y[1:update_index]
        push!(fs.cs, L' \ (L \ y))
    end
end

function condition!(fs::FantasySurrogate, xnew::Vector{T}, ynew::T) where T <: Real
    # The preallocated surrogate is always allocated based on a base surrogate and has similar
    # memory allocations + enough for the fantasized trajectories, hence, we won't need a method
    # for resizing fantasy trajectories.
    insert!(fs, xnew, ynew)
    increment!(fs)
    update_covariance!(fs, xnew, ynew)
    update_cholesky!(fs)
    update_coefficients!(fs)
    return fs
end

"""
We preallocate enough space for arbitrarily long fantasized surrogates, but once we make an observation
and incorporate that new observation into our base surrogate, we need to update a few things on the
fantasized surrogate. Namely,
    - The number of observed true values
    - The covariate matrix
    - The cholesky factorization
    - The observation vector
    - The coefficient  vector
"""
function update!(fs::FantasySurrogate, s::Surrogate)
    @views begin
        # Reset fantasy observed to 0
        fs.fantasies_observed = 0
        N = get_observed(s)
        fs.observed = N
        # Update single column with new covariate
        fs.X[:, N] = get_covariates(s)[:, N]
        # Update row, column and main diagonal with new covariance measures
        KxX_update = get_covariance(s)[N, 1:N]
        fs.K[N, 1:N - 1] = KxX_update[1:N - 1]
        fs.K[1:N - 1, N] = fs.K[N, 1:N - 1]
        fs.K[N, N] = KxX_update[N]
        # Update row, column and main diagonal with new cholesky factorization
        LxX_update = get_cholesky(s)[N, 1:N]
        fs.L[N, 1:N - 1] = LxX_update[1:N - 1]
        fs.L[N, N] = LxX_update[N]
        # Update the coefficient vector
        fs.cs[1] = get_coefficients(s)[1:N] 
    end
end


function reset!(fs::FantasySurrogate)
    N = get_known_observations(fs)
    fs.fantasies_observed = 0
    fs.cs = [fs.cs[1]]
end

function eval(
    fs::FantasySurrogate,
    x::Vector{T},
    θ::Vector{T};
    fantasy_index::Int) where T <: Real
    @assert fantasy_index <= fs.h "Can only observed fantasized locations. Maximum fantasy index is $(fs.h)"
    @views begin
        sx = LazyStruct()
        set(sx, :fs, fs)
        set(sx, :x, x)
        set(sx, :θ, θ)
        set(sx, :fantasy_index, fantasy_index)

        d, N = size(fs.X)
        ZERO_BASED_OFFSET = 1
        FANTASY_BASED_OFFSET = 1
        TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET
        slice = 1:fs.observed + fantasy_index + FANTASY_BASED_OFFSET

        # We use this slice notation to actively recover the subset of data of interest. Eventually
        # this will be all the active covariates, but need not be when arbitrarily evaluating our surrogate
        X = get_covariates(fs)[:, slice]
        L = get_cholesky(fs)[slice, slice]
        c = fs.cs[fantasy_index + TOTAL_OFFSET]
        y = get_observations(fs)[slice]
        kernel = get_kernel(fs)


        sx.kx = () -> eval_KxX(kernel, x, X)
        sx.∇kx = () -> eval_∇KxX(kernel, x, X)

        sx.μ = () -> dot(sx.kx, c)
        sx.∇μ = () -> sx.∇kx * c
        sx.dμ = () -> vcat(sx.μ, sx.∇μ)
        sx.Hμ = function()
            H = zeros(d, d)
            # for j = 1:N
            for j = slice
                H += c[j] * eval_Hk(kernel, x-X[:,j])
            end
            return H
        end

        sx.w = () -> L'\(L\sx.kx)
        sx.Dw = () -> L'\(L\(sx.∇kx'))
        sx.∇w = () -> sx.Dw'
        sx.σ = () -> sqrt(kernel(0.) - dot(sx.kx', sx.w))
        sx.∇σ = () -> -(sx.∇kx * sx.w) / sx.σ
        sx.dσ = function()
            kxx = eval_Dk(kernel, zeros(d))
            kxX = [
                eval_KxX(kernel, x, X)'; 
                eval_∇KxX(kernel, x, X)
            ]
            σx = Symmetric(kxx - kxX * (L' \ (L \ kxX')))
            σx = cholesky(σx).L
            return σx
        end 
        sx.Hσ = function()
            H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
            w = sx.w
            for j = slice
                H -= w[j] * eval_Hk(kernel, x-X[:,j])
            end
            H /= sx.σ
            return H
        end

        sx.y = () -> y
        sx.g = () -> get_decision_rule(fs)

        sx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.dg_dθ = () -> first_partial(sx.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμ = () -> second_partial(sx.g, symbol=:μ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσ = () -> second_partial(sx.g, symbol=:σ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dθ = () -> second_partial(sx.g, symbol=:θ)(sx.μ, sx.σ, sx.θ, sx)

        sx.d2g_dμdθ = () -> mixed_partial(sx.fs.g, symbol=:μθ)(sx.μ, sx.σ, sx.θ, sx)
        sx.d2g_dσdθ = () -> mixed_partial(sx.fs.g, symbol=:σθ)(sx.μ, sx.σ, sx.θ, sx)

        sx.αxθ = () -> sx.g(sx.μ, sx.σ, sx.θ, sx)

        # Spatial derivatives
        sx.∇αx = () -> sx.dg_dμ * sx.∇μ + sx.dg_dσ * sx.∇σ
        sx.Hαx = () -> sx.d2g_dμ*sx.∇μ*sx.∇μ' + sx.dg_dμ*sx.Hμ + sx.d2g_dσ*sx.∇σ*sx.∇σ' + sx.dg_dσ*sx.Hσ
        
        # Hyperparameter derivatives
        sx.∇αθ = () -> sx.dg_dθ
        sx.Hαθ = () -> sx.d2g_dθ

        # Mixed partials
        sx.d2α_dσdθ = () -> sx.∇σ * sx.d2g_dσdθ'
        sx.d2α_dμdθ = () -> sx.∇μ * sx.d2g_dμdθ'
        sx.d2α_dxdθ = () -> sx.d2α_dμdθ + sx.d2α_dσdθ
    end

    return sx
end

function (fs::FantasySurrogate)(x::Vector{T}, θ::Vector{T}; fantasy_index::Int = GROUND_TRUTH_OBSERVATIONS) where T <: Real
    return eval(fs, x, θ, fantasy_index=fantasy_index)
end


function gp_draw(
    s::AS,
    xloc::Vector{T},
    θ::Vector{T};
    stdnormal::Union{Vector{T}, T},
    with_gradient::Bool = false,
    fantasy_index::Union{Int64, Nothing} = nothing) where {T <: Real, AS <: AbstractSurrogate}
    # We can actually embed this logic directly into the evaluation of the surrogate at some arbitrary location
    dim = length(xloc)

    if isnothing(fantasy_index)
        sx = s(xloc, θ)
    else
        sx = s(xloc, θ, fantasy_index=fantasy_index)
    end

    if with_gradient
        @assert length(stdnormal) == dim + 1 "stdnormal has dim = $(length(stdnormal)) but observation vector has dim = $(length(xloc))"
        return sx.dμ + sx.dσ * stdnormal
    else
        @assert length(stdnormal) == 1 "Function observation expects a scalar gaussian random number"
        return sx.μ + sx.σ * stdnormal
    end
end

function get_active_locations(fs::AbstractFantasySurrogate, fantasy_index::Int)
    @views begin
        return fs.X[:, 1:get_known_observations(fs) + fantasy_index + 1]
    end
end

function get_active_cholesky_factor(fs::AbstractFantasySurrogate, fantasy_index::Int64)
    @views begin
        slice = 1:get_known_observations(fs) + fantasy_index + 1
        return fs.L[slice, slice]
    end
end

function construct_perturbation_matrix(fs::AbstractFantasySurrogate, fantasy_index)
    known_observed = get_known_observations(fs)
    d, N = size(get_covariates(fs))
    δX = zeros(d, known_observed + fantasy_index + 1)
    return δX
end

mutable struct SpatialPerturbationSurrogate <: AbstractPerturbationSurrogate
    s::FantasySurrogate
    X::Matrix{Float64}
    max_fantasized_step::Int
end

function SpatialPerturbationSurrogate(; reference_surrogate::FantasySurrogate, fantasy_step::Int)
    δX = construct_perturbation_matrix(reference_surrogate, fantasy_step)
    return SpatialPerturbationSurrogate(reference_surrogate, δX, fantasy_step)
end

"""
The spatial perturbation surrogate is fit to a given base surrogate. Given `sx`, which is the evaluation of
our surrogate at some arbitrary point, this method allows us to consider how the acquisition function
varies when provided some variation `δx` for the `sample_index`-th sample.

Hence, invoking this method for each respective variation allows us to collect how the gradient of our policy changes
with respect to changes in each dimension of the `sample_index`-th sample.
"""
function eval(δs::SpatialPerturbationSurrogate, sx; δx::Vector{T}, sample_index::Int) where T <: Real
    # @assert 0 <= current_step "Can only perturb fantasized locations"
    # @assert current_step <= δs.max_fantasized_step "Attempting to perturb an observation beyong our trajectory"
    @views begin
        δsx = LazyStruct()
        set(δsx, :sx, sx)

        s = δs.s
        x = sx.x
        X = get_active_locations(s, δs.max_fantasized_step)
        # Set's the desired location's perturbation while keeping every other location constant
        δs.X[:,:] .= 0.
        δs.X[:, s.observed + sample_index + 1] .= δx
        # Since we maintain the first set of learned coefficients, we need to offset by 1 to grab
        # the first fantasized pair
        FANTASY_BASED_OFFSET = 1
        # Our fantasy logic assumes the current step can take on the value 0, so we need to offset by
        # 1 again to account for this
        ZERO_BASED_OFFSET = 1
        TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET

        δsx.K = () -> eval_δKXX(get_kernel(s), get_active_locations(s, δs.max_fantasized_step), δs.X)
        δsx.L = () -> get_active_cholesky_factor(s, δs.max_fantasized_step)
        δsx.c = () -> -(δsx.L' \ (δsx.L \ (δsx.K*s.cs[δs.max_fantasized_step + TOTAL_OFFSET])))

        δsx.kx = () -> eval_δKxX(get_kernel(s), x, X, δs.X)
        δsx.∇kx = () -> eval_δ∇KxX(get_kernel(s), x, X, δs.X)

        δsx.μ = () -> δsx.kx'*s.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.kx'*δsx.c
        δsx.∇μ = () -> δsx.∇kx*s.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.∇kx*δsx.c

        δsx.σ = () -> (-2*δsx.kx'*sx.w + sx.w'*(δsx.K*sx.w)) / (2*sx.σ)
        δsx.∇σ = () -> (sx.∇w*(δsx.K*sx.w) - δsx.∇kx*sx.w - sx.∇w*δsx.kx - δsx.σ*sx.∇σ) / sx.σ

        # Write logic for perturbed acquisition. Cost function is attached to sx
        δsx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(δsx.μ, δsx.σ, sx.θ, sx)
        δsx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(δsx.μ, δsx.σ, sx.θ, sx)
        δsx.αxθ = () -> sx.dg_dμ*δsx.μ + sx.dg_dσ*δsx.σ
        δsx.∇αx = () -> sx.dg_dμ*δsx.∇μ + sx.dg_dσ*δsx.∇σ + δsx.dg_dμ*sx.∇μ + δsx.dg_dσ*sx.∇σ
    end

    return δsx
end

function (δs::SpatialPerturbationSurrogate)(sx; variation::Vector{T}, sample_index::Int) where T <: Real
    return eval(δs, sx, δx=variation, sample_index=sample_index)
end

mutable struct DataPerturbationSurrogate <: AbstractPerturbationSurrogate
    s::FantasySurrogate
    X::Matrix{Float64}
    max_fantasized_step::Int
end

function DataPerturbationSurrogate(; reference_surrogate::FantasySurrogate, fantasy_step::Int)
    δX = construct_perturbation_matrix(reference_surrogate, fantasy_step)
    return DataPerturbationSurrogate(reference_surrogate, δX, fantasy_step)
end

function eval(δs::DataPerturbationSurrogate, sx; δx::Vector{T}, ∇y::AbstractVector{T}, sample_index::Int) where T <: Real
    # @assert 0 <= current_step "Can only perturb fantasized locations"
    # @assert current_step <= δs.max_fantasized_step "Attempting to perturb an observation beyong our trajectory"
    @views begin
        δsx = LazyStruct()
        set(δsx, :sx, sx)

        s = δs.s
        x = sx.x
        X = get_active_locations(s, δs.max_fantasized_step)
        # Set's the desired location's perturbation while keeping every other location constant
        δs.X[:,:] .= 0.
        δs.X[:, s.observed + sample_index + 1] .= δx
        # Since we maintain the first set of learned coefficients, we need to offset by 1 to grab
        # the first fantasized pair
        FANTASY_BASED_OFFSET = 1
        # Our fantasy logic assumes the current step can take on the value 0, so we need to offset by
        # 1 again to account for this
        ZERO_BASED_OFFSET = 1
        TOTAL_OFFSET = ZERO_BASED_OFFSET + FANTASY_BASED_OFFSET

        δsx.K = () -> eval_δKXX(get_kernel(s), get_active_locations(s, δs.max_fantasized_step), δs.X)
        δsx.L = () -> get_active_cholesky_factor(s, δs.max_fantasized_step)
        δsx.y = function()
            ys = zeros(s.observed + δs.max_fantasized_step + 1)
            ys[fs.known_observed + sample_index + 1] = ∇y' * δs.X[:, s.observed + sample_index + 1]
            return ys
        end
        δsx.c = () -> -(δsx.L' \ (δsx.L \ (δsx.K*s.cs[δs.max_fantasized_step + TOTAL_OFFSET])))

        δsx.kx = () -> eval_δKxX(get_kernel(s), x, X, δs.X)
        δsx.∇kx = () -> eval_δ∇KxX(get_kernel(s), x, X, δs.X)

        δsx.μ = () -> δsx.kx'*s.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.kx'*δsx.c
        δsx.∇μ = () -> δsx.∇kx*s.cs[δs.max_fantasized_step + TOTAL_OFFSET] + sx.∇kx*δsx.c

        δsx.σ = () -> (-2*δsx.kx'*sx.w + sx.w'*(δsx.K*sx.w)) / (2*sx.σ)
        # δsx.∇σ = () -> zeros(length(x))

        # Write logic for perturbed acquisition. Cost function is attached to sx
        δsx.dg_dμ = () -> first_partial(sx.g, symbol=:μ)(δsx.μ, δsx.σ, sx.θ, sx)
        δsx.dg_dσ = () -> first_partial(sx.g, symbol=:σ)(δsx.μ, δsx.σ, sx.θ, sx)
        δsx.δμσ = () -> (sx.dg_dμ*δsx.μ + sx.dg_dσ*δsx.σ)
        δsx.αxθ = () -> δsx.δμσ
        # δsx.∇αx = () -> sx.dg_dμ*δsx.∇μ + sx.dg_dσ*δsx.∇σ + δsx.dg_dμ*sx.∇μ + δsx.dg_dσ*sx.∇σ
        δsx.∇αx = () -> sx.dg_dμ*δsx.∇μ + δsx.dg_dμ*sx.∇μ + δsx.dg_dσ*sx.∇σ
    end

    return δsx
end

function (δs::DataPerturbationSurrogate)(sx; variation::Vector{T}, ∇y::Vector{T}, sample_index::Int) where T <: Real
    return eval(δs, sx, δx=variation, sample_index=sample_index, ∇y=∇y)
end


# ------------------------------------------------------------------
# Operations for computing optimal hyperparameters.
# ------------------------------------------------------------------
function log_likelihood(s::Surrogate)
    n = get_observed(s)
    y = get_active_observations(s)
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return -y'*c/2 - sum(log.(diag(L))) - n*log(2π)/2
end

function δlog_likelihood(s::Surrogate, δθ::Vector{T}) where T <: Real
    kernel = get_kernel(s)
    X = get_active_covariates(s)
    δK = eval_Dθ_KXX(kernel, X, δθ)
    c = get_active_coefficients(s)
    L = get_active_cholesky(s)
    return (c'*δK*c - tr(L'\(L\δK)))/2
end

function ∇log_likelihood(s::Surrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)

    for j in 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood(s, δθ)
    end

    return ∇L
end

"""
This only optimizes for lengthscale hyperparameter where the lengthscale is the
same in each respective dimension.
"""
function optimize!(
    s::Surrogate;
    lowerbounds::Vector{T},
    upperbounds::Vector{T},
    optim_options = Optim.Options(iterations=30)) where T <: Real

    function fg!(F, G, θ::Vector{T}) where T <: Real
        set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))
        if G !== nothing G .= -∇log_likelihood(s) end
        if F !== nothing return -log_likelihood(s) end
    end

    res = optimize(
        Optim.only_fg!(fg!),
        lowerbounds,
        upperbounds,
        get_hyperparameters(get_kernel(s)),
        Fminbox(LBFGS()),
        optim_options
    )
    θ = Optim.minimizer(res)
    set_kernel!(s, set_hyperparameters!(get_kernel(s), θ))

    return nothing
end

Distributions.mean(sx) = sx.μ
Distributions.std(sx) = sqrt(sx.σ)