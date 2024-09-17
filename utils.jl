"""
Generate low discrepancy Sobol sequence of uniform random variables
"""
function gen_uniform(samples; dim=1)
    sobol = SobolSeq(zeros(dim), ones(dim))
    S = zeros(dim, samples)
    
    for j in 1:samples
        S[:,j] = next!(sobol)
    end
    
    S
end

function is_odd(num)
    return mod(num,2) == 1
end

"""
Transforms an even-sized multivariate uniform distribution to be normally
distributed with mean 0 and variance 1.
"""
function box_muller_transform(S)
    dim, samples = size(S)
    N = zeros(dim, samples)
    
    for j in 1:samples
        y = zeros(dim)
        x = S[:,j]
        
        for i in 1:dim
            if is_odd(i)
                y[i] = sqrt(-2log10(x[i]))*cos(2π*x[i+1])
            else
                y[i] = sqrt(-2log10(x[i-1]))*sin(2π*x[i])
            end
        end
        
        N[:,j] = y
    end
    
    N
end

dense_1D_discretization(;lb, ub, stepsize) = lb:stepsize:ub

"""
Produces a sequence of standard normally distributed values in 1D
"""
function uniform1d_to_normal(samples)
    uniform2d = gen_uniform(samples, dim=2)
    normal2d = box_muller_transform(uniform2d)
    marginal_normal = normal2d[1,:]
    
    marginal_normal
end

"""
Generate a low discrepancy multivariate normally distributed sequence
for monte carlo simulation of rollout acquisition functions with a max
horizon of h. The resulting tensor will be of size Mx(D+1)xH, where M
is the number of Monte Carlo iterations, D is the dimension of the
input, and H is our max horizon.
"""
function gen_low_discrepancy_sequence(samples, dim, horizon)
    # We need function and gradient samples, so dim here corresponds to the input space.
    # The +1 here corresponds to the function observation.
    offset = isodd(dim+1) ? 1 : 0
    S = gen_uniform(samples*horizon, dim=dim+1+offset)
    N = box_muller_transform(S)
    N = reshape(N, samples, dim+1+offset, horizon)
    
    return N[:,1:end-offset,:]
end

function randsample(N, d, lbs, ubs)
    X = zeros(d, N)
    for j = 1:N
        for i = 1:d
            X[i,j] = rand(Uniform(lbs[i], ubs[i]))
        end
    end
    return X
end


function stdize(series)
    smax, smin = maximum(series), minimum(series)
    return [(s-smin)/(smax-smin) for s in series]
end

function pairwise_diff_issmall(prev, next; tol=1e-6)
    # If minimizing, prev-next. If maximizing, next-prev
    return abs(prev-next) < tol
end

function pairwise_forward_diff(series)
    N = length(series)
    result = []
    
    for i in 1:N-1
        diff = series[i] - series[i+1]
        push!(result, diff)
    end
    
    result
end

function is_still_increasing(series)
    a, b, c = series[end-2:end]
    return a < b < c
end

function print_divider(char="-"; len=80)
    for _ in 1:len print(char) end
    println()
end

"""
Returns true as long as we need to keep performing steps of our optimization
iteration. As soon as we reach our maximum iteration count we terminate, but
if the pairwise difference is sufficiently large (> tol), we believe we can
keep improving and continue our iteration. If the pairwise difference is not
large, there is a chance we can improve, so we continue. The next iteration
then checks if our rate of change has gone from increasing to decreasing,
w.l.o.g.
"""
function sgd_hasnot_converged(iter, grads, max_iters; tol=1e-4)
    # bpdiff captures the relative improvement 
    return ((!(iter > 3 && pairwise_diff_issmall(grads[end-1:end]..., tol=tol))
           || (iter > 3 && is_still_increasing(grads)))
           && iter < max_iters)
end


"""
Generate a batch of N points inbounds relative to the lowerbounds and
upperbounds
"""
function generate_batch(N; lbs, ubs, ϵinterior=1e-2)
    s = SobolSeq(lbs, ubs)
    B = reduce(hcat, next!(s) for i = 1:N)
    # Concatenate a few points to B near the bounds in each respective dimension.
    lbs_interior = lbs .+ ϵinterior
    ubs_interior = ubs .- ϵinterior
    B = hcat(B, lbs_interior)
    B = hcat(B, ubs_interior)
    return B
end

function filter_batch!(B::Matrix{Float64}, X::Matrix{Float64}; ϵ=1e-2)
    # For all the points in the batch B, check if any of the points in X are within ϵ
    # of the points in B. If so, remove that point from B.
    for i = 1:size(B, 2)
        for j = 1:size(X, 2)
            if norm(B[:,i] - X[:,j]) < ϵ
                B = B[:, 1:i-1]
                break
            end
        end
    end
end

centered_fd(f, u, du, h) = (f(u+h*du)-f(u-h*du)) / (2h)

"""
This assumes stochastic gradient ascent

x0: input to function g
∇g: gradient of function g
λ: learning rate
β1, β2, ϵ
m: first moment estimate
v: second moment estimate
"""
function update_x_adam!(x0::Vector{Float64}; ∇g,  λ, β1, β2, ϵ, m, v, lbs, ubs)
    # Print the dimension and name of each parameter in this function
    mt = β1 * m[end] + (1 - β1) * ∇g  # Update first moment estimate
    push!(m, vec(mt))
    vt = β2 * v[end] + (1 - β2) * ∇g.^2  # Update second moment estimate
    push!(v, vec(vt))
    mt_hat = mt / (1 - β1)  # Correct for bias in first moment estimate
    vt_hat = vt / (1 - β2)  # Correct for bias in second moment estimate
    x = x0 + λ * mt_hat ./ (sqrt.(vt_hat) .+ ϵ)  # Compute updated position
    x = max.(x, lbs)
    x = min.(x, ubs)
    return x  # Return updated position and updated moment estimates
end

"""
Early Stopping without a Validation Set: `https://arxiv.org/abs/1703.09580`
"""
function early_stopping_without_a_validation_set(;
    ∇f::AbstractVector,
    var_∇f::AbstractVector,
    sample_size::Integer)
    dim = length(∇f)

    ratio = sum((∇f .^ 2) ./ var_∇f)
    return (1. - (sample_size / dim) * ratio) > 0.
end
eswavs = early_stopping_without_a_validation_set


function measure_gap(observations::Vector{T}, fbest::T) where T <: Number
    ϵ = 1e-8
    initial_minimum = observations[1]
    subsequent_minimums = [
        minimum(observations[1:j]) for j in 1:length(observations)
    ]
    numerator = initial_minimum .- subsequent_minimums
    
    if abs(fbest - initial_minimum) < ϵ
        return 1. 
    end
    
    denominator = initial_minimum - fbest
    result = numerator ./ denominator

    for i in 1:length(result)
        if result[i] < ϵ
            result[i] = 0
        end
    end

    return result
end

function generate_initial_guesses(N::Integer, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end

struct ExperimentSetup
    tp::TrajectoryParameters
    inner_solve_xstarts::AbstractMatrix
    resolutions::AbstractVector
    spatial_gradients_container::Union{Nothing, AbstractMatrix}
    hyperparameter_gradients_container::Union{Nothing, AbstractMatrix}
end

function ExperimentSetup(;
    tp::TrajectoryParameters,
    number_of_starts::Integer)
    lbs, ubs = get_spatial_bounds(tp)
    inner_solve_xstarts = generate_initial_guesses(number_of_starts, lbs, ubs)
    resolutions = zeros(tp.mc_iters)
    spatial_gradients_container = zeros(length(get_starting_point(tp)), tp.mc_iters)
    hyperparameter_gradients_container = zeros(length(get_hyperparameters(tp)), tp.mc_iters)

    return ExperimentSetup(
        tp, inner_solve_xstarts, resolutions, spatial_gradients_container, hyperparameter_gradients_container
    )
end

function get_container(es::ExperimentSetup; symbol::Symbol)
    if symbol == :f
        return es.resolutions
    elseif symbol == :grad_f
        return es.spatial_gradients_container
    elseif symbol == :grad_hypers
        return es.hyperparameter_gradients_container
    else
        error("Unknown symbol. Use either :f, :grad_f, or :grad_hypers")
    end
end

get_starts(es::ExperimentSetup) = es.inner_solve_xstarts

function to(n; key="KB")
    mapping = Dict("KB" => 1, "MB" => 2, "GB" => 3)
    factor = mapping[key]
    conversion = n / (1024 ^ mapping[key])
    return "$(conversion)$(key)"
end

function stochastic_solve(;
    optimizer::StochasticGradientAscent,
    surrogate::Surrogate,
    tp::TrajectoryParameters,
    es::ExperimentSetup,
    start::AbstractVector)
    tpc = deepcopy(tp)
    set_starting_point!(tpc, deepcopy(start))

    for iter in 1:50
        # Compute stochastic estimates of function and gradient
        eto = simulate_adjoint_trajectory(
            surrogate,
            tpc,
            inner_solve_xstarts=get_starts(es),
            resolutions=get_container(es, symbol=:f),
            spatial_gradients_container=get_container(es, symbol=:grad_f),
            hyperparameter_gradients_container=get_container(es, symbol=:grad_hypers),
        )

        # Check for convergence using statistic developed in "Early Stopping Without a Validation Set"
        # by Mahsereci et al.
        if eswavs(∇f=gradient(eto), var_∇f=std_gradient(eto) .^ 2, sample_size=tp.mc_iters)
            break
        end

        update!(optimizer, x=get_starting_point(tpc), ∇f=gradient(eto))
    end

    return get_starting_point(tpc)
end

function deterministic_solve(;
    optimizer::StochasticGradientAscent,
    surrogate::Surrogate,
    tp::TrajectoryParameters,
    es::ExperimentSetup,
    start::AbstractVector,
    func::Function,
    grad::Function,
    tol::Float64 = 1e-6,   # Tolerance for gradient norm
    max_iters::Int = 200   # Maximum number of iterations
)
    # Copy the trajectory parameters and set the starting point
    tpc = deepcopy(tp)
    set_starting_point!(tpc, deepcopy(start))

    for iter in 1:max_iters
        # Compute deterministic estimates of function and gradient
        eto = deterministic_simulate_trajectory(
            surrogate,
            tpc,
            inner_solve_xstarts=get_starts(es),
            func=func,
            grad=grad
        )

        # Get the gradient from the simulation
        ∇f = gradient(eto)

        # Check convergence based on the gradient norm
        if norm(∇f) <= tol
            break
        end

        # Update the optimizer with the current point and gradient
        update!(optimizer, x=get_starting_point(tpc), ∇f=∇f)
    end

    # Return the final starting point as the solution
    return get_starting_point(tpc)
end