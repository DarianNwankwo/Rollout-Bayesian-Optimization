using Distributions

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

function sample_random_matrix(xmin, xmax, d, n)
    matrix = zeros(d, n)
    for i in 1:d
        for j in 1:n
            matrix[i, j] = xmin + (xmax - xmin) * rand()
        end
    end
    return matrix
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


function inbounds(x0, lbs, ubs)
    return all(lbs .< x0 .< ubs)
end


function find_zeros(vector::Vector{Float64})
    indices = Int[]
    
    for i in 2:length(vector)
        if (vector[i - 1] > 0 && vector[i] <= 0) || (vector[i - 1] <= 0 && vector[i] > 0)
            push!(indices, i)
        end
    end
    
    return indices
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


function generate_batch(N, X; lbs, ubs)
    s = SobolSeq(lbs, ubs)
    B = reduce(hcat, next!(s) for i = 1:N*2)
    B = convert(Matrix{Float64}, filter(x -> !(x in X), B)')
    return B[:, 1:N]
end

function centered_fd(f, u, du, h)
    (f(u+h*du)-f(u-h*du))/(2h)
end

function update_λ(λ, ∇g)
    k = ceil(log10(norm(∇g)))
    return λ * 10. ^ (-k)
end

function update_x(x; λ, ∇g, lbs, ubs)
    # λ = update_λ(λ, ∇g)
    x = x .+ λ*∇g
    x = max.(x, lbs)
    x = min.(x, ubs)
    return x
end

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
    push!(m, mt)
    vt = β2 * v[end] + (1 - β2) * ∇g.^2  # Update second moment estimate
    push!(v, vt)
    mt_hat = mt / (1 - β1)  # Correct for bias in first moment estimate
    vt_hat = vt / (1 - β2)  # Correct for bias in second moment estimate
    x = x0 + λ * mt_hat ./ (sqrt.(vt_hat) .+ ϵ)  # Compute updated position
    x = max.(x, lbs)
    x = min.(x, ubs)
    return x  # Return updated position and updated moment estimates
end



function stochastic_gradient_ascent_adam1(;
    λ=0.01, β1=0.9, β2=0.999, ϵ=1e-8, ftol=1e-6, gtol=1e-6, varred=true,
    sur::RBFsurrogate, tp::TrajectoryParameters, max_sgd_iters::Int, xstarts::Matrix{Float64},
    candidate_locations::SharedMatrix{Float64}, candidate_values::SharedArray{Float64}
    )
    x0 = tp.x0
    m = [zeros(size(x0))]
    v = [zeros(size(x0))]
    xstart = x0
    xfinish = x0
    xall = [x0]

    rewards, rewards_grads = [], []
    iters = 1

    for epoch in 1:max_sgd_iters
        iters = epoch

        # Compute stochastic estimates of function and gradient
        μx, ∇μx, μx_stderr, ∇μx_stderr = simulate_trajectory(
            sur, tp, xstarts; variance_reduction=varred,
            candidate_locations=candidate_locations, candidate_values=candidate_values
        )
        # μx, ∇μx, μx_stderr, ∇μx_stderr = distributed_simulate_trajectory(sur, tp, xstarts; variance_reduction=varred)

        # Update position and moment estimates
        tp.x0 .= update_x_adam!(tp.x0; ∇g=∇μx, λ=λ, β1=β1, β2=β2, ϵ=ϵ, m=m, v=v, lbs=tp.lbs, ubs=tp.ubs)
        xfinish .= tp.x0
        push!(xall, tp.x0)
        push!(rewards, μx)
        push!(rewards_grads, ∇μx)

        # Check for convergence: gradient is approx 0,
        if length(rewards) > 2 && rewards[end] - rewards[end-1] < 0.
            # println("Objective is small")
            xfinish .= xall[end - 1]
            break
        end

        if length(rewards_grads) > 2 && sign(first(rewards_grads[end - 1])) != sign(first(rewards_grads[end]))
            # println("Gradient is small")
            xfinish .= xall[end - 1]
            break
        end
    end

    result = (
        start=xstart, finish=xfinish, final_obj=rewards[end], final_grad=rewards_grads[end], iters=iters, success=true,
        sequence=xall, grads=rewards_grads, obj=rewards
    )
    return result
end


function stochastic_gradient_ascent_adam(;
    λ=0.01, β1=0.9, β2=0.999, ϵ=1e-8, ftol=1e-6, gtol=1e-6, varred=true,
    sur::RBFsurrogate, tp::TrajectoryParameters, max_sgd_iters::Int, xstarts::Matrix{Float64},
    candidate_locations::SharedMatrix{Float64}, candidate_values::SharedArray{Float64},
    αxs::SharedArray{Float64}, ∇αxs::SharedMatrix{Float64}
    )
    x0 = tp.x0
    m = [zeros(size(x0))]
    v = [zeros(size(x0))]
    xstart = x0
    xfinish = x0
    xall = [x0]

    rewards, rewards_grads = [], []
    iters = 1

    for epoch in 1:max_sgd_iters
        iters = epoch

        # Compute stochastic estimates of function and gradient
        μx, ∇μx, μx_stderr, ∇μx_stderr = distributed_simulate_trajectory(
            sur, tp, xstarts; variance_reduction=varred,
            candidate_locations=candidate_locations, candidate_values=candidate_values,
            αxs=αxs, ∇αxs=∇αxs
        )

        # Update position and moment estimates
        tp.x0 = update_x_adam!(tp.x0; ∇g=∇μx, λ=λ, β1=β1, β2=β2, ϵ=ϵ, m=m, v=v, lbs=tp.lbs, ubs=tp.ubs)
        xfinish = tp.x0
        push!(xall, tp.x0)
        push!(rewards, μx)
        push!(rewards_grads, ∇μx)

        # Check for convergence: gradient is approx 0,
        if length(rewards) > 2 && rewards[end] - rewards[end-1] < 0.
            # println("Objective is small")
            xfinish = xall[end - 1]
            break
        end

        if length(rewards_grads) > 2 && sign(first(rewards_grads[end - 1])) != sign(first(rewards_grads[end]))
            # println("Gradient is small")
            xfinish = xall[end - 1]
            break
        end
    end

    result = (
        start=xstart, finish=xfinish, final_obj=rewards[end], final_grad=rewards_grads[end], iters=iters, success=true,
        sequence=xall, grads=rewards_grads, obj=rewards
    )
    return result
end


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


function generate_initial_guesses(N::Int, lbs::Vector{T}, ubs::Vector{T},) where T <: Number
    ϵ = 1e-6
    seq = SobolSeq(lbs, ubs)
    initial_guesses = reduce(hcat, next!(seq) for i = 1:N)
    initial_guesses = hcat(initial_guesses, lbs .+ ϵ)
    initial_guesses = hcat(initial_guesses, ubs .- ϵ)

    return initial_guesses
end