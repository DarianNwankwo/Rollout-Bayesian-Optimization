# Define the abstract types
abstract type Optimizer end
abstract type StochasticGradientAscent <: Optimizer end

# Define StandardSGA with its parameters
struct StandardSGA <: StochasticGradientAscent
    η::Float64  # learning rate

    # Inner constructor with default values and keyword arguments
    function StandardSGA(; η::Float64 = 0.01)
        new(η)
    end
end

# Update function for Standard SGA
function update!(
    optimizer::StandardSGA;
    x::AbstractVector,
    ∇f::AbstractVector)
    x .+= optimizer.η * ∇f  # Standard SGA update (ascent)
    return x
end

# Define Adam optimizer with its required parameters
mutable struct Adam <: StochasticGradientAscent
    η::Float64   # learning rate
    β1::Float64  # decay rate for the first moment
    β2::Float64  # decay rate for the second moment
    ε::Float64   # small constant for numerical stability
    m::Vector{AbstractVector}  # first moment vector
    v::Vector{AbstractVector}  # second moment vector
    t::Integer      # time step counter

    # Inner constructor with default values and keyword arguments
    function Adam(;
        η::Float64 = 0.001, 
        β1::Float64 = 0.9, 
        β2::Float64 = 0.999, 
        ε::Float64 = 1e-8,
        t::Int = 0)
        m = []
        v = []
        new(η, β1, β2, ε, m, v, t)
    end
end

# Update function for Adam optimizer
function update!(
    optimizer::Adam;
    x::AbstractVector,
    ∇f::AbstractVector)
    if length(optimizer.m) == length(optimizer.v) == 0
        push!(optimizer.m, zeros(length(∇f)))
        push!(optimizer.v, zeros(length(∇f)))
    end
    # Increment the time step
    optimizer.t += 1

    # Update biased first moment estimate
    push!(optimizer.m, optimizer.β1 .* optimizer.m[end] .+ (1 - optimizer.β1) .* ∇f)

    # Update biased second moment estimate
    push!(optimizer.v, optimizer.β2 .* optimizer.v[end] .+ (1 - optimizer.β2) .* (∇f .^ 2))

    # Compute bias-corrected first moment estimate
    m̂ = optimizer.m[end] ./ (1 - optimizer.β1^optimizer.t)

    # Compute bias-corrected second moment estimate
    v̂ = optimizer.v[end] ./ (1 - optimizer.β2^optimizer.t)

    # Update parameters
    x .+= optimizer.η * m̂ ./ (sqrt.(v̂) .+ optimizer.ε)

    return x
end