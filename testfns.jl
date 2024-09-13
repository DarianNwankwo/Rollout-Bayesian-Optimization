import Base:+, *
# https://www.sfu.ca/~ssurjano/optimization.html
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

struct TestFunction
    dim::Integer
    bounds::AbstractMatrix
    xopt::NTuple{N, AbstractVector{<:Real}} where N
    f::Function
    ∇f::Function
end

(testfn::TestFunction)(x::Vector{Float64}) = testfn.f(x)
gradient(testfn::TestFunction) = testfn.∇f
function (testfn::TestFunction)(X::AbstractMatrix; grad=false)
    return map(
        !grad ? testfn.f : testfn.∇f,
        eachcol(X)
    )
end


function get_collapsed_bounds(t1::TestFunction, t2::TestFunction)
    closest_to_origin(x::AbstractVector) = x[findmin(abs, x)[2]]

    bounds = zeros(t1.dim, 2)
    union_lowerbounds = [t1.bounds[:, 1] t2.bounds[:, 1]]
    union_upperbounds = [t1.bounds[:, 2] t2.bounds[:, 2]]

    for i = 1:t1.dim
        bounds[i, 1] = closest_to_origin(union_lowerbounds[i, :])
        bounds[i, 2] = closest_to_origin(union_upperbounds[i, :])
    end

    return bounds
end


function +(t1::TestFunction, t2::TestFunction)
    @assert t1.dim == t2.dim "dim(t1) must equal dim(t2)"
    return TestFunction(
        t1.dim,
        get_collapsed_bounds(t1, t2),
        (zeros(t1.dim),),
        (x) -> t1.f(x) + t2.f(x),
        (x) -> t1.∇f(x) + t2.∇f(x)
    )
end

function *(t1::TestFunction, t2::TestFunction)
    @assert t1.dim == t2.dim "dim(t1) must equal dim(t2)"
    return TestFunction(
        t1.dim,
        get_collapsed_bounds(t1, t2),
        (zeros(t1.dim),),
        (x) -> t1.f(x) * t2.f(x),
        (x) -> t1.f(x) * t2.∇f(x) + t1.∇f(x) * t2.f(x)
    )
end


function scalar_scale(testfn::TestFunction, s::Number)
    function f(x)
        return testfn.f(x/s)
    end
    function ∇f(x)
        return testfn.∇f(x/s) / s
    end
    return TestFunction(testfn.dim, testfn.bounds*s, testfn.xopt .* s, f, ∇f)
end


function vshift(testfn::TestFunction, s::Number)
    function f(x)
        return testfn.f(x) + s
    end
    function ∇f(x)
        return testfn.∇f(x)
    end
    return TestFunction(testfn.dim, testfn.bounds, testfn.xopt, f, ∇f)
end

function hshift(testfn::TestFunction, s::AbstractVector)
    function f(x)
        return testfn.f(x + s)
    end
    function ∇f(x)
        return testfn.∇f(x)
    end
    return TestFunction(testfn.dim, testfn.bounds, testfn.xopt + s , f, ∇f)
end

get_bounds(t::TestFunction) = (t.bounds[:, 1], t.bounds[:, 2])


function tplot(f :: TestFunction)
    if f.dim == 1
        xx = range(f.bounds[1,1], f.bounds[1,2], length=250)
        p = plot(xx, (x) -> f([x]), label="f(x)")
        scatter!(p, [xy[1] for xy in f.xopt], [f(xy) for xy in f.xopt], label="xopt")
        return p
    elseif f.dim == 2
        xx = range(f.bounds[1,1], f.bounds[1,2], length=100)
        yy = range(f.bounds[2,1], f.bounds[2,2], length=100)
        plot(xx, yy, (x,y) -> f([x,y]), st=:contour)
        scatter!([xy[1] for xy in f.xopt], [xy[2] for xy in f.xopt], label="xopt")
        # scatter!([f.xopt[1]], [f.xopt[2]], label="xopt")
    else
        error("Can only plot 1- or 2-dimensional TestFunctions")
    end
end


function TestBraninHoo(; a=1, b=5.1/(4π^2), c=5/π, r=6, s=10, t=1/(8π))
    function f(xy)
        x = xy[1]
        y = xy[2]
        a*(y-b*x^2+c*x-r)^2 + s*(1-t)*cos(x) + s
    end
    function ∇f(xy)
        x = xy[1]
        y = xy[2]
        dx = 2*a*(y-b*x^2+c*x-r)*(-b*2*x+c) - s*(1-t)*sin(x)
        dy = 2*a*(y-b*x^2+c*x-r)
        [dx, dy]
    end
    bounds = [-5.0 10.0 ; 0.0 15.0]
    xopt = ([-π, 12.275], [π, 2.275], [9.42478, 2.475])
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestRosenbrock()
    f(xy) = (1-xy[1])^2 + 100*(xy[2]-xy[1]^2)^2
    ∇f(xy) = [-2*(1-xy[1]) - 400*xy[1]*(xy[2]-xy[1]^2), 200*(xy[2]-xy[1]^2)]
    return TestFunction(2, [-2.0 2.0 ; -1.0 3.0 ], (ones(2),), f, ∇f)
end


function TestRastrigin(n)
    f(x) = 10*n + sum(x.^2 - 10*cos.(2π*x))
    ∇f(x) = 2*x + 20π*sin.(2π*x)
    bounds = zeros(n, 2)
    bounds[:,1] .= -5.12
    bounds[:,2] .=  5.12
    xopt = (zeros(n),)
    return TestFunction(n, bounds, xopt, f, ∇f)
end


function TestAckley(d; a=20.0, b=0.2, c=2π)
    
    function f(x)
        nx = norm(x)
        cx = sum(cos.(c*x))
        -a*exp(-b/sqrt(d)*nx) - exp(cx/d) + a + exp(1)
    end
    
    function ∇f(x)
        nx = norm(x)
        if nx == 0.0
            return zeros(d)
        else
            cx = sum(cos.(c*x))
            dnx = x/nx
            dcx = -c*sin.(c*x)
            (a*b)/sqrt(d)*exp(-b/sqrt(d)*norm(x))*dnx - exp(cx/d)/d*dcx
        end
    end

    bounds = zeros(d,2)
    bounds[:,1] .= -32.768
    bounds[:,2] .=  32.768
    xopt = (zeros(d),)

    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestSixHump()

    function f(xy)
        x = xy[1]
        y = xy[2]
        xterm = (4.0-2.1*x^2+x^4/3)*x^2
        yterm = (-4.0+4.0*y^2)*y^2
        xterm + x*y + yterm
    end

    function ∇f(xy)
        x = xy[1]
        y = xy[2]
        dxterm = (-4.2*x+4.0*x^3/3)*x^2 + (4.0-2.1*x^2+x^4/3)*2.0*x
        dyterm = (8.0*y)*y^2 + (-4.0+4.0*y^2)*2.0*y
        [dxterm + y, dyterm + x]
    end

    # There's a symmetric optimum
    xopt = ([0.089842, -0.712656], [-0.089842, 0.712656])

    return TestFunction(2, [-3.0 3.0 ; -2.0 2.0], xopt, f, ∇f)
end


function TestGramacyLee()
    f(x) = sin(10π*x[1])/(2*x[1]) + (x[1]-1.0)^4
    ∇f(x) = [5π*cos(10π*x[1])/x[1] - sin(10π*x[1])/(2*x[1]^2) + 4*(x[1]-1.0)^3]
    bounds = zeros(1, 2)
    bounds[1,1] = 0.5
    bounds[1,2] = 2.5
    xopt=([0.548563],)
    return TestFunction(1, bounds, xopt, f, ∇f)
end


function TestGoldsteinPrice()
    function f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = x1 + x2 + 1
        t2 = 2 * x1 - 3 * x2
        t3 = x1^2
        t4 = x2^2
        
        term1 = 1 + t1^2 * (19 - 14 * x1 + 3 * t3 - 14 * x2 + 6 * x1 * x2 + 3 * t4)
        term2 = 30 + t2^2 * (18 - 32 * x1 + 12 * t3 + 48 * x2 - 36 * x1 * x2 + 27 * t4)
        
        return term1 * term2
    end
    
    function ∇f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = x1 + x2 + 1
        t2 = 2 * x1 - 3 * x2
        t3 = x1^2
        t4 = x2^2
        
        common1 = 2 * t1 * (3 * t3 + 6 * x1 * x2 - 14 * x1 + 3 * t4 - 14 * x2 + 19)
        common2 = t2^2 * (12 * t3 - 36 * x1 * x2 + 18 - 32 * x1 + 27 * t4 + 48 * x2)
        
        df1 = common1 + common2 * (2 * t3 - 36 * x1 * x2 - 32 * x2 + 48 * t4 + 18 - 32 * x1)
        df2 = common1 + common2 * (48 * x1 - 36 * x1 * x2 - 32 * x2 + 27 * t4 + 18 - 32 * x1)
        
        return [df1, df2]
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -2.0
    bounds[:,2] .=  2.0

    xopt = ([0.0, -1.0],)

    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestBeale()
    function f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = 1.5 - x1 + x1 * x2
        t2 = 2.25 - x1 + x1 * x2^2
        t3 = 2.625 - x1 + x1 * x2^3
        return t1^2 + t2^2 + t3^2
    end
    
    function ∇f(xy)
        x1 = xy[1]
        x2 = xy[2]
        t1 = 1.5 - x1 + x1 * x2
        t2 = 2.25 - x1 + x1 * x2^2
        t3 = 2.625 - x1 + x1 * x2^3
        
        df1 = 2 * (t1 * (x2 - 1) + t2 * (x2^2 - 1) + t3 * (x2^3 - 1))
        df2 = 2 * (t1 * x1 + 2 * t2 * x1 * x2 + 3 * t3 * x1 * x2^2)
        
        return [df1, df2]
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -4.5
    bounds[:,2] .=  4.5

    xopt = ([3.0, 0.5],)

    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestEasom()
    f(x) = -cos(x[1])*cos(x[2])*exp(-((x[1]-π)^2 + (x[2]-π)^2))

    function ∇f(x)
        function common_subexpression(x)
            c = cos(x[1]) * cos(x[2])
            e = exp(-((x[1] - π)^2 + (x[2] - π)^2))
            term = 2 * (x[1] - π) * cos(x[2]) + 2 * (x[2] - π) * cos(x[1])
            return c, e, term
        end

        c, e, term = common_subexpression(x)
        
        df1 = c * e * term - sin(x[1]) * cos(x[2]) * e
        df2 = c * e * term - sin(x[2]) * cos(x[1]) * e
        
        return [df1, df2]
    end

    bounds = zeros(2, 2)
    bounds[:, 1] .= -100.0
    bounds[:, 2] .= 100.0
    
    xopt=([π, π],)

    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestStyblinskiTang(d)
    f(x) = 0.5*sum(x.^4 - 16*x.^2 + 5*x)
    ∇f(x) = 2*x.^3 - 16*x + 2.5
    bounds = zeros(d, 2)
    bounds[:,1] .= -5.0
    bounds[:,2] .=  5.0
    xopt = (repeat([-2.903534], d),)
    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestBukinN6()
    function f(x)
        x1 = x[1]
        x2 = x[2]
        t1 = abs(x2 - 0.01 * x1^2)
        
        term1 = 100 * sqrt(t1) + 0.01 * abs(x1 + 10)
        return term1
    end
    
    function ∇f(x)
        x1 = x[1]
        x2 = x[2]
        t1 = abs(x2 - 0.01 * x1^2)
        t2 = sqrt(t1)
        
        df1 = 0.01 * x1 / t2 + 0.01
        df2 = 50 * (x2 - 0.01 * x1^2) / t2
        
        return [df1, df2]
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -15.0
    bounds[:,2] .=  3.0
    xopt = ([-10.0, 1.0],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestCrossInTray()
    f(x) = -0.0001 * (abs(sin(x[1]) * sin(x[2]) * exp(abs(100 - sqrt(x[1]^2 + x[2]^2) / π))) + 1)^0.1
    ∇f(x) = zeros(2) # TODO
    bounds = zeros(2, 2)
    bounds[:,1] .= -10.0
    bounds[:,2] .=  10.0
    xopt = (repeat([1.34941], 2),)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestEggHolder()
    f(x) = -(x[2] + 47) * sin(sqrt(abs(x[2] + x[1] / 2 + 47))) - x[1] * sin(sqrt(abs(x[1] - (x[2] + 47))))
    ∇f(x) = zeros(2) # TODO
    bounds = zeros(2, 2)
    bounds[:,1] .= -512.0
    bounds[:,2] .=  512.0
    xopt = ([512, 404.2319],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestHolderTable()
    f(x) = -abs(sin(x[1]) * cos(x[2]) * exp(abs(1 - sqrt(x[1]^2 + x[2]^2) / π)))
    ∇f(x) = zeros(2) # TODO
    bounds = zeros(2, 2)
    bounds[:,1] .= -10.0
    bounds[:,2] .=  10.0
    xopt = ([8.05502, 9.66459],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestSchwefel(d)
    f(x) = 418.9829 * d - sum(x .* sin.(sqrt.(abs.(x))))
    ∇f(x) = zeros(d) # TODO
    bounds = zeros(d, 2)
    bounds[:,1] .= -500.0
    bounds[:,2] .=  500.0
    xopt = (repeat([420.9687], d),)
    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestLevyN13()
    f(x) = sin(3 * π * x[1])^2 + (x[1] - 1)^2 * (1 + sin(3 * π * x[2])^2) + (x[2] - 1)^2 * (1 + sin(2 * π * x[2])^2)
    ∇f(x) = zeros(2) # TODO
    bounds = zeros(2, 2)
    bounds[:,1] .= -10.0
    bounds[:,2] .=  10.0
    xopt = ([1, 1],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestTrid(d)
    f(x) = sum((x .- 1).^2) - sum(x[2:end] .* x[1:end-1])
    ∇f(x) = zeros(d) # TODO
    bounds = zeros(d, 2)
    bounds[:,1] .= -d^2
    bounds[:,2] .=  d^2
    xopt = ([i*(d + 1. - i) for i in 1:d],)
    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestMccormick()
    f(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
    ∇f(x) = zeros(2) # TODO
    bounds = zeros(2, 2)
    bounds[:,1] .= -1.5
    bounds[:,2] .=  4.0
    xopt = ([-0.54719, -1.54719],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestHartmann3D()
    α = [1.0, 1.2, 3.0, 3.2]
    A = [
        3.0 10 30;
        0.1 10 35;
        3.0 10 30;
        0.1 10 35
    ]
    P = 1e-4 * [
        3689 1170 2673;
        4699 4387 7470;
        1091 8732 5547;
        381 5743 8828
    ]
    
    function f(x)
        f = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:3
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            f += α[i] * exp(-t)
        end
        return -f
    end
    
    ∇f(x) = zeros(3) # TODO
    bounds = zeros(3, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    xopt = ([0.114614, 0.555649, 0.852547],)
    return TestFunction(3, bounds, xopt, f, ∇f)
end


function TestHartmann4D()
    α = [1.0, 1.2, 3.0, 3.2]
    A = [
        10 3 17 3.5 1.7 8;
        0.05 10 17 0.1 8 14;
        3 3.5 1.7 10 17 8;
        17 8 0.05 10 0.1 14
    ]
    P = 1e-4 * [
        1312 1696 5569 124 8283 5886;
        2329 4135 8307 3736 1004 9991;
        2348 1451 3522 2883 3047 6650;
        4047 8828 8732 5743 1091 381
    ]
    
    function f(x)
        f = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:6
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            f += α[i] * exp(-t)
        end
        return -f
    end
    
    ∇f(x) = zeros(6) # TODO
    bounds = zeros(6, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    xopt = ([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],)
    return TestFunction(6, bounds, xopt, f, ∇f)
end


function TestHartmann6D()
    α = [1.0, 1.2, 3.0, 3.2]
    A = [
        10 3 17 3.5 1.7 8;
        0.05 10 17 0.1 8 14;
        3 3.5 1.7 10 17 8;
        17 8 0.05 10 0.1 14
    ]
    P = 1e-4 * [
        1312 1696 5569 124 8283 5886;
        2329 4135 8307 3736 1004 9991;
        2348 1451 3522 2883 3047 6650;
        4047 8828 8732 5743 1091 381
    ]
    
    function f(x)
        f = 0.0
        for i in 1:4
            t = 0.0
            for j in 1:6
                t += A[i,j] * (x[j] - P[i,j])^2
            end
            f += α[i] * exp(-t)
        end
        return -f
    end
    
    ∇f(x) = zeros(6) # TODO
    bounds = zeros(6, 2)
    bounds[:,1] .= 0.0
    bounds[:,2] .= 1.0
    xopt = ([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],)
    return TestFunction(6, bounds, xopt, f, ∇f)
end


function TestConstant(n=0.; lbs::Vector{<:T}, ubs::Vector{<:T}) where T <: Real
    f(x) = n
    ∇f(x) = zeros(length(x))
    xopt = (zeros(length(lbs)),)
    bounds = hcat(lbs, ubs)
    return TestFunction(length(lbs), bounds, xopt, f, ∇f)
end


function TestQuadratic1D(a=1, b=0, c=0; lb=-1.0, ub=1.0)
    f(x) = a*first(x)^2 + b*first(x) + c
    ∇f(x) = [2*a*first(x) + b]
    bounds = [lb ub]
    xopt = (zeros(1),)
    return TestFunction(1, bounds, xopt, f, ∇f)
end

# Create a test function named TestLinearCosine1D that takes a paramater for the frequency of the cosine
# and a parameter for the amplitude of the cosine. The function should be a linear function of x plus a cosine
# function of the form a*cos(b*x). The function should have a single optimum at x=0.
function TestLinearCosine1D(a=1, b=1; lb=-1.0, ub=1.0)
    f(x) = a*first(x) * cos(b*first(x))
    ∇f(x) = [a*cos(b*first(x)) - a*b*first(x)*sin(b*first(x))]
    bounds = [lb ub]
    xopt = (zeros(1),) # TODO
    return TestFunction(1, bounds, xopt, f, ∇f)
end


function TestShekel()
    C = [4. 1. 8. 6. 3. 2. 5. 8. 6. 7.;
         4. 1. 8. 6. 7. 9. 3. 1. 2. 3.;
         4. 1. 8. 6. 3. 2. 5. 8. 6. 7.;
         4. 1. 8. 6. 7. 9. 3. 1. 2. 3.]
    β = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    m = 10

    function f(x)
        f = 0.0
        for i in 1:m
            t = 0.0
            for j in 1:4
                t += (x[j] - C[j,i])^2
            end
            f += 1 / (t + β[i])
        end
        return -f
    end

    function ∇f(x)
        df = zeros(4)
        for i in 1:m
            t = 0.0
            for j in 1:4
                t += (x[j] - C[j,i])^2
            end
            for j in 1:4
                df[j] += 2 * (x[j] - C[j,i]) / (t + β[i])^2
            end
        end
        return -df
    end

    bounds = [zeros(4) 10ones(4)]
    xopt = ([4.0, 4.0, 4.0, 4.0],)
    return TestFunction(4, bounds, xopt, f, ∇f)
end


function TestDropWave()
    f(x) = -(1 + cos(12*sqrt(sum(x.^2)))) / (0.5*sum(x.^2) + 2)
    
    function ∇f(x)
        t = 12 * sqrt(sum(x.^2))
        df = zeros(2)
        for i in 1:2
            df[i] = 12 * x[i] * sin(t) / sqrt(sum(x.^2)) / (0.5*sum(x.^2) + 2) - (1 + cos(t)) * (x[i] / (0.5*sum(x.^2) + 2)^2)
        end
        return -df
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -5.12
    bounds[:,2] .=  5.12
    xopt = ([0.0, 0.0],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestGriewank(d)
    f(x) = sum(x.^2) / 4000 - prod(cos.(x ./ sqrt.(collect(1:d)))) + 1
    
    function ∇f(x)
        df = zeros(d)
        for i in 1:d
            df[i] = x[i] / 2000 + prod(cos.(x ./ sqrt.(collect(1:d)))) * sin(x[i] / sqrt(i))
        end
        return df
    end

    bounds = zeros(d, 2)
    bounds[:,1] .= -600.0
    bounds[:,2] .=  600.0
    xopt = (zeros(d),)
    return TestFunction(d, bounds, xopt, f, ∇f)
end


function TestBohachevsky()
    f(x) = x[1]^2 + 2*x[2]^2 - 0.3*cos(3π*x[1]) - 0.4*cos(4π*x[2]) + 0.7
    
    function ∇f(x)
        df = zeros(2)
        df[1] = 2*x[1] + 0.9*π*sin(3π*x[1])
        df[2] = 4*x[2] + 1.6*π*sin(4π*x[2])
        return df
    end

    bounds = zeros(2, 2)
    bounds[:,1] .= -100.0
    bounds[:,2] .=  100.0
    xopt = ([0.0, 0.0],)
    return TestFunction(2, bounds, xopt, f, ∇f)
end


function TestGriewank(d)
    function f(x)
        sum = 0.0
        product = 1.0
        for i in 1:length(x)
            sum += x[i]^2
            product *= cos(x[i]/sqrt(i))
        end
        return 1 + sum/4000 - product
    end
    
    function ∇f(x)
        grad = zeros(length(x))
        for i in 1:length(x)
            sin_term = sin(x[i]/sqrt(i))
            cos_term = prod([cos(x[j]/sqrt(j)) for j=1:length(x) if j != i])
            grad[i] = 2*x[i]/4000 + sin_term * cos_term
        end
        return grad
    end

    bounds = zeros(d, 2)
    bounds[:, 1] = -600.
    bounds[:, 2] = 600.
    xopt = (zeros(d),)

    return TestFunction(d, bounds, xopt, f, ∇f)
end