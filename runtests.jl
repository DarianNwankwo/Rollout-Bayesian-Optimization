using StatsFuns
using LinearAlgebra
using Test

include("lazy_struct.jl")
include("low_discrepancy.jl")
include("optim.jl")
include("rbf_surrogate.jl")
include("testfns.jl")

function centered_fd(f, u, du, h)
    (f(u+h*du)-f(u-h*du))/(2h)
end

# General fd check harness
function fd_check(f, df, u, du, h; name="")
    dfu_fd = centered_fd(f, u, du, h)
    dfu_ex = df(u)'*du
    @test dfu_fd ≈ dfu_ex rtol=1e-7
end

# Sanity check analytic gradients via finite differencing
function kernel_fd_check(rbf, ρ, θ, h=1e-6)
    rbfθ = rbf(θ)

    @test centered_fd(rbfθ.ψ, ρ, 1.0, h) ≈ rbfθ.Dρ_ψ(ρ) rtol=1e-8
    @test centered_fd(rbfθ.Dρ_ψ, ρ, 1.0, h) ≈ rbfθ.Dρρ_ψ(ρ) rtol=1e-8
    
    ∇θ_ψ_ex = rbfθ.∇θ_ψ(ρ)
    for k = 1:length(θ)
        
        θp = copy(θ)
        θp[k] += h
        rbfθp = rbf(θp)
        
        θm = copy(θ)
        θm[k] -= h
        rbfθm = rbf(θm)

        ∇θk_ψ_fd = (rbfθp(ρ)-rbfθm(ρ))/(2h)
        @test ∇θk_ψ_fd ≈ ∇θ_ψ_ex[k] rtol=1e-8
    end
end

@testset "Test radial basis functions" begin
    kernel_fd_check(kernel_matern12, 0.123, [0.456])
    kernel_fd_check(kernel_matern32, 0.123, [0.456])
    kernel_fd_check(kernel_matern52, 0.123, [0.456])
    kernel_fd_check(kernel_SE, 0.123, [0.456])
    kernel_fd_check(kernel_invmq, 0.123, [0.456])
    kernel_fd_check(kernel_mq, 0.123, [0.456])
    kernel_fd_check(kernel_poly, 0.123, [])
    kernel_fd_check(kernel_polylog, 0.123, [])
end

@testset "Test kernels based on RBFs" begin
    ψ = kernel_SE([1.0])
    r = rand(4)
    dr = rand(4)
    fd_check((u) -> eval_k(ψ, u), (u) -> eval_∇k(ψ, u), r, dr, 1e-6)
    fd_check((u) -> eval_∇k(ψ, u), (u) -> eval_Hk(ψ, u), r, dr, 1e-6)
end

rosen = TestRosenbrock()
xy = rand(2)
dxy = rand(2)

@testset "Test test fun derivative correctness" begin
    testf = TestRosenbrock()
    fd_check(testf.f, testf.∇f, xy, dxy, 1e-6; name="rosenbrock")
    testf = TestRastrigin(2)
    fd_check(testf.f, testf.∇f, xy, dxy, 1e-6; name="rastrigin")
    testf = TestAckley(2)
    fd_check(testf.f, testf.∇f, xy, dxy, 1e-6; name="ackley")
    testf = TestSixHump()
    fd_check(testf.f, testf.∇f, xy, dxy, 1e-6; name="sixhump")
    testf = TestBranin()
    fd_check(testf.f, testf.∇f, xy, dxy, 1e-6; name="branin")
    testf = TestGramacyLee()
    fd_check(testf.f, testf.∇f, [xy[1]], [dxy[1]], 1e-6; name="branin")
end

npts = 20
Xdemo = 4.0 * kronecker_quasirand(2, npts) .- 2.0
Xdemo[2,:] .+= 1.0
ψ = kernel_matern52([1.0])
s = fit_surrogate(ψ, Xdemo, rosen.f)

@testset "Spline quantity derivatives" begin
    fd_check((x) -> s(x).μ,  (x) -> s(x).∇μ, xy, dxy, 1e-5)
    fd_check((x) -> s(x).σ,  (x) -> s(x).∇σ, xy, dxy, 1e-5)
    fd_check((x) -> s(x).∇μ, (x) -> s(x).Hμ, xy, dxy, 1e-5)
    fd_check((x) -> s(x).∇σ, (x) -> s(x).Hσ, xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).z,
             (x) -> eval(s, x, 3340).∇z,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).∇z,
             (x) -> eval(s, x, 3340).Hz,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).EI,
             (x) -> eval(s, x, 3340).∇EI,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).∇EI,
             (x) -> eval(s, x, 3340).HEI,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).logEI,
             (x) -> eval(s, x, 3340).∇logEI,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 40).logEI,
             (x) -> eval(s, x, 40).∇logEI,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 3340).∇logEI,
             (x) -> eval(s, x, 3340).HlogEI,
             xy, dxy, 1e-5)
    fd_check((x) -> eval(s, x, 40).∇logEI,
             (x) -> eval(s, x, 40).HlogEI,
             xy, dxy, 1e-5)
end

@testset "Spline derivatives wrt data locations" begin
    δXdemo = rand(2, npts)
    h = 1e-5

    s  = fit_surrogate(ψ, Xdemo, rosen.f)
    sp = fit_surrogate(ψ, Xdemo + h*δXdemo, rosen.f)
    sm = fit_surrogate(ψ, Xdemo - h*δXdemo, rosen.f)
    δs = fit_δsurrogate(s, δXdemo, rosen.∇f)

    spx = sp([0.0, 0.0])
    smx = sm([0.0, 0.0])
    sx = s([0.0, 0.0])
    δsx = δs(sx)

    @test (sp.K-sm.K)/(2h) ≈ δs.K rtol=1e-7
    @test (sp.y-sm.y)/(2h) ≈ δs.y rtol=1e-7
    @test (sp.c-sm.c)/(2h) ≈ δs.c rtol=1e-7

    @test (spx.μ-smx.μ)/(2h) ≈ δsx.μ rtol=1e-7
    @test (spx.σ-smx.σ)/(2h) ≈ δsx.σ rtol=1e-7
    @test (spx.z-smx.z)/(2h) ≈ δsx.z rtol=1e-7
    @test (spx.EI-smx.EI)/(2h) ≈ δsx.EI rtol=1e-7
    @test (spx.∇μ-smx.∇μ)/(2h) ≈ δsx.∇μ rtol=1e-7
    @test (spx.∇σ-smx.∇σ)/(2h) ≈ δsx.∇σ rtol=1e-7
    @test (spx.∇EI-smx.∇EI)/(2h) ≈ δsx.∇EI rtol=1e-7

    xopt, sxopt = tr_newton_EI(s, [1.0, 1.0], rtol=1e-12, nsteps=200)
    xoptp, sxoptp = tr_newton_EI(sp, xopt, rtol=1e-12, nsteps=200)
    xoptm, sxoptm = tr_newton_EI(sm, xopt, rtol=1e-12, nsteps=200)

    δxopt_ex = -sxopt.HEI \ δs(sxopt).∇EI
    δxopt_fd = (xoptp-xoptm)/(2h)
    @test δxopt_ex ≈ δxopt_fd rtol=1e-5

    δoptEI_fd = (sxoptp.EI-sxoptm.EI)/(2h)
    δoptEI_ex = δs(sxopt).EI
    @test δoptEI_fd ≈ δoptEI_ex rtol=1e-7
end
