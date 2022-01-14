using LinearAlgebra, DelimitedFiles, Optim, Statistics, StatsPlots, Distributions, FastGaussQuadrature, Parameters

@with_kw mutable struct LRJ
    κ::Float64 = 1.0
    θ::Float64 = -1.0
    σ::Float64 = 0.1
    η::Float64 = 2.0
    λ::Float64 = 1.0
end

function simulate(m::LRJ; pathlen = 100, Δt = 1.0)
    @unpack κ, θ, σ, η, λ = m
    v = fill(0.0, pathlen)
    z = rand(Normal(0, σ * sqrt(Δt)), pathlen)
    q = rand(Poisson(λ * Δt), pathlen)
    y = rand(Exponential(1.0 / η), pathlen)
    a, b = (1.0 - κ), κ * θ
    v[1] = θ
    for j ∈ 2:pathlen
        v[j] = a * v[j-1] + b + z[j] + y[j] * q[j]
    end
    return v
end
