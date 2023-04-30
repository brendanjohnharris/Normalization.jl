import StatsBase as SB
using BenchmarkTools
using Normalization
using Statistics
using Test

@testset "1D normalization" begin
    # * 1D array
    _X = rand(100)
    X = copy(_X)
    T = fit(ZScore, X)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X.-mean(X))./std(X)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end

normalizations = [ZScore, RobustZScore, Sigmoid, RobustSigmoid, MinMax, Center, RobustCenter, UnitEnergy]
for N in normalizations
    @testset "$N" begin
        _X = rand(100)
        X = copy(_X)
        T = fit(N, X)
        Y = normalize(X, T)
        @test !isnothing(T.p)
        @test denormalize(Y, T) ≈ X
        @test_nowarn normalize!(X, T)
        @test X == Y
        @test_nowarn denormalize!(Y, T)
        @test Y ≈ _X
    end
end



@testset "2D normalization" begin # Adapted from https://github.com/JuliaStats/StatsBase.jl/blob/e8ab26500d9a34ef358b2d3cf619ae41c71785fc/test/transformations.jl

    #* 2D array
    _X = rand(10, 5)
    X = copy(_X)

    # * ZScore a 2D array over the first dim.
    T = fit(ZScore, X, dims=1)
    Y = normalize(X, T)
    N = ZScore(X; dims=1) # Alternate syntax
    @test N(X) == Y
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test Y ≈ (X.-mean(X, dims=1))./std(X, dims=1)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end

@testset "3D normalization" begin
    _X = randn(10, 10, 100)
    X = copy(_X)
    T = fit(ZScore, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= (x.-mean(x))./std(x)
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1:2] == size(T.p[2])[1:2] == size(X)[1:2]
    @test Y ≈ Z
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end


@testset "Sigmoid" begin
    _X = randn(10, 10, 100)
    X = copy(_X)
    T = fit(Sigmoid, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= 1.0./(1 .+ exp.(.-(x.-mean(x))./std(x)))
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1:2] == size(T.p[2])[1:2] == size(X)[1:2]
    @test Y ≈ Z
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end

@testset "RobustSigmoid" begin
    _X = randn(10, 10, 100)
    X = copy(_X)
    T = fit(RobustSigmoid, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= 1.0./(1 .+ exp.(.-(x.-median(x))./(SB.iqr(x)./1.35)))
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1:2] == size(T.p[2])[1:2] == size(X)[1:2]
    @test Y ≈ Z
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end


@testset "NaNZScore" begin
    _X = randn(10, 10, 100)
    idxs = rand(1:prod(size(_X)), 100)
    _X[idxs] .= NaN
    X = copy(_X)
    T = fit(nansafe(ZScore), X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= (x.-mean(filter(!isnan, x)))./std(filter(!isnan, x))
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1:2] == size(T.p[2])[1:2] == size(X)[1:2]
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, Z[:])
    @test filter!(!isnan, denormalize(Y, T)[:]) ≈ filter!(!isnan, X[:])
    @test_nowarn normalize!(X, T)
    @test filter!(!isnan, X[:]) == filter!(!isnan, Y[:])
    @test_nowarn denormalize!(Y, T)
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, _X[:])
end


@testset "NaNSigmoid" begin
    _X = randn(10, 10, 100)
    idxs = rand(1:prod(size(_X)), 100)
    _X[idxs] .= NaN
    X = copy(_X)
    T = fit(nansafe(Sigmoid), X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= 1.0./(1 .+ exp.(.-(x.-mean(filter(!isnan, x)))./(std(filter(!isnan, x)))))
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1:2] == size(T.p[2])[1:2] == size(X)[1:2]
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, Z[:])
    @test filter!(!isnan, denormalize(Y, T)[:]) ≈ filter!(!isnan, X[:])
    @test_nowarn normalize!(X, T)
    @test filter!(!isnan, X[:]) == filter!(!isnan, Y[:])
    @test_nowarn denormalize!(Y, T)
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, _X[:])
end


@testset "ND normalization" begin
    Nmax = 5
    for _ = 1:10
        ds = rand(2:Nmax)
        sz = rand(2:ds, ds)
        _X = rand(sz...)
        Nnorm = rand(1:Nmax÷2)
        normdims = unique(rand(1:ds, Nnorm))
        notnormdims = setdiff(1:ndims(_X), normdims)
        X = copy(_X)
        T = fit(ZScore, X, dims=normdims)
        Y = normalize(X, T)
        @test !isnothing(T.p)
        @test length(T.p) == 2
        @test size(T.p[1])[notnormdims] == size(T.p[2])[notnormdims] == size(X)[notnormdims]
        @test denormalize(Y, T) ≈ X
        @test_nowarn normalize!(X, T)
        @test X == Y
        @test_nowarn denormalize!(Y, T)
        @test Y ≈ _X
    end
end

@testset "StatsBase comparison" begin
    X = rand(1000, 50)
    Y = normalize(X, ZScore; dims=1)
    Z = SB.standardize(SB.ZScoreTransform, X; dims=1)
    @test Y ≈ Z
end

println("\nNormalization.jl")
display(@benchmark normalize(rand(1000, 10000), ZScore; dims=1))
println("\nStatsBase.jl")
display(@benchmark SB.standardize(SB.ZScoreTransform, rand(1000, 10000); dims=1))
