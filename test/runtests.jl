import StatsBase as SB
using BenchmarkTools
using DataFrames
using Normalization
using Statistics
using Unitful
using DimensionalData
using SparseArrays
using Test

@testset "Constructor" begin
    p = ([0], [1])
    dims = 1
    ZScore(dims, p)

    p = (randn(3, 3), randn(3, 3))
    dims = [1, 2]
    ZScore(dims, p)
end


@testset "1D normalization" begin
    # * 1D array
    _X = rand(100)
    X = copy(_X)
    T = fit(ZScore, X)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X .- mean(X)) ./ std(X)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
    @test eltype(Y) == eltype(_X) == eltype(T)

    _X = rand(100)
    X = copy(_X)
    _T = deepcopy(T)
    fit!(T, X)
    Y = normalize(X, T)
    @test _T != T
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X .- mean(X)) ./ std(X)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
    @test eltype(Y) == eltype(_X) == eltype(T)
end

normalizations = [ZScore, RobustZScore, Sigmoid, RobustSigmoid, MinMax, Center, RobustCenter, UnitEnergy, HalfZScore]
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

        _X = Float32.(_X)
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
    @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end

normalizations = [ZScore, RobustZScore, Sigmoid, RobustSigmoid, MinMax, Center, RobustCenter, UnitEnergy, HalfZScore]
for N in normalizations
    @testset "$N 2D" begin
        #* 2D array
        _X = rand(10, 5)
        X = copy(_X)

        T = fit(N, X; dims=1)
        Y = normalize(X, T)
        @test !isnothing(T.p)
        @test denormalize(Y, T) ≈ X
        @test_nowarn normalize!(X, T)
        @test X == Y
        @test_nowarn denormalize!(Y, T)
        @test Y ≈ _X
    end
end


@testset "3D normalization" begin
    _X = randn(10, 10, 100)
    X = copy(_X)
    T = fit(ZScore, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= (x .- mean(x)) ./ std(x)
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
        x .= 1.0 ./ (1 .+ exp.(.-(x .- mean(x)) ./ std(x)))
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
        x .= 1.0 ./ (1 .+ exp.(.-(x .- median(x)) ./ (SB.iqr(x) ./ 1.35)))
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

@testset "Nansafe" begin
    _X = randn(1000)
    idxs = rand(1:prod(size(_X)), 100)
    _X[idxs] .= NaN
    X = copy(_X)
    T = fit(nansafe(ZScore), X)
    Y = normalize(X, T)
    Z = copy(_X)
    Z = (Z .- mean(filter(!isnan, Z))) ./ std(filter(!isnan, Z))
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, Z[:])
    @test filter!(!isnan, denormalize(Y, T)[:]) ≈ filter!(!isnan, X[:])
    @test_nowarn normalize!(X, T)
    @test filter!(!isnan, X[:]) == filter!(!isnan, Y[:])
    @test_nowarn denormalize!(Y, T)
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, _X[:])
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
        x .= (x .- mean(filter(!isnan, x))) ./ std(filter(!isnan, x))
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
        x .= 1.0 ./ (1 .+ exp.(.-(x .- mean(filter(!isnan, x))) ./ (std(filter(!isnan, x)))))
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


@testset "Unitful Normalization compat" begin
    _X = rand(100) * u"V"
    X = copy(_X)
    T = fit(ZScore, X)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X .- mean(X)) ./ std(X)
    @test eltype(X) == eltype(_X) == eltype(T)
end


@testset "DimensionalData compat" begin
    #* 2D array
    _X = rand(10, 5)
    _X = DimArray(_X, (Ti(1:size(_X, 1)), Y(1:size(_X, 2))))
    X = copy(_X)

    # * ZScore a 2D array over the first dim.
    T = fit(ZScore, X, dims=1)
    Z = normalize(X, T)
    N = ZScore(X; dims=1) # Alternate syntax
    @test N(X) == Z
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test Z ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    @test denormalize(Z, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Z
    @test_nowarn denormalize!(Z, T)
    @test Z ≈ _X

    # * ZScore a 3D array over the first dim.
    _X = DimArray(randn(5, 6, 7), (Dim{:a}(1:5), Dim{:b}(1:6), Dim{:c}(1:7)))
    X = copy(_X)
    T = fit(RobustZScore, X, dims=1)
    Z = normalize(X, T)
    N = RobustZScore(X; dims=1)
    @test N(X) == Z
    @test !isnothing(T.p)
end

# @testset "SparseArrays compat" begin
#     #* 2D array
#     _X = sprand(100000, 50, 1e-4)
#     _X = DimArray(_X, (Ti(1:size(_X, 1)), Y(1:size(_X, 2))));
#     X = deepcopy(_X);

#     # * ZScore a 2D array over the first dim.
#     T = fit(ZScore, X, dims=1)
#     Z = normalize(X, T)
#     N = ZScore(X; dims=1) # Alternate syntax
#     @test N(X) == Z
#     @test !isnothing(T.p)
#     @test length(T.p) == 2
#     @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
#     @test Z ≈ (X.-mean(X, dims=1))./std(X, dims=1)
#     @test denormalize(Z, T) ≈ X
#     @test_nowarn normalize!(X, T)
#     @test X == Z
#     @test_nowarn denormalize!(Z, T)
#     @test Z ≈ _X
# end


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

@testset "DataFrames ext" begin

    _X = DataFrame(
        :temperature_A => [18.1, 19.5, 21.1],
        :temperature_B => [16.2, 17.2, 17.5],
        :temperature_C => [12.8, 13.1, 14.4],
    )

    X = copy(_X)

    # * ZScore a DataFrame over the first dim.
    T = fit(ZScore, X, dims=1)
    Z = normalize(X, T)
    N = ZScore(X; dims=1) # Alternate syntax
    @test N(X) == Z
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test all(DataFrames.Tables.matrix(Z) .≈ (DataFrames.Tables.matrix(X) .- mean(DataFrames.Tables.matrix(X), dims=1)) ./ std(DataFrames.Tables.matrix(X), dims=1))
    @test denormalize(Z, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X ≈ Z
    @test_nowarn denormalize!(Z, T)
    @test Z ≈ _X

    # * ZScore a dataframe over the second dim.
    X = copy(_X)
    T = fit(ZScore, X, dims=2)
    Z = normalize(X, T)
    N = ZScore(X; dims=2) # Alternate syntax
    @test N(X) == Z
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 1)
    @test all(DataFrames.Tables.matrix(Z) .≈ (DataFrames.Tables.matrix(X) .- mean(DataFrames.Tables.matrix(X), dims=2)) ./ std(DataFrames.Tables.matrix(X), dims=2))
    @test denormalize(Z, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X ≈ Z
    @test_nowarn denormalize!(Z, T)
    @test Z ≈ _X
end
