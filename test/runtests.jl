using Test
using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    import StatsBase as SB
    using BenchmarkTools
    using DataFrames
    using Normalization
    using Statistics
    using Unitful
    using DimensionalData
    using SparseArrays
    using Random
end

@testitem "negdims" setup = [Setup] begin
    import Normalization.negdims
    X = randn(100, 200, 300)
    @test negdims(2, ndims(X)) == (1, 3)
    @test negdims(1:2, ndims(X)) == (3,)
    @test negdims([1], ndims(X)) == (2, 3)
end

@testitem "Mapdims" setup = [Setup] begin
    import Normalization.mapdims!
    _X = randn(1000, 200, 300)
    X = deepcopy(_X)
    Y = 1.0 ./ _X[:, 1, 1]
    f = (x, y) -> (x .= x .^ 2 .+ y .^ 2)
    mapdims!(f, X, Y; dims=(2, 3))
    @test X == _X .^ 2 .+ Y .^ 2
end

@testitem "Constructor" setup = [Setup] begin
    p = ([0], [1])
    dims = 1
    N = @test_nowarn ZScore(dims, p)
    @test N isa AbstractNormalization

    p = (randn(3, 3), randn(3, 3))
    dims = [1, 2]
    N = @test_nowarn ZScore(dims, p)
    @test N isa AbstractNormalization
end

@testitem "1D normalization" setup = [Setup] begin
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

@testitem "1D half z-score normalization" setup = [Setup] begin
    # * Check this normalization is correct
    _X = abs.(randn(100000))
    X = copy(_X)
    @test mean(X) ≈ 1 * sqrt(2 / pi) rtol = 2e-2
    @test var(X) ≈ 1 - (2 / pi) rtol = 2e-2

    T = fit(HalfZScore, X)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ X rtol = 1e-2

    _X = _X .+ 10
    X = copy(_X)
    T = fit(HalfZScore, X)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ X .- 10 rtol = 1e-2

    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
    @test eltype(Y) == eltype(_X) == eltype(T)
end

normalizations = [ZScore, RobustZScore, Sigmoid, RobustSigmoid, MinMax, Center, RobustCenter, UnitEnergy, HalfZScore, RobustHalfZScore]
forward_normalizations = [OutlierSuppress, RobustOutlierSuppress]
for N in normalizations
    @testitem "$N" setup = [Setup] begin
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
for N in forward_normalizations
    @testitem "$N" setup = [Setup] begin
        _X = rand(100)
        X = copy(_X)
        T = fit(N, X)
        Y = normalize(X, T)
        @test !isnothing(T.p)
        @test_nowarn normalize!(X, T)
        @test X == Y

        _X = Float32.(_X)
        X = copy(_X)
        T = fit(N, X)
        Y = normalize(X, T)
        @test !isnothing(T.p)
        @test_nowarn normalize!(X, T)
        @test X == Y
    end
end

@testitem "2D normalization" setup = [Setup] begin
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

for N in normalizations
    @testitem "$N 2D" setup = [Setup] begin
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

@testitem "3D normalization" setup = [Setup] begin
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

    # * Multiple dims
    X = copy(_X)
    T1 = fit(ZScore, X, dims=[2, 3])
    T = fit(ZScore, X, dims=(2, 3))
    @test T1.dims == T.dims == [2, 3]
    @test T1.p == T.p
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ axes(Z, 1)
        x = @view Z[i, :, :]
        x .= (x .- mean(x)) ./ std(x)
    end
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test size(T.p[1])[1] == size(T.p[2])[1] == size(X)[1]
    @test Y ≈ Z
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end

@testitem "Sigmoid" setup = [Setup] begin
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

@testitem "RobustSigmoid" setup = [Setup] begin
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

@testitem "Nansafe" setup = [Setup] begin
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

@testitem "NaNZScore" setup = [Setup] begin
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

@testitem "NaNSigmoid" setup = [Setup] begin
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

@testitem "ND normalization" setup = [Setup] begin
    Nmax = 5
    for _ = 1:10
        ds = rand(2:Nmax)
        sz = rand(2:ds, ds)
        _X = rand(sz...)
        Nnorm = rand(1:Nmax÷2)
        normdims = unique(rand(1:ds, Nnorm))
        notnormdims = setdiff(1:ndims(_X), normdims)
        X = copy(_X)
        T = fit(ZScore, X, dims=normdims[randperm(length(normdims))]) # Randomize dims order
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

@testitem "Unitful Normalization compat" setup = [Setup] begin
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

@testitem "DimensionalData compat" setup = [Setup] begin
    # * 1D array
    _X = rand(100)
    _X = DimArray(_X, (Ti(1:size(_X, 1)),))
    X = copy(_X)
    T = @test_nowarn fit(ZScore, X, dims=1)
    Z = @test_nowarn normalize(X, T)
    N = @test_nowarn ZScore(X; dims=1) # Alternate syntax
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

    #* 2D array
    _X = rand(10, 5)
    _X = DimArray(_X, (Ti(1:size(_X, 1)), Y(1:size(_X, 2))))
    X = copy(_X)

    # * ZScore a 2D array over the first dim.
    T = @test_nowarn fit(ZScore, X, dims=1)
    Z = @test_nowarn normalize(X, T)
    N = @test_nowarn ZScore(X; dims=1) # Alternate syntax
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

@testitem "StatsBase comparison" setup = [Setup] begin
    X = rand(1000, 50)
    Y = normalize(X, ZScore; dims=1)
    Z = SB.standardize(SB.ZScoreTransform, X; dims=1)
    @test Y ≈ Z
end

@testitem "Benchmark" setup = [Setup] begin
    println("\nNormalization.jl (in-place)")
    display(@benchmark normalize!(X, ZScore; dims=1) setup = X = rand(1000, 10000))
    println("\nNormalization.jl (out-of-place)")
    display(@benchmark normalize(X, ZScore; dims=1) setup = X = rand(1000, 10000))
    println("\nStatsBase.jl")
    display(@benchmark SB.standardize(SB.ZScoreTransform, X; dims=1) setup = X = rand(1000, 10000))
end

@testitem "DataFrames ext" setup = [Setup] begin
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

@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(Normalization)
end
