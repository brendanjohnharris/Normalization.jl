using TestItems
using TestItemRunner
using Normalization

@run_package_tests

@testsnippet Setup begin
    using Test
    import StatsBase as SB
    using BenchmarkTools
    using DataFrames
    using Normalization
    using Statistics
    using Unitful
    using DimensionalData
    using SparseArrays
    using Random
    import Normalization: params
end

@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(Normalization; unbound_args=false) # unbound_args=true
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
    function f!(y)
        function _f!(x)
            x^2 + y^2
        end
    end
    mapdims!(f!, X, (Y,); dims=(2, 3))
    @test X == _X .^ 2 .+ Y .^ 2
end

@testitem "Constructor" setup = [Setup] begin
    p = ([0], [1])
    dims = 1
    N = @inferred ZScore(dims, p)
    @test N isa AbstractNormalization

    p = (randn(3, 3), randn(3, 3))
    dims = [1, 2]
    N = @inferred ZScore(dims, p)
    @test N isa AbstractNormalization
end

@testitem "Inverses" setup = [Setup] begin
    using InverseFunctions
    using Test
    import Normalization: zscore, sigmoid, minmax, center, unitenergy

    invnorms = [Center, UnitEnergy, ZScore, Sigmoid, MinMax]

    invnorms = map(invnorms) do invnorm
        ps = rand(length(Normalization.estimators(invnorm)))
        return Normalization.forward(invnorm)(ps...)
    end
    InverseFunctions.test_inverse.(invnorms, randn())
end

@testitem "1D normalization" setup = [Setup] begin
    # * 1D array
    _X = rand(100)
    X = copy(_X)

    @inferred ZScore{Float64}(; p=([0], [1]))
    @inferred ZScore(nothing, ([0], [1]))
    @inferred fit(ZScore, X)
    N = @inferred ZScore{Float64}()
    @inferred normalize(X, N)

    T = @inferred fit(ZScore, X)
    Y = @inferred normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X .- mean(X)) ./ std(X)
    D = @inferred denormalize(Y, T)
    @test D ≈ X
    @test_nowarn @inferred normalize!(X, T)
    @test X == Y
    @test_nowarn @inferred denormalize!(Y, T)
    @test Y ≈ _X
    @test eltype(Y) == eltype(_X) == eltype(T)

    T = @inferred ZScore{Float64}(; dims=2)
    @test_throws DimensionMismatch fit!(T, X)

    T = @inferred ZScore{Float64}(; dims=1)
    @test_nowarn fit!(T, X)

    T = @inferred ZScore{Float64}()
    @test_nowarn fit!(T, X; dims=1)
    @test Normalization.dims(T) == 1

    _X = rand(100)
    X = copy(_X)
    _T = deepcopy(T)
    @inferred fit!(T, X)
    Y = @inferred normalize(X, T)
    @test _T != T
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == 1 == length(T.p[2])
    @test Y ≈ (X .- mean(X)) ./ std(X)
    @test @inferred denormalize(Y, T) ≈ X
    @test_nowarn @inferred normalize!(X, T)
    @test X == Y
    @test_nowarn @inferred denormalize!(Y, T)
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
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test length(params(T)[1]) == 1 == length(params(T)[2])
    @test Y ≈ X rtol = 1e-2

    _X = _X .+ 10
    X = copy(_X)
    T = fit(HalfZScore, X)
    Y = normalize(X, T)
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test length(params(T)[1]) == 1 == length(params(T)[2])
    @test Y ≈ X .- 10 rtol = 1e-2

    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
    @test eltype(Y) == eltype(_X) == eltype(T)
end

@testitem "Outlier suppress" setup = [Setup] begin
    # * Check this normalization is correct
    using Statistics
    _X = abs.(randn(10000))
    X = copy(_X)
    X[1:5] .= 10000

    T = fit(OutlierSuppress, X)
    Y = normalize(X, T)
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test length(params(T)[1]) == 1 == length(params(T)[2])
    @test all(Y[1:5] .== (std(X) * 5.0 + mean(X)))


    T = fit(Robust{OutlierSuppress}, X)
    Y = normalize(X, T)
    @test Normalization.normalization(T) isa OutlierSuppress
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test length(params(T)[1]) == 1 == length(params(T)[2])
    @test all(Y[1:5] .< X[1:5])
    @test all(Y[1:5] .< 10)
end

normalizations = [ZScore, Robust{ZScore}, Sigmoid, Robust{Sigmoid}, MinMax, Center, Robust{Center}, UnitEnergy, HalfZScore, Robust{HalfZScore}]
forward_normalizations = [OutlierSuppress, Robust{OutlierSuppress}]
for N in normalizations
    @testitem "$N" setup = [Setup] begin
        _X = rand(100)
        X = copy(_X)
        T = @inferred fit(N, X)
        Y = @inferred normalize(X, T)
        @test !isnothing(params(T))
        @test @inferred denormalize(Y, T) ≈ X
        @test_nowarn @inferred normalize!(X, T)
        @test X == Y
        @test_nowarn @inferred denormalize!(Y, T)
        @test Y ≈ _X

        _X = Float32.(_X)
        X = copy(_X)
        T = @inferred fit(N, X)
        Y = @inferred normalize(X, T)
        @test !isnothing(params(T))
        @test @inferred denormalize(Y, T) ≈ X
        @test_nowarn @inferred normalize!(X, T)
        @test X == Y
        @test_nowarn @inferred denormalize!(Y, T)
        @test Y ≈ _X
    end
end
for N in forward_normalizations
    @testitem "$N" setup = [Setup] begin
        @test Normalization.normalization(N) == N
        @test !isfit(N)

        _X = rand(100)
        X = copy(_X)
        T = @inferred fit(N, X)
        Y = @inferred normalize(X, T)
        @test fit(T, randn(100)) != T
        @test !isnothing(T.p)
        @test_nowarn @inferred normalize!(X, T)
        @test X == Y

        _X = Float32.(_X)
        X = copy(_X)
        T = @inferred fit(N, X)
        Y = @inferred normalize(X, T)
        @test !isnothing(T.p)
        @test_nowarn @inferred normalize!(X, T)
        @test X == Y

        @test_throws "Cannot denormalize" denormalize!(X, N; dims=1)
    end
end

@testitem "2D normalization" setup = [Setup] begin
    #* 2D array
    _X = rand(10, 5)
    X = copy(_X)

    # * ZScore a 2D array over the first dim.
    T = @inferred fit(ZScore, X, dims=1)
    Y = @inferred normalize(X, T)
    N = @inferred ZScore(X; dims=1) # Alternate syntax
    @test N(X) == Y
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    @test @inferred denormalize(Y, T) ≈ X
    @test_nowarn @inferred normalize!(X, T)
    @test X == Y
    @test_nowarn @inferred denormalize!(Y, T)
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
    T = @inferred fit(ZScore, X, dims=3)
    Y = @inferred normalize(X, T)
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
    @test all(T1.dims .== T.dims .== [2, 3])
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

    N = @inferred Normalization.Robust{Sigmoid}(X, dims=3)
    n = @inferred ZScore{Float64}(; dims=1)

    # * Preferred patterns
    N = Normalization.Robust{Sigmoid{Float64}}()
    fit!(N, X)

    N = Normalization.Robust{Sigmoid}
    T = fit(N, X, dims=3)
    Y = normalize(X, T)

    @test params(T)[1] == median(X, dims=3)
    @test params(T)[2] == mapslices(SB.iqr, X, dims=3) ./ 1.35

    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= 1.0 ./ (1 .+ exp.(.-(x .- median(x)) ./ (SB.iqr(x) ./ 1.35)))
    end
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test size(params(T)[1])[1:2] == size(params(T)[2])[1:2] == size(X)[1:2]
    @test Y ≈ Z
    @inferred denormalize(Y, T)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X
end


@testitem "RobustCenter" setup = [Setup] begin
    _X = randn(10, 10, 100)
    X = copy(_X)

    N = @inferred Normalization.Robust{Center}(X, dims=3)

    # * Preferred patterns
    N = Normalization.Robust{Center{Float64}}(; dims=nothing)
    fit!(N, X)

    N = Normalization.Robust{Center}
    T = fit(N, X, dims=3)
    Y = normalize(X, T)

    @test params(T)[1] == median(X, dims=3)

    @test !isnothing(params(T))
    @test length(params(T)) == 1
    @inferred denormalize(Y, T)
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
    N = NaNSafe{ZScore}
    T = fit(N, X)
    Y = normalize(X, T)
    Z = copy(_X)
    Z = (Z .- mean(filter(!isnan, Z))) ./ std(filter(!isnan, Z))
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, Z[:])
    @test filter!(!isnan, denormalize(Y, T)[:]) ≈ filter!(!isnan, X[:])
    @test_nowarn normalize!(X, T)
    @test filter!(!isnan, X[:]) == filter!(!isnan, Y[:])
    @test_nowarn denormalize!(Y, T)
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, _X[:])

    # * 2D array
    _X = randn(100, 10)
    idxs = rand(1:prod(size(_X)), 100)
    _X[idxs] .= NaN

    @test all(!isnan, nansafe(sum)(X, dims=2))

    X = copy(_X)
    N = NaNSafe{ZScore}
    T = @test_nowarn fit(N, X; dims=2)
    Y = @test_nowarn normalize(X, T)
end

@testitem "NaNZScore" setup = [Setup] begin
    _X = randn(10, 10, 100)
    idxs = rand(1:prod(size(_X)), 100)
    _X[idxs] .= NaN
    X = copy(_X)
    T = fit(NaNSafe{ZScore}, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= (x .- mean(filter(!isnan, x))) ./ std(filter(!isnan, x))
    end
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test size(params(T)[1])[1:2] == size(params(T)[2])[1:2] == size(X)[1:2]
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
    T = fit(NaNSafe{Sigmoid}, X, dims=3)
    Y = normalize(X, T)
    Z = copy(_X)
    for i ∈ CartesianIndices((axes(Z, 1), axes(Z, 2)))
        x = @view Z[i, :]
        x .= 1.0 ./ (1 .+ exp.(.-(x .- mean(filter(!isnan, x))) ./ (std(filter(!isnan, x)))))
    end
    @test !isnothing(params(T))
    @test length(params(T)) == 2
    @test size(params(T)[1])[1:2] == size(params(T)[2])[1:2] == size(X)[1:2]
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, Z[:])
    @test filter!(!isnan, denormalize(Y, T)[:]) ≈ filter!(!isnan, X[:])
    @test_nowarn normalize!(X, T)
    @test filter!(!isnan, X[:]) == filter!(!isnan, Y[:])
    @test_nowarn denormalize!(Y, T)
    @test filter!(!isnan, Y[:]) ≈ filter!(!isnan, _X[:])
end

@testitem "Mixed" setup = [Setup] begin
    using StatsBase
    _X = [NaN, 0.0, 0.5, 0.5, 0.5, 1.0] # ? Has iqr 0
    X = copy(_X)
    N = NaNSafe{Mixed{ZScore}}
    T = fit(N, X)
    Y = normalize(X, T)
    Z = copy(_X)
    Z = (Z .- mean(filter(!isnan, Z))) ./ std(filter(!isnan, Z))
    @test (Z.≈Y)[2:end] |> all

    _X = [NaN, 0.0, 0.5, 0.5, 1.0]
    X = copy(_X)
    N = NaNSafe{Mixed{ZScore}}
    T = fit(N, X)
    Y = normalize(X, T)
    Z = copy(_X)
    Z = (Z .- mean(filter(!isnan, Z))) ./ std(filter(!isnan, Z))
    @test (Z.≈Y)[2:end] |> !all
    @test nansafe(iqr)(Y) ≈ 1.35

    @test !isnothing(params(T))
    @test length(params(T)) == 2
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

    @test denormalize(Y, T) ≈ X
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
    T = fit(Robust{ZScore}, X, dims=1)
    Z = normalize(X, T)
    N = Robust{ZScore}(X; dims=1)
    @test N(X) == Z
    @test !isnothing(params(T))
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
    using DataFrames
    _X = DataFrame(
        :A => [18.1, 19.5, 21.1],
        :B => [16.2, 17.2, 17.5],
        :C => [12.8, 13.1, 14.4],
    )

    X = copy(_X)

    # * ZScore a DataFrame over the first dim.
    T = fit(ZScore, X, dims=1)
    Z = normalize(X, T)
    @test T(X) == Z
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
    @test T(X) == Z
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
