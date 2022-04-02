import StatsBase as SB
using BenchmarkTools
using Normalization
using Statistics
using Test

@testset "2D normalization" begin # Adapted from https://github.com/JuliaStats/StatsBase.jl/blob/e8ab26500d9a34ef358b2d3cf619ae41c71785fc/test/transformations.jl

    #* 2D array
    _X = rand(10, 5)
    X = copy(_X)

    # * ZScore a 2D array over the first dim.
    T = fit(ZScore, X, dims=1)
    Y = normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test Y ≈ (X.-mean(X, dims=1))./std(X, dims=1)
    @test denormalize(Y, T) ≈ X
    @test_nowarn normalize!(X, T)
    @test X == Y
    @test_nowarn denormalize!(Y, T)
    @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=1, scale=false)
    # Y = transform(t, X)
    # @test length(t.mean) == 8
    # @test isempty(t.scale)
    # @test Y ≈ X .- mean(X, dims=1)
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=1)
    # Y = transform(t, X)
    # @test length(t.mean) == 8
    # @test length(t.scale) == 8
    # @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(ZScoreTransform, X, dims=1)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=2)
    # Y = transform(t, X)
    # @test length(t.mean) == 5
    # @test length(t.scale) == 5
    # @test Y ≈ (X .- mean(X, dims=2)) ./ std(X, dims=2)
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(ZScoreTransform, X, dims=2)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(UnitRangeTransform, X, dims=1, unit=false)
    # Y = transform(t, X)
    # @test length(t.min) == 8
    # @test length(t.scale) == 8
    # @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(UnitRangeTransform, X, dims=1)
    # Y = transform(t, X)
    # @test isa(t, AbstractDataTransform)
    # @test length(t.min) == 8
    # @test length(t.scale) == 8
    # @test Y ≈ (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(UnitRangeTransform, X, dims=1)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(UnitRangeTransform, X, dims=2)
    # Y = transform(t, X)
    # @test isa(t, AbstractDataTransform)
    # @test length(t.min) == 5
    # @test length(t.scale) == 5
    # @test Y ≈ (X .- minimum(X, dims=2)) ./ (maximum(X, dims=2) .- minimum(X, dims=2))
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # # vector
    # X = rand(10)
    # _X = copy(X)

    # t = fit(ZScoreTransform, X, dims=1, center=false, scale=false)
    # Y = transform(t, X)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=1, center=false)
    # Y = transform(t, X)
    # @test Y ≈ X ./ std(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=1, scale=false)
    # Y = transform(t, X)
    # @test Y ≈ X .- mean(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(ZScoreTransform, X, dims=1)
    # Y = transform(t, X)
    # @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(ZScoreTransform, X, dims=1)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(UnitRangeTransform, X, dims=1)
    # Y = transform(t, X)
    # @test Y ≈ (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X

    # X = copy(_X)
    # t = fit(UnitRangeTransform, X, dims=1, unit=false)
    # Y = transform(t, X)
    # @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(UnitRangeTransform, X, dims=1, unit=false)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ _X
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
