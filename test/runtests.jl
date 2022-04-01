using Normalization
import Normalization: ZScore
using Statistics
using Test

@testset "Normalization" begin # Adapted from https://github.com/JuliaStats/StatsBase.jl/blob/e8ab26500d9a34ef358b2d3cf619ae41c71785fc/test/transformations.jl

    #* 2D array
    X = rand(5, 8)
    X_ = copy(X)

    # * ZScore a 2D array over the first dim.
    X = copy(X_)
    T = Normalization.fit(ZScore, X, dims=1)
    Y = Normalization.normalize(X, T)
    @test !isnothing(T.p)
    @test length(T.p) == 2
    @test length(T.p[1]) == length(T.p[2]) == size(X, 2)
    @test Y ≈ (X.-mean(X, dims=1))./std(X, dims=1)
    # @test reconstruct(T, Y) ≈ X
    # @test transform!(T, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(ZScoreTransform, X, dims=1, scale=false)
    # Y = transform(t, X)
    # @test length(t.mean) == 8
    # @test isempty(t.scale)
    # @test Y ≈ X .- mean(X, dims=1)
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
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
    # @test Y ≈ X_

    # X = copy(X_)
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
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(UnitRangeTransform, X, dims=1, unit=false)
    # Y = transform(t, X)
    # @test length(t.min) == 8
    # @test length(t.scale) == 8
    # @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
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
    # @test Y ≈ X_

    # X = copy(X_)
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
    # @test Y ≈ X_

    # # vector
    # X = rand(10)
    # X_ = copy(X)

    # t = fit(ZScoreTransform, X, dims=1, center=false, scale=false)
    # Y = transform(t, X)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(ZScoreTransform, X, dims=1, center=false)
    # Y = transform(t, X)
    # @test Y ≈ X ./ std(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(ZScoreTransform, X, dims=1, scale=false)
    # Y = transform(t, X)
    # @test Y ≈ X .- mean(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(ZScoreTransform, X, dims=1)
    # Y = transform(t, X)
    # @test Y ≈ (X .- mean(X, dims=1)) ./ std(X, dims=1)
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(ZScoreTransform, X, dims=1)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(UnitRangeTransform, X, dims=1)
    # Y = transform(t, X)
    # @test Y ≈ (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

    # X = copy(X_)
    # t = fit(UnitRangeTransform, X, dims=1, unit=false)
    # Y = transform(t, X)
    # @test Y ≈ X ./ (maximum(X, dims=1) .- minimum(X, dims=1))
    # @test transform(t, X) ≈ Y
    # @test reconstruct(t, Y) ≈ X
    # @test Y ≈ standardize(UnitRangeTransform, X, dims=1, unit=false)
    # @test transform!(t, X) === X
    # @test isequal(X, Y)
    # @test reconstruct!(t, Y) === Y
    # @test Y ≈ X_

end
