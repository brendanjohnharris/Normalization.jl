module Normalization

using Statistics

abstract type AbstractNormalization end
mutable struct ZScore <: AbstractNormalization
    dims
    p::Union{Nothing, NTuple{2, AbstractArray}}
    𝑝::NTuple{2, Function}
    𝑓::Function
    𝑓⁻¹::Function

    function ZScore(dims, p=nothing)
        isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
        𝑝 = (mean, std)
        𝑓 = (x, 𝜇, 𝜎)->(x - 𝜇)/𝜎
        𝑓⁻¹ = (y, 𝜇, 𝜎) -> y*𝜎 + 𝜇
        new(dims, p, 𝑝, 𝑓, 𝑓⁻¹)
    end
end
ZScore(dims, 𝜇, 𝜎) = ZScore(dims, (𝜇, 𝜎))

fit!(T::AbstractNormalization, X::AbstractArray) = T.p = mapslices.(T.𝑝, (X,); dims=T.dims)
fit(𝒯::Type{<:AbstractNormalization}, X::AbstractArray; dims) = (T = 𝒯(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
normalize!(X::AbstractArray, 𝒯::Type{<:AbstractNormalization}; dims) = normalize!(X, fit(𝒯, X; dims))
normalize(X::AbstractArray, args...) = (Y=copy(X); normalize!(Y, args...); Y)


"""
Map the function `f` over the `dims` of all of the arguments. `f` should accept the same number of (scalar) arguments as there are variables in `x...`. The first element of `x` is the considered as the reference array, and all other arguments must have sizes consistent with the reference array, or equal to 1.
"""
function mapdims!(f, x...; dims)
    totaldims = 1:ndims(x[1])
    isnothing(dims) && (dims = totaldims)
    overdims = dims isa Vector ? dims : [dims...]
    @assert overdims isa Vector{Int}
    underdims = setdiff(totaldims, overdims)
    # `dims` contains the dimensions
    #@assert all(all(size(x[1], i) .== size.(x, i) .|| size.(x, i) .== 1) for i ∈ totaldims)
    @assert all(all(size.(x[2:end], i) .== 1) for i ∈ overdims)
    @assert all(all(size(x[1], i) .== size.(x, i)) for i ∈ underdims)
    overidxs = [axes(x[1], i) for i ∈ overdims]
    idxs = [(i ∈ overdims) ? overidxs[i] : 1 for i ∈ totaldims]
    pidxs = ones(Int, size(totaldims))
    underidxs = CartesianIndices([axes(x[1], i) for i ∈ underdims]...)
    for i ∈ Tuple.(underidxs)
        idxs[underdims] .= i
        pidxs[underdims] .= i
        @inbounds x[1][idxs...] = f.(x[1][idxs...], [p[pidxs...] for p ∈ x[2:end]]...)
    end
end


end
