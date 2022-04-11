module Normalization

using Statistics

export ZScore,
       RobustZScore,
       fit,
       normalize!,
       normalize,
       denormalize!,
       denormalize

abstract type AbstractNormalization end
(𝒯::Type{<:AbstractNormalization})(dims) = 𝒯(;dims)
function (𝒯::Type{<:AbstractNormalization})(dims, p)
    isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
    𝒯(;dims, p)
end

Base.@kwdef mutable struct ZScore <: AbstractNormalization
    dims
    p::Union{Nothing, NTuple{2, AbstractArray}} = nothing
    𝑝::NTuple{2, Function} = (mean, std)
    𝑓::Function = (x, 𝜇, 𝜎)->(x .- 𝜇)./𝜎
    𝑓⁻¹::Function = (y, 𝜇, 𝜎) -> y.*𝜎 .+ 𝜇
end

iqr = x -> quantile(x[:], 0.75) - quantile(x[:], 0.25)
Base.@kwdef mutable struct RobustZScore <: AbstractNormalization
    dims
    p::Union{Nothing, NTuple{2, AbstractArray}} = nothing
    𝑝::NTuple{2, Function} = (median, iqr)
    𝑓::Function = (x, 𝜇, 𝜎)->1.35.*(x .- 𝜇)./𝜎 # ? Factor of 1.35 for consistency with SD of normal distribution
    𝑓⁻¹::Function = (y, 𝜇, 𝜎) -> y.*𝜎/1.35 .+ 𝜇
end

function fit!(T::AbstractNormalization, X::AbstractArray)
    dims = isnothing(T.dims) ? (1:ndims(X)) : T.dims
    T.p = mapslices.(T.𝑝, (X,); dims)
end
fit(𝒯::Type{<:AbstractNormalization}, X::AbstractArray; dims=nothing) = (T = 𝒯(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
normalize!(X::AbstractArray, 𝒯::Type{<:AbstractNormalization}; dims=nothing) = normalize!(X, fit(𝒯, X; dims))
normalize(X::AbstractArray, args...; kwargs...) = (Y=copy(X); normalize!(Y, args...; kwargs...); Y)

function denormalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && error("Cannot denormalize with an unfit normalization")
    mapdims!(T.𝑓⁻¹, X, T.p...; T.dims)
end
denormalize(X::AbstractArray, args...) = (Y=copy(X); denormalize!(Y, args...); Y)



"""
Map the function `f` over the `dims` of all of the arguments. `f` should accept the same number of arguments as there are variables in `x...`. The first element of `x` is the considered as the reference array, and all other arguments must have sizes consistent with the reference array, or equal to 1.
"""
function mapdims!(f, x...; dims)
    totaldims = 1:ndims(x[1])
    isnothing(dims) && (dims = totaldims)
    overdims = dims isa Vector ? dims : [dims...]
    @assert overdims isa Vector{Int}
    underdims = setdiff(totaldims, overdims)
    @assert all(all(size.(x[2:end], i) .== 1) for i ∈ overdims)
    @assert all(all(size(x[1], i) .== size.(x, i)) for i ∈ underdims)
    if sort([dims...]) == totaldims
        return (x[1] .= f.(x...))
    end
    _mapdims!(x, f, underdims, CartesianIndices(size(x[1])[underdims]))
end

function _mapdims!(x, f, dims, underidxs)
    f!(x...) = (x[1] .= f(x...))
    st = sortperm([dims...])
    dims = dims[st] .- (0:length(dims)-1)
    Threads.@threads for idxs ∈ underidxs
        f!(_selectslice(dims, Tuple(idxs)[st]).(x)...)
    end
end

_selectdim(a, b) = x -> selectdim(x, a, b)
_selectslice(dims, idxs) = ∘(reverse([_selectdim(dim, idxs[i]) for (i, dim) ∈ enumerate(dims)])...)
function selectslice(x, dims, idxs)
    st = sortperm([dims...])
    dims = dims[st] .- (0:length(dims)-1)
    idxs = Tuple(idxs)[st]
    _selectslice(dims, idxs)(x)
end



end
