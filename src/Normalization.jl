module Normalization

using Statistics

export ZScore,
       fit,
       normalize!,
       normalize,
       denormalize!,
       denormalize

abstract type AbstractNormalization end
(T::AbstractNormalization)(dims, p) = T(dims, p)

include.(["ZScore.jl", "Sigmoid.jl", "Robust.jl"])

fit!(T::AbstractNormalization, X::AbstractArray) = T.p = mapslices.(T.𝑝, (X,); dims=T.dims)
fit(𝒯::Type{<:AbstractNormalization}, X::AbstractArray; dims) = (T = 𝒯(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
normalize!(X::AbstractArray, 𝒯::Type{<:AbstractNormalization}; dims) = normalize!(X, fit(𝒯, X; dims))
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
    underidxs = CartesianIndices(Tuple([axes(x[1], i) for i ∈ underdims]))
    _mapdims!(x, f, underdims, underidxs)
end

function _mapdims!(x, f, underdims, underidxs)
    f!(x...) = (x[1] .= f(x...))
    for idxs ∈ Tuple.(underidxs)
        selector = _selectslice(underdims, idxs)
        f!(selector.(x)...)
    end
end

function _selectslice(dims, idxs)
    _selectdim(a, b) = x -> selectdim(x, a, b)
    st = sortperm([dims...])
    dims = dims[st] .- (0:length(dims)-1)
    idxs = idxs[st]
    ∘(reverse([_selectdim(dim, idxs[i]) for (i, dim) ∈ enumerate(dims)])...)
    #[(dim, idxs[i]) for (i, dim) ∈ enumerate(dims[st])]
end
selectslice(x, args...) = _selectslice(args...)(x)



end
