#! format: off
module Normalization

import LinearAlgebra:   normalize,
                        normalize!
import StatsAPI: fit, fit!, params
import InverseFunctions: inverse

export  fit,
        fit!,
        isfit,
        params,
        params!,
        normalize!,
        normalize,
        denormalize!,
        denormalize,
        AbstractNormalization,
        @_Normalization

"""
    AbstractNormalization
Abstract type for normalizations.

## Constructors
You can work with `AbstractNormalization`s as either types or instances. The type approach
is useful for concise code, whereas the instance approach is useful for performant
mutations.
In the examples below we use the `ZScore` normalization, but the same syntax applies to all
`Normalization`s.

### Fit to a type
```julia
    X = randn(100, 10)
    N = fit(ZScore, X; dims=nothing) # eltype inferred from X
    N = fit(ZScore{Float32}, X; dims=nothing) # eltype set to Float32
    N isa AbstractNormalization && N isa ZScore # Returns a concrete AbstractNormalization
```

### Fit to an instance
```julia
    X = randn(100, 10)
    N = ZScore{Float64}(; dims=2) # Initializes with empty parameters
    N isa AbstractNormalization && N isa ZScore # Returns a concrete AbstractNormalization
    !isfit(N)

    fit!(N, X; dims=1) # Fit normalization in-place, and update the `dims`
    Normalization.dims(N) == 1
```

## Normalization and denormalization
With a fit normalization, there are two approaches to normalizing data: in-place and
out-of-place.
```julia
    _X = copy(X)
    normalize!(_X, N) # Normalizes in-place, updating _X
    Y = normalize(X, N) # Normalizes out-of-place, returning a new array
    normalize(X, ZScore; dims=1) # For convenience, fits and then normalizes
```
For most normalizations, there is a corresponding denormalization that
transforms data to the original space.
```julia
    Z = denormalize(Y, N) # Denormalizes out-of-place, returning a new array
    Z ≈ X
    denormalize!(Y, N) # Denormalizes in-place, updating Y
```

## Properties and traits
### Type traits
- `Normalization.estimators(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the estimators `N` as a tuple of functions
- `forward(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the forward normalization function (e.g. x-> x - 𝜇 / 𝜎 for the `ZScore`)
- inverse(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization}})` returns the inverse normalization function e.g. `forward(N)(ps...) |> InverseFunctions.inverse`
- `eltype(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the eltype of the normalization parameters

### Concrete properties
- `Normalization.dims(N::<:AbstractNormalization)` returns the dimensions of the normalization. The dimensions are determined by `dims` and correspond to the mapped slices of the input array.
- `params(N::<:AbstractNormalization)` returns the parameters of `N` as a tuple of arrays. The dimensions of arrays are the complement of `dims`.
- `isfit(N::<:AbstractNormalization)` checks if all parameters are non-empty

"""
abstract type AbstractNormalization{T} end
const NormUnion = Union{<:AbstractNormalization, Type{<:AbstractNormalization}}

function forward end
macro _Normalization(name, 𝑝, 𝑓)
    :(mutable struct $(esc(name)){T} <: AbstractNormalization{T}
        dims
        p::NTuple{length($𝑝), AbstractArray{T}}
     end;
     Normalization.estimators(::Type{N}) where {N<:$(esc(name))} = $𝑝;
     Normalization.forward(::Type{N}) where {N<:$(esc(name))} = $𝑓;
     ($(esc(name))){T}(; dims = nothing, p = ntuple(_->Vector{T}(), length($𝑝))) where {T} = $(esc(name))(dims, p);
     )
end

# * Interface traits
estimators(::N) where {N<:AbstractNormalization} = estimators(N)
function inverse(::Type{N}) where {N<:AbstractNormalization}
    function _inverse(ps...,)
        inverse(forward(N)(ps...))
    end
end
forward(::N) where {N<:AbstractNormalization} = forward(N)
inverse(::N) where {N<:AbstractNormalization} = inverse(N)
normalization(::Type{N}) where {N<:AbstractNormalization} = N
Base.eltype(::AbstractNormalization{T}) where {T} = T
Base.eltype(::Type{<:AbstractNormalization{T}}) where {T} = T
isfit(::Type{N}) where {N<:AbstractNormalization} = false

# * Interface properties
dims(N::AbstractNormalization) = N.dims
params(N::AbstractNormalization) = N.p
normalization(N::AbstractNormalization) = N
isfit(T::AbstractNormalization) = !all(isempty, params(T))

# * Mutators
function dims!(N::AbstractNormalization, ds)
    normalization(N).dims = ds
end
function params!(N::AbstractNormalization, ps)
    all(x->x==ps[1], length.(ps)) && error("Inconsistent parameter dimensions")
    normalization(N).p = ps
end

function __mapdims!(z, f, x, y)
    @inbounds map!(f(map(only, y)...), z, x)
end
function _mapdims!(zs::Slices{<:AbstractArray}, f, xs::Slices{<:AbstractArray}, ys::NTuple{N, <:AbstractArray}) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        __mapdims!(zs[i], f, xs[i], y)
    end
end
function mapdims!(z, f, x::AbstractArray{T, n}, y; dims) where {T, n}
    isnothing(dims) && (dims = 1:n)
     max(dims...) <= n || error("A chosen dimension is greater than the number of dimensions of the reference array")
    unique(dims) == [dims...] || error("Repeated dimensions")
    length(dims) == n && return __mapdims!(z, f, x, y) # ? Shortcut for global normalisation
    all(all(size.(y, i) .== 1) for i ∈ dims) || error("Inconsistent dimensions; dimensions $dims must have size 1")

    negs = negdims(dims, n)
    all(all(size(x, i) .== size.(y, i)) for i ∈ negs) || error("Inconsistent dimensions; dimensions $negs must have size $(size(x)[collect(negs)])")

    xs = eachslice(x; dims=negs)
    zs = eachslice(z; dims=negs)
    ys = eachslice.(y; dims=negs)
    _mapdims!(zs, f, xs, ys)
end

# * mapdims! with same input & output
mapdims!(f, x, y; kwargs...) = mapdims!(x, f, x, y; kwargs...)

# * Fitting
reshape(args...; kwargs...) = Base.reshape(args...; kwargs...)
reshape(x::Number, dims...) = reshape([x], dims...)
negdims(dims, n)::NTuple{N, Integer} where {N} = filter(i->!(i in dims), 1:n) |> Tuple
@inline function dimparams(dims, X)
    nd = ndims(X)
    if dims === nothing
        dims = collect(1:nd)
    else
        (isempty(dims) || maximum(dims) ≤ nd) ||
            throw(DimensionMismatch("Chosen dimension is greater than ndims(X)=$nd"))
    end
    sz  = size(X)
    flag = ntuple(i -> 0, nd)
    for d in dims
        @inbounds flag = Base.setindex(flag, 1, d)
    end
    nps = ntuple(i -> (flag[i] == 1 ? 1 : sz[i]), nd)

    return dims, nps
end
function fit!(T::AbstractNormalization, X::AbstractArray{A}; dims=Normalization.dims(T)) where {A}
    eltype(T) == A || throw(TypeError(:fit!, "Normalization", eltype(T), X))

    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(T)) do f
        reshape(map(f, Xs), nps...)
    end

    dims!(T, dims)
    params!(T, ps)
    nothing
end

function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing) where {A,T,𝒯<:AbstractNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(map(f, Xs), nps...)
    end
    𝒯(dims, ps)
end
function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing) where {A,𝒯<:AbstractNormalization}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(map(f, Xs), nps...)
    end
    𝒯{A}(dims, ps)
end
fit(N::AbstractNormalization, X::AbstractArray{A}; dims=Normalization.dims(N)) where {A} = fit(typeof(N), X; dims)

# * Normalizations
function normalize!(Z::AbstractArray, X::AbstractArray, T::AbstractNormalization)
    dims = Normalization.dims(T)
    isfit(T) || fit!(T, X; dims)
    mapdims!(Z, forward(T), X, params(T); dims)
    return nothing
end
function normalize!(Z, X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractNormalization}
    normalize!(Z, X, fit(𝒯, X; kwargs...))
end
normalize!(X, T::NormUnion; kwargs...) = normalize!(X, X, T; kwargs...)

function normalize(X, T::AbstractNormalization; kwargs...)
    Y = copy(X)
    normalize!(Y, T; kwargs...)
    return Y
end
function normalize(X, ::Type{𝒯}; kwargs...) where {𝒯 <: AbstractNormalization}
    normalize(X, fit(𝒯, X; kwargs...))
end

NotFitError() = error("Cannot denormalize with a normalization that has not been fitted")
function denormalize!(Z::AbstractArray, X::AbstractArray, T::AbstractNormalization)
    isfit(T) || NotFitError()
    mapdims!(Z, inverse(T), X, params(T); dims=Normalization.dims(T))
    return nothing
end
function denormalize!(Z, X,::Type{𝒯}; dims=nothing) where {𝒯 <: AbstractNormalization}
    NotFitError()
end
denormalize!(X, T::NormUnion; kwargs...) = denormalize!(X, X, T; kwargs...)
function denormalize(X, args...)
    Y = copy(X)
    denormalize!(Y, args...)
    return Y
end

# * Additional constructors
(::Type{𝒯})(X::AbstractArray; dims=nothing) where {𝒯<:AbstractNormalization} = fit(𝒯, X; dims)
(T::AbstractNormalization)(X::AbstractArray{A}) where {A} = normalize(X, T)

include("Normalizations.jl")
include("Modifiers.jl")

end
