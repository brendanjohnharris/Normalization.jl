#! format: off
module Normalization

using Statistics
import Accessors: @o
import LinearAlgebra:   normalize,
                        normalize!
import StatsAPI: fit, fit!, params
import InverseFunctions: inverse

export  fit,
        fit!,
        params,
        params!,
        normalize!,
        normalize,
        denormalize!,
        denormalize,
        nansafe,
        AbstractNormalization,
        @_Normalization


abstract type AbstractNormalization{T} end
# function (::Type{𝒯})(dims, p::NTuple{N, AbstractArray{T}}; kwargs...) where {𝒯 <: AbstractNormalization, N, T}
#     (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
#     𝒯{T}(; dims, p, kwargs...)
# end
# function (::Type{𝒯})(dims, p::NTuple{N, AbstractArray{T}}; kwargs...) where {N, T, 𝒯 <: AbstractNormalization{T}}
#     (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
#     𝒯{T}(; dims, p, kwargs...)
# end


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

# * AbstractNormalization interface
dims(N::AbstractNormalization) = N.dims
params(N::AbstractNormalization) = N.p
estimators(::N) where {N<:AbstractNormalization} = estimators(N)

function dims!(N::AbstractNormalization, ds)
    normalization(N).dims = ds
end
function params!(N::AbstractNormalization, ps)
    all(x->x==ps[1], length.(ps)) && error("Inconsistent parameter dimensions")
    normalization(N).p = ps
end

function inverse(::Type{N}) where {N<:AbstractNormalization}
    function _inverse(ps...,)
        inverse(forward(N)(ps...))
    end
end
forward(::N) where {N<:AbstractNormalization} = forward(N)
inverse(::N) where {N<:AbstractNormalization} = inverse(N)
# forward!(N::AbstractNormalization) = x -> map!(forward(N)(params(N)...), x, x)
# inverse!(N::AbstractNormalization) = x -> map!(inverse(N)(params(N)...), x, x)

function _mapdims!(f, xs::Slices{<:AbstractArray}, ys)
    Threads.@threads for i in eachindex(xs)
        y = getindex.(ys, i)
        map!(f(only.(y)...), xs[i], xs[i])
    end
end

function mapdims!(f, x::AbstractArray{T, n}, y...; dims) where {T, n}
    isnothing(dims) && (dims = 1:n)
     max(dims...) <= n || error("A chosen dimension is greater than the number of dimensions of the reference array")
    unique(dims) == [dims...] || error("Repeated dimensions")
    length(dims) == n && return map!(f(only.(y)...), x, x) # ? Shortcut for global normalisation
    all(all(size.(y, i) .== 1) for i ∈ dims) || error("Inconsistent dimensions; dimensions $dims must have size 1")

    negs = negdims(dims, n)
    all(all(size(x, i) .== size.(y, i)) for i ∈ negs) || error("Inconsistent dimensions; dimensions $negs must have size $(size(x)[collect(negs)])")

    xs = eachslice(x; dims=negs)
    ys = eachslice.(y; dims=negs)
    _mapdims!(f, xs, ys)
end

function (::Type{<:AbstractNormalization})(; kwargs...)
    throw(ArgumentError("Supply a type and dimensions to create an unfit Normalization, e.g. `ZScore{Float64}(; dims=1:2)`"))
end

reshape(args...; kwargs...) = Base.reshape(args...; kwargs...)
reshape(x::Number, dims...) = reshape([x], dims...)
normalization(N::AbstractNormalization) = N
normalization(::Type{N}) where {N<:AbstractNormalization} = N
Base.eltype(::AbstractNormalization{T}) where {T} = T
Base.eltype(::Type{<:AbstractNormalization{T}}) where {T} = T

negdims(dims, n)::NTuple{N, Integer} where {N} = filter(i->!(i in dims), 1:n) |> Tuple
function dimparams(dims, X)
    (isnothing(dims) || maximum(dims) ≤ ndims(X)) || throw(DimensionMismatch("Chosen dimension is greater than the number of dimensions of the reference array"))
    dims = isnothing(dims) ? (1:ndims(X)) : dims # !!!! Pull this out and make common funcitonf or fit! as well...
    dims = (length(dims) > 1 ? collect(dims) : dims)
    nps = size(X) |> collect
    nps[[dims...]] .= 1
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

# fit(T::AbstractNormalization, X::AbstractArray; kw...) = fit(N, X; kw...)
(::Type{𝒯})(X::AbstractArray; dims=nothing) where {𝒯<:AbstractNormalization} = fit(𝒯, X; dims)

isfit(T::AbstractNormalization) = !all(isempty, params(T))
function normalize!(X::AbstractArray, T::AbstractNormalization)
    isfit(T) || fit!(T, X)
    mapdims!(forward(T), X, params(T)...; dims=Normalization.dims(T))
    return nothing
end
function normalize!(X, ::Type{𝒯}; dims=nothing) where {𝒯 <: AbstractNormalization}
    normalize!(X, fit(𝒯, X; dims))
end
function normalize(X, T::AbstractNormalization; kwargs...)
    Y = copy(X)
    normalize!(Y, T; kwargs...)
    return Y
end

(T::AbstractNormalization)(X::AbstractArray{A}) where {A} = normalize(X, T)

NotFitError() = error("Cannot denormalize with a normalization that has not been fitted")
function denormalize!(X::AbstractArray, T::AbstractNormalization)
    isfit(T) || NotFitError()
    mapdims!(inverse(T), X, params(T)...; dims=Normalization.dims(T))
    return nothing
end
function denormalize!(X, ::Type{𝒯}; dims=nothing) where {𝒯 <: AbstractNormalization}
    NotFitError()
end
function denormalize(X, args...)
    Y = copy(X)
    denormalize!(Y, args...)
    return Y
end

include("Normalizations.jl")
include("Modifiers.jl")

end
