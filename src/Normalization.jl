#! format: off
module Normalization

using Statistics
import LinearAlgebra:   normalize,
                        normalize!
import StatsAPI: fit
using Accessors

export  fit,
        fit!,
        normalize!,
        normalize,
        denormalize!,
        denormalize,
        nansafe,
        AbstractNormalization,
        @_Normalization,
        ZScore,
        RobustZScore,
        MixedSigmoid,
        Sigmoid,
        RobustSigmoid,
        MixedSigmoid,
        MinMax,
        Center,
        RobustCenter,
        UnitEnergy,
        OutlierSuppress,
        RobustOutlierSuppress,
        HalfZScore,
        RobustHalfZScore


abstract type AbstractNormalization{T} end
function (𝒯::Type{<:AbstractNormalization})(dims, p)
    isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
    𝒯(;dims, p)
end

macro _Normalization(name, 𝑝, 𝑓, 𝑓⁻¹)
    :(mutable struct $(esc(name)){T} <: AbstractNormalization{T}
        dims
        p::NTuple{length($𝑝), AbstractArray{T}}
        𝑝::NTuple{length($𝑝), Function}
        𝑓::Function
        𝑓⁻¹::Function
     end;
     ($(esc(name))){T}(; dims = nothing,
                         p = ntuple(_->Vector{T}(), length($𝑝)),
                         𝑝 = $𝑝,
                         𝑓 = $𝑓,
                         𝑓⁻¹ = $𝑓⁻¹) where T = $(esc(name))(((isnothing(dims) || length(dims) < 2) ? dims : collect(dims)), p, 𝑝, 𝑓, 𝑓⁻¹);
     ($(esc(name)))(; kwargs...) = ($(esc(name))){Nothing}(; kwargs...);
     )
end

halfstd(x, args...; kwargs...) = std(x, args...; kwargs...)./convert(eltype(x), sqrt(1-(2/π)))

# * Common normalizations
@_Normalization ZScore (mean, std)         (x, 𝜇, 𝜎) -> x .= (x .- 𝜇)./𝜎  #=
                                        =# (y, 𝜇, 𝜎) -> y .= y.*𝜎 .+ 𝜇
@_Normalization HalfZScore (minimum, halfstd) (x, 𝜇, 𝜎) -> x .= (x .- 𝜇)./𝜎  #=
                                        =# (y, 𝜇, 𝜎) -> y .= y.*𝜎 .+ 𝜇
@_Normalization Sigmoid (mean, std)        (x, 𝜇, 𝜎)->x.=1.0./(1 .+exp.(.-(x.-𝜇)./𝜎)) #=
                                        =# (y, 𝜇, 𝜎) -> y .= .-𝜎.*log.(1.0./y .- 1) .+ 𝜇
@_Normalization MinMax (minimum, maximum)  (x, l, u) -> x .= (x.-l)./(u-l) #=
                                        =# (y, l, u) -> y .= (u-l).*y .+ l
@_Normalization Center (mean,)             (x, 𝜇) -> x .= x .- 𝜇     (y, 𝜇) -> y .= y .+ 𝜇
@_Normalization RobustCenter (median,)     Center().𝑓   Center().𝑓⁻¹
@_Normalization UnitEnergy (x->sum(x.^2),) #=
                                        =# (x, 𝐸) -> x .= x./sqrt.(𝐸) #=
                                        =# (y, 𝐸) -> y .= y.*sqrt.(𝐸)
@_Normalization OutlierSuppress (mean, std) #=
                                        =# (x, 𝜇, 𝜎) -> (x[x .- 𝜇 .> 5.0.*𝜎] .= 𝜇 .+ 5.0.*𝜎, x[𝜇 .- x .> 5.0.*𝜎] .= 𝜇 .- 5.0.*𝜎) #=
                                        =# (y, 𝜇, 𝜎) -> identity # No denormalization here

# * Robust versions of typical 2-parameter normalizations
common_norms = [:ZScore, :Sigmoid, :OutlierSuppress, :HalfZScore]
function _iqr(x::AbstractArray{T})::T where {T}
    eltype(x).((quantile(x[:], 0.75) - quantile(x[:], 0.25))/1.35) # ? Divide by 1.35 so that std(x) ≈ _iqr(x) when x contains normally distributed values
end
_robustNorm(N::Symbol; name="Robust"*string(N)|>Symbol) = eval(:(@_Normalization $name (median, _iqr) ($N)().𝑓 ($N)().𝑓⁻¹))
_robustNorm.(common_norms)

# * Mixed versions of typical 2-parameter normalizations
mixedcenter(x) = (_iqr(x) == 0) ? mean(x) : median(x)
mixedscale(x) = (𝜎 = _iqr(x); 𝜎 == 0 ? std(x) : 𝜎)
_mixedNorm(N::Symbol; name="Mixed"*string(N)|>Symbol) = eval(:(@_Normalization $name (mixedcenter, mixedscale) ($N)().𝑓 ($N)().𝑓⁻¹))
_mixedNorm.(common_norms)

# * NaN-safe versions
_nansafe(p) = x -> p(filter(!isnan, x))
nansafe!(T::AbstractNormalization) = (T.𝑝=_nansafe.(T.𝑝); ())
nansafe(T::AbstractNormalization) = (N = deepcopy(T); nansafe!(N); N)
nansafe(𝒯::Type{<:AbstractNormalization}; dims=nothing) = 𝒯(; dims) |> nansafe

function nansafe(f::Function; dims = nothing)
    function g(x)
        isnothing(dims) && (dims = 1:ndims(x))
        mapslices(y -> f(filter(!isnan, y)), x; dims = dims)
    end
end

reshape(args...; kwargs...) = Base.reshape(args...; kwargs...)
reshape(x::Number, dims...) = reshape([x], dims...)
Base.eltype(::AbstractNormalization{T}) where {T} = T
Base.eltype(::Type{<:AbstractNormalization{T}}) where {T} = T

negdims(dims, n)::NTuple{N, Integer} where {N} = filter(i->!(i in dims), 1:n) |> Tuple

function fit!(T::AbstractNormalization, X::AbstractArray; dims=nothing)
    𝒳 = eltype(X)
    𝒯 = eltype(T)
    @assert 𝒳 == 𝒯 "$𝒯 type does not match data type ($𝒳)"
    dims = isnothing(dims) ? (1:ndims(X)) : dims
    dims = (length(dims) > 1 ? collect(dims) : dims) |> Tuple
    _dims = negdims(dims, ndims(X)) |> Tuple
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T.dims = dims
    T.p = reshape.(map.(T.𝑝, (eachslice(X; dims=_dims, drop=false),)), psz...)
    nothing
end
function fit(T::AbstractNormalization{Nothing}, X::AbstractArray; dims=nothing)
    dims = isnothing(dims) ? (1:ndims(X)) : dims
    dims = (length(dims) > 1 ? collect(dims) : dims)
    _dims = negdims(dims, ndims(X)) |> Tuple
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T = @set T.dims = dims
    T = @set T.p = reshape.(map.(T.𝑝, (eachslice(X; dims=_dims, drop=false),)), psz...)
end

fit(𝒯::Type{<:AbstractNormalization}, X; dims=nothing) = fit(𝒯(), X; dims)
# fit(T::AbstractNormalization, X::AbstractArray; kw...) = fit(N, X; kw...)
(𝒯::Type{<:AbstractNormalization})(X; dims=nothing) = fit(𝒯, X; dims)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
NormUnion = Union{AbstractNormalization, Type{<:AbstractNormalization}}
normalize!(X, 𝒯::NormUnion; dims=nothing) = normalize!(X, fit(𝒯, X; dims))
normalize(X, T::NormUnion; kwargs...) = (Y=copy(X); normalize!(Y, T; kwargs...); Y)

(T::AbstractNormalization)(X) = normalize(X, T)

function denormalize!(X::AbstractArray, T::AbstractNormalization)
    any(isempty.(T.p)) && error("Cannot denormalize with an unfit normalization")
    mapdims!(T.𝑓⁻¹, X, T.p...; T.dims)
end
denormalize(X, args...) = (Y=copy(X); denormalize!(Y, args...); Y)


function _mapdims!(f, xs::Slices{<:AbstractArray}, ys)
    Threads.@threads for i in eachindex(xs)
        f(xs[i], getindex.(ys, i)...)
    end
end

"""
Map the function `f` over the `dims` of all of the arguments.
`f` should accept the same number of arguments as there are variables in `x...`.
The first element of `x` is the considered as the reference array, and all other arguments
must have sizes consistent with the reference array (along the specified `dims`), or equal
to 1 (along the remaining dims).
"""
function mapdims!(f, x::AbstractArray{T, n}, y...; dims) where {T, n}
    isnothing(dims) && (dims = 1:n)
     max(dims...) <= n || error("A chosen dimension is greater than the number of dimensions of the reference array")
    unique(dims) == [dims...] || error("Repeated dimensions")
    length(dims) == n && return f(x, y...) # ? Shortcut for global normalisation
    all(all(size.(y, i) .== 1) for i ∈ dims) || error("Inconsistent dimensions; dimensions $dims must have size 1")

    negs = negdims(dims, n)
    all(all(size(x, i) .== size.(y, i)) for i ∈ negs) || error("Inconsistent dimensions; dimensions $negs must have size $(size(x)[collect(negs)])")

    xs = eachslice(x; dims=negs)
    ys = eachslice.(y; dims=negs)
    _mapdims!(f, xs, ys)
end


end
