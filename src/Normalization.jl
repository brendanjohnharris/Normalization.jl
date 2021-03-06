module Normalization

using Statistics
using JuliennedArrays
import LinearAlgebra:   normalize,
                        normalize!

export  fit,
        fit!,
        normalize!,
        normalize,
        denormalize!,
        denormalize,
        nansafe,
        ZScore,
        RobustZScore,
        Sigmoid,
        RobustSigmoid,
        MinMax

abstract type AbstractNormalization end
(๐ฏ::Type{<:AbstractNormalization})(dims) = ๐ฏ(;dims)
function (๐ฏ::Type{<:AbstractNormalization})(dims, p)
    isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
    ๐ฏ(;dims, p)
end
(T::AbstractNormalization)(dims) = dims == () || (T.dims = dims)

macro _Normalization(name, ๐, ๐, ๐โปยน)
    :(mutable struct $(esc(name)) <: AbstractNormalization
        dims
        p::Union{Nothing, NTuple{length($๐), AbstractArray}}
        ๐::NTuple{length($๐), Function}
        ๐::Function
        ๐โปยน::Function
     end;
     ($(esc(name)))(; dims = nothing,
                         p = nothing,
                         ๐ = $๐,
                         ๐ = $๐,
                         ๐โปยน = $๐โปยน) = $(esc(name))(dims, p, ๐, ๐, ๐โปยน)
     )
end

# * Common normalizations
@_Normalization ZScore (mean, std)         (x, ๐, ๐) -> x .= (x .- ๐)./๐  #=
                                        =# (y, ๐, ๐) -> y .= y.*๐ .+ ๐
@_Normalization Sigmoid (mean, std)        (x, ๐, ๐)->x.=1.0./(1 .+exp.(.-(x.-๐)./๐)) #=
                                        =# (y, ๐, ๐) -> y .= .-๐.*log.(1.0./y .- 1) .+ ๐
@_Normalization MinMax (minimum, maximum)  (x, l, u) -> x .= (x.-l)./(u-l) #=
                                        =# (y, l, u) -> y .= (u-l).*y .+ l
@_Normalization Center (mean,)             (x, ๐) -> x .= x .- ๐     (y, ๐) -> y .= y .+ ๐
@_Normalization RobustCenter (median,)     Centre().๐   Centre().๐โปยน

# * Robust versions of typical 2-parameter normalizations
common_norms = [:ZScore, :Sigmoid,]
_iqr = x -> (quantile(x[:], 0.75) - quantile(x[:], 0.25))/1.35 # ? Divide by 1.35 so that std(x) โ _iqr(x) when x contains normally distributed values
_robustNorm(N::Symbol; name="Robust"*string(N)|>Symbol) = eval(:(@_Normalization $name (median, _iqr) ($N)().๐ ($N)().๐โปยน))
_robustNorm.(common_norms)

# * Mixed versions of typical 2-parameter normalizations
mixedcenter(x) = (_iqr(x) == 0) ? mean(x) : median(x)
mixedscale(x) = (๐ = _iqr(x); ๐ == 0 ? std(x) : ๐)
_mixedNorm(N::Symbol; name="Mixed"*string(N)|>Symbol) = eval(:(@_Normalization $name (mixedcenter, mixedscale) ($N)().๐ ($N)().๐โปยน))
_mixedNorm.(common_norms)

# * NaN-safe versions
_nansafe(p) = x -> p(filter(!isnan, x))
nansafe!(T::AbstractNormalization) = (T.๐=_nansafe.(T.๐); ())
nansafe(T::AbstractNormalization) = (N = deepcopy(T); nansafe!(N); N)
nansafe(๐ฏ::Type{<:AbstractNormalization}; dims=nothing) = dims |> ๐ฏ |> nansafe

Base.reshape(x::Number, dims...) = reshape([x], dims...)
function fit!(T::AbstractNormalization, X::AbstractArray; dims=())
    T(dims)
    dims = isnothing(T.dims) ? (1:ndims(X)) : T.dims
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T.p = reshape.(map.(T.๐, (Slices(X, dims...),)), psz...)
end
fit(T::AbstractNormalization, X::AbstractArray; kw...)=(T=deepcopy(T); fit!(T, X; kw...); T)
fit(๐ฏ::Type{<:AbstractNormalization}, X::AbstractArray; dims=nothing) = (T = ๐ฏ(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.๐, X, T.p...; T.dims)
end
NormUnion = Union{AbstractNormalization, Type{<:AbstractNormalization}}
normalize!(X::AbstractArray, ๐ฏ::NormUnion; dims=nothing) = normalize!(X, fit(๐ฏ, X; dims))
normalize(X::AbstractArray, T::NormUnion; kwargs...) = (Y=copy(X); normalize!(Y, T; kwargs...); Y)

function denormalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && error("Cannot denormalize with an unfit normalization")
    mapdims!(T.๐โปยน, X, T.p...; T.dims)
end
denormalize(X::AbstractArray, args...) = (Y=copy(X); denormalize!(Y, args...); Y)

"""
Map the function `f` over the `dims` of all of the arguments.
`f` should accept the same number of arguments as there are variables in `x...`.
The first element of `x` is the considered as the reference array, and all other arguments must have sizes consistent with the reference array, or equal to 1.
"""
function mapdims!(f, x...; dims)
    n = ndims(x[1])
    isnothing(dims) && (dims = 1:n)
    dims = sort([dims...])
    @assert max(dims...) <= n
    @assert unique(dims) == dims
    length(dims) == n && return f(x...) # Shortcut for global normalisation
    negdims = Base._negdims(n, dims)
    @assert all(all(size.(x[2:end], i) .== 1) for i โ dims)
    @assert all(all(size(x[1], i) .== size.(x, i)) for i โ negdims)
    idxs = Base.compute_itspace(x[1], (negdims...,)|>Val)
    Threads.@threads for i โ idxs # map(f!, Slices.(x, dims...)...)
        selectslice = x -> view(x, i...)
        f(selectslice.(x)...)
    end
end

end
