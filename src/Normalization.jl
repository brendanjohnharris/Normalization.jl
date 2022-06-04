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
(𝒯::Type{<:AbstractNormalization})(dims) = 𝒯(;dims)
function (𝒯::Type{<:AbstractNormalization})(dims, p)
    isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
    𝒯(;dims, p)
end
(T::AbstractNormalization)(dims) = dims == () || (T.dims = dims)

macro _Normalization(name, 𝑝, 𝑓, 𝑓⁻¹)
    :(mutable struct $(esc(name)) <: AbstractNormalization
        dims
        p::Union{Nothing, NTuple{length($𝑝), AbstractArray}}
        𝑝::NTuple{length($𝑝), Function}
        𝑓::Function
        𝑓⁻¹::Function
     end;
     ($(esc(name)))(; dims = nothing,
                         p = nothing,
                         𝑝 = $𝑝,
                         𝑓 = $𝑓,
                         𝑓⁻¹ = $𝑓⁻¹) = $(esc(name))(dims, p, 𝑝, 𝑓, 𝑓⁻¹)
     )
end

# * Common normalizations
@_Normalization ZScore (mean, std)         (x, 𝜇, 𝜎) -> x .= (x .- 𝜇)./𝜎  #=
                                        =# (y, 𝜇, 𝜎) -> y .= y.*𝜎 .+ 𝜇
@_Normalization Sigmoid (mean, std)        (x, 𝜇, 𝜎)->x.=1.0./(1 .+exp.(.-(x.-𝜇)./𝜎)) #=
                                        =# (y, 𝜇, 𝜎) -> y .= .-𝜎.*log.(1.0./y .- 1) .+ 𝜇
@_Normalization MinMax (minimum, maximum)  (x, l, u) -> x .= (x.-l)./(u-l) #=
                                        =# (y, l, u) -> y .= (u-l).*y .+ l
@_Normalization Center (mean,)             (x, 𝜇) -> x .= x .- 𝜇     (y, 𝜇) -> y .= y .+ 𝜇
@_Normalization RobustCenter (median,)     Centre().𝑓   Centre().𝑓⁻¹

# * Robust versions of typical 2-parameter normalizations
common_norms = [:ZScore, :Sigmoid,]
_iqr = x -> (quantile(x[:], 0.75) - quantile(x[:], 0.25))/1.35 # ? Divide by 1.35 so that std(x) ≈ _iqr(x) when x contains normally distributed values
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
nansafe(𝒯::Type{<:AbstractNormalization}; dims=nothing) = dims |> 𝒯 |> nansafe

Base.reshape(x::Number, dims...) = reshape([x], dims...)
function fit!(T::AbstractNormalization, X::AbstractArray; dims=())
    T(dims)
    dims = isnothing(T.dims) ? (1:ndims(X)) : T.dims
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T.p = reshape.(map.(T.𝑝, (Slices(X, dims...),)), psz...)
end
fit(T::AbstractNormalization, X::AbstractArray; kw...)=(T=deepcopy(T); fit!(T, X; kw...); T)
fit(𝒯::Type{<:AbstractNormalization}, X::AbstractArray; dims=nothing) = (T = 𝒯(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
NormUnion = Union{AbstractNormalization, Type{<:AbstractNormalization}}
normalize!(X::AbstractArray, 𝒯::NormUnion; dims=nothing) = normalize!(X, fit(𝒯, X; dims))
normalize(X::AbstractArray, T::NormUnion; kwargs...) = (Y=copy(X); normalize!(Y, T; kwargs...); Y)

function denormalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && error("Cannot denormalize with an unfit normalization")
    mapdims!(T.𝑓⁻¹, X, T.p...; T.dims)
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
    @assert all(all(size.(x[2:end], i) .== 1) for i ∈ dims)
    @assert all(all(size(x[1], i) .== size.(x, i)) for i ∈ negdims)
    idxs = Base.compute_itspace(x[1], (negdims...,)|>Val)
    Threads.@threads for i ∈ idxs # map(f!, Slices.(x, dims...)...)
        selectslice = x -> view(x, i...)
        f(selectslice.(x)...)
    end
end

end
