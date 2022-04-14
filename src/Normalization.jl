module Normalization

using Statistics
import LinearAlgebra:   normalize,
                        normalize!

export  fit,
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
        p::Union{Nothing, NTuple{2, AbstractArray}}
        𝑝::NTuple{2, Function}
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
@_Normalization ZScore (mean, std)         (x, 𝜇, 𝜎) -> (x .- 𝜇)./𝜎  #=
                                        =# (y, 𝜇, 𝜎) -> y.*𝜎 .+ 𝜇
@_Normalization Sigmoid (mean, std)        (x, 𝜇, 𝜎) -> 1.0./(1 .+ exp.(.-(x.-𝜇)./𝜎)) #=
                                        =# (y, 𝜇, 𝜎) -> .-𝜎.*log.(1.0./y .- 1) .+ 𝜇
@_Normalization MinMax (minimum, maximum)  (x, l, u) -> (x.-l)./(u-l) #=
                                        =# (y, l, u) -> (u-l).*y .+ l

_ZScore(name::Symbol, 𝑝) = eval(:(@_Normalization $name $𝑝 ZScore().𝑓 ZScore().𝑓⁻¹))
_Sigmoid(name::Symbol, 𝑝) = eval(:(@_Normalization $name $𝑝 Sigmoid().𝑓 Sigmoid().𝑓⁻¹))

# * Robust versions
_iqr = x -> (quantile(x[:], 0.75) - quantile(x[:], 0.25))/1.35 # ? Divide by 1.35 so that std(x) ≈ _iqr(x) when x contains normally distributed values
_robustNorm(name::Symbol, N::Symbol) = eval(:(@_Normalization $name (median, _iqr) ($N)().𝑓 ($N)().𝑓⁻¹))
_robustNorm.([:RobustZScore,  :RobustSigmoid,],
             [:ZScore,        :Sigmoid,])

# * NaN-safe versions
_nansafe(p) = x -> p(filter(!isnan, x))
nansafe!(T::AbstractNormalization) = (T.𝑝=_nansafe.(T.𝑝); ())
nansafe(T::AbstractNormalization) = (N = deepcopy(T); nansafe!(N); N)
nansafe(𝒯::Type{<:AbstractNormalization}; dims=nothing) = dims |> 𝒯 |> nansafe


function fit!(T::AbstractNormalization, X::AbstractArray; dims=())
    T(dims)
    dims = isnothing(T.dims) ? (1:ndims(X)) : T.dims
    T.p = mapslices.(T.𝑝, (X,); dims)
end
fit(T::AbstractNormalization, X::AbstractArray; kw...)=(T=deepcopy(T); fit!(T, X; kw...); T)
fit(𝒯::Type{<:AbstractNormalization}, X::AbstractArray; dims=nothing) = (T = 𝒯(dims); fit!(T, X); T)

function normalize!(X::AbstractArray, T::AbstractNormalization)
    isnothing(T.p) && fit!(T, X)
    mapdims!(T.𝑓, X, T.p...; T.dims)
end
normalize!(X::AbstractArray, 𝒯::Type{<:AbstractNormalization}; dims=nothing) = normalize!(X, fit(𝒯, X; dims))
normalize(X::AbstractArray, T::Union{AbstractNormalization, Type{<:AbstractNormalization}}; kwargs...) = (Y=copy(X); normalize!(Y, T; kwargs...); Y)

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
