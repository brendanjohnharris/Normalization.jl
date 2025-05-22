export AbstractModifier, Robust

# * Modifiers
abstract type AbstractModifier{N<:AbstractNormalization} <: AbstractNormalization{eltype(N)} end
normalization(N::AbstractModifier) = N.normalization
normalization(::Type{M}) where {N<:AbstractNormalization,M<:AbstractModifier{N}} = N
Base.eltype(M::AbstractModifier) = eltype(normalization(M))
Base.eltype(::Type{M}) where {M<:AbstractModifier} = eltype(normalization(M))

# (::Type{M})(n::N) where {T,N<:AbstractNormalization{T},M<:AbstractModifier} = Robust{N}(n)
# (::Type{M})(n::Type{N}; dims) where {T,N<:AbstractNormalization{T},M<:AbstractModifier} = M{N,T}(dims, ntuple(_ -> Vector{T}(), length(estimators(n))), n)
function (::Type{M})(dims, p::NTuple{n,AbstractArray{T}}; kwargs...) where {n,T,N,M<:AbstractModifier{N}}
    (all(x -> x == p[1], length.(p)) && error("Inconsistent parameter dimensions"))
    norm = N(dims, p) # * Should cascade for nested modifiers
    return M(norm)
end

dims(N::AbstractModifier) = N |> normalization |> dims
params(N::AbstractModifier) = N |> normalization |> params
forward(::Type{N}) where {N<:AbstractModifier} = N |> normalization |> forward
# forward!(::Type{<:AbstractModifier{N}}) where {N} = N |> forward!
# inverse!(::Type{<:AbstractModifier{N}}) where {N} = N |> inverse!
estimators(::Type{ℳ}) where {ℳ<:AbstractModifier} = error("Estimators undefined for $ℳ")
# function fit(::Type{𝒯}, X::AbstractArray{A}; kwargs...) where {A,𝒯<:AbstractModifier}
#     n = fit(normalization(𝒯), X; kwargs...)
#     N = 𝒯(n)
#     return N
# end


# * Robust normalizations
mutable struct Robust{N<:AbstractNormalization} <: AbstractModifier{N}
    normalization::N
end

function _iqr(x::AbstractArray{T})::T where {T}
    eltype(x).((quantile(x[:], 0.75) - quantile(x[:], 0.25)) / 1.35) # ? Divide by 1.35 so that std(x) ≈ _iqr(x) when x contains normally distributed values
end

robust(::typeof(mean)) = median
robust(::typeof(std)) = _iqr # * Can define other robust methods if needed
estimators(::Type{T}) where {N,T<:Robust{N}} = robust.(estimators(N))


mutable struct NaNSafe{T,N} <: AbstractNormalization{T}
    normalization::AbstractNormalization{N}
end
# ........add to interface



mutable struct Mixed{T,N} <: AbstractNormalization{T}
    normalization::AbstractNormalization{N}
end
# ..........add to interface..............


# * Require a certain order of modifiers
# ..........nansafe before mixed or robust.........



# * Mixed versions of typical 2-parameter normalizations
# mixedcenter(x) = (_iqr(x) == 0) ? mean(x) : median(x)
# mixedscale(x) = (𝜎 = _iqr(x); 𝜎 == 0 ? std(x) : 𝜎)
# _mixedNorm(N::Symbol; name="Mixed"*string(N)|>Symbol) = eval(:(@_Normalization $name (mixedcenter, mixedscale) forward!($N) inverse!($N)))
# _mixedNorm.(common_norms)

# * NaN-safe versions
# _nansafe(p) = x -> p(filter(!isnan, x))
# nansafe!(T::AbstractNormalization) = (T.𝑝=_nansafe.(T.𝑝); ())
# nansafe(T::AbstractNormalization) = (N = deepcopy(T); nansafe!(N); N)
# nansafe(𝒯::Type{<:AbstractNormalization}; dims=nothing) = 𝒯(; dims) |> nansafe

function nansafe(f::Function; dims=nothing)
    function g(x)
        isnothing(dims) && (dims = 1:ndims(x))
        mapslices(y -> f(filter(!isnan, y)), x; dims)
    end
end
