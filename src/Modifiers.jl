export AbstractModifier, Robust, NaNSafe, Mixed

# * Modifiers
"""
    AbstractModifier{N<:AbstractNormalization} <: AbstractNormalization{eltype(N)}

Abstract type for modifiers that wrap an existing normalization, altering its behavior (for
example, by making it robust to outliers or NaN-safe).

## Interface
Any concrete modifier type `Modifier <: AbstractModifier` (such as `Robust`, `Mixed`, or
`NaNSafe`) can be applied to a concrete normalization type `Normalization <:
AbstractNormalization`:
```julia
    N = Modifier{Normalization} # A combined type with a free `eltype` of `Any`
    N = Modifier{Normalization{Float64}} # A concrete `eltype` of `Float64`
```
All `AbstractNormalization` constructors and traits are then defined for `AbstractModifier` types.
"""
abstract type AbstractModifier{N<:AbstractNormalization} <: AbstractNormalization{eltype(N)} end
normalization(N::AbstractModifier) = N.normalization
normalization(::Type{M}) where {N<:AbstractNormalization,M<:AbstractModifier{N}} = N
Base.eltype(M::AbstractModifier) = eltype(normalization(M))
Base.eltype(::Type{M}) where {M<:AbstractModifier} = eltype(normalization(M))
dims(N::AbstractModifier) = N |> normalization |> dims
params(N::AbstractModifier) = N |> normalization |> params
forward(::Type{N}) where {N<:AbstractModifier} = N |> normalization |> forward
estimators(::Type{ℳ}) where {ℳ<:AbstractModifier} = error("Estimators undefined for $ℳ")

function (::Type{M})(dims, p::Tuple{Vararg{<:AbstractArray{T}}}) where {T,M<:AbstractModifier}
    (length(unique(length.(p))) == 1) || throw(error("Inconsistent parameter dimensions"))
    norm = normalization(M)(dims, p) # * Should cascade for nested modifiers
    return M(norm)
end
function (::Type{M})(; dims=nothing) where {M<:AbstractModifier}
    T = eltype(M)
    p = ntuple(_ -> Vector{T}(), length(estimators(M)))
    norm = normalization(M)(dims, p) # * Should cascade for nested modifiers
    return M(norm)
end

"""
    Robust{N<:AbstractNormalization} <: AbstractModifier{N}

A modifier type that wraps an existing normalization and replaces its estimators with robust statistics.

`Robust` replaces the mean with the median and the standard deviation with the interquartile range (IQR, divided by 1.35 so that it matches the standard deviation for normally distributed data). This makes the normalization more robust to outliers.

"""
mutable struct Robust{N<:AbstractNormalization} <: AbstractModifier{N}
    normalization::N
    Robust{N}(norm::N) where {N<:AbstractNormalization} = new{N}(norm)
end

function _iqr(x::AbstractArray{T})::T where {T}
    eltype(x).((quantile(x[:], 0.75) - quantile(x[:], 0.25)) / 1.35) # ? Divide by 1.35 so that std(x) ≈ _iqr(x) when x contains normally distributed values
end

robust(::typeof(mean)) = median
robust(::typeof(std)) = _iqr # * Can define other robust methods if needed
estimators(::Type{T}) where {N,T<:Robust{N}} = robust.(estimators(N))

"""
    Mixed{N<:AbstractNormalization} <: AbstractModifier{N}

A modifier type that wraps an existing normalization and replaces its estimators with a mixture of robust and classical statistics.

`Mixed` uses the median instead of the mean if the interquartile range (IQR) is nonzero, otherwise it falls back to the mean. For the standard deviation, it uses the IQR (divided by 1.35) if nonzero, otherwise it falls back to the standard deviation. This makes the normalization robust to outliers while still using classical estimators when the data is degenerate.
"""
mutable struct Mixed{N<:AbstractNormalization} <: AbstractModifier{N}
    normalization::N
    Mixed{N}(norm::N) where {N<:AbstractNormalization} = new{N}(norm)
end
mixed(::typeof(mean)) = x -> (_iqr(x) == 0) ? mean(x) : median(x)
function mixed(::typeof(std))
    function _mixed(x)
        𝜎 = _iqr(x)
        𝜎 == 0 ? std(x) : 𝜎
    end
end
estimators(::Type{T}) where {N,T<:Mixed{N}} = mixed.(estimators(N))

"""
    NaNSafe{N<:AbstractNormalization} <: AbstractModifier{N}

A modifier type that wraps an existing normalization and replaces its estimators with NaN-safe versions.

`NaNSafe` modifies the estimators of the underlying normalization so that they ignore any `NaN` values in the input data.
"""
mutable struct NaNSafe{N<:AbstractNormalization} <: AbstractModifier{N}
    normalization::N
    NaNSafe{N}(norm::N) where {N<:AbstractNormalization} = new{N}(norm)
end

function nansafe(f::Function)
    function g(x; dims=nothing)
        if isnothing(dims)
            f(filter(!isnan, x))
        else
            mapslices(y -> f(filter(!isnan, y)), x; dims)
        end
    end
end

estimators(::Type{T}) where {N,T<:NaNSafe{N}} = nansafe.(estimators(N))
