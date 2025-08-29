module UnitfulExt
using Unitful
import Normalization: normalize, normalize!, denormalize, denormalize!, AbstractNormalization, params, params!, dims, NotFitError, mapdims!, forward, inverse, estimators, fit, dimparams, negdims

function fit(::Type{𝒯}, X::AbstractArray{A}; kwargs...) where {A<:Quantity,𝒯<:AbstractNormalization}
    pu = map(estimators(𝒯)) do f
        f(X[1:5]) # A little slice to get the right eltype
    end
    fit(𝒯{eltype(pu)}, X; kwargs...)
end
function fit(::Type{𝒯}, X::AbstractArray{A}; dims=nothing) where {A<:Quantity,T,𝒯<:AbstractNormalization{T}}
    dims, nps = dimparams(dims, X)
    Xs = eachslice(X; dims=negdims(dims, ndims(X)), drop=false)
    ps = map(estimators(𝒯)) do f
        reshape(map(f, Xs), nps...)
    end
    𝒯(dims, ps)
end
function normalize(X::AbstractArray{<:Quantity}, T::AbstractNormalization; kwargs...)
    # * Infer the units by applying one element
    pu = map(estimators(T)) do f
        f(X[1:5]) # A little slice to get the right eltype
    end
    u = forward(T)(pu...)(first(X)) |> eltype
    Y = similar(X, u)
    normalize!(Y, X, T; kwargs...)
    return Y
end

function denormalize(X::AbstractArray, T::AbstractNormalization{<:Quantity}; kwargs...)
    pu = inverse(T)(params(T)...)(first(X)) # Get the right eltype
    Y = similar(X, eltype(pu))
    denormalize!(Y, X, T; kwargs...)
    return Y
end

end # module
