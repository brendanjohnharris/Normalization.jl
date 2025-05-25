module UnitfulExt
using Unitful
import Normalization: normalize, normalize!, denormalize, denormalize!, AbstractNormalization, params, params!, dims, NotFitError, mapdims!, forward, inverse, estimators

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
    Y = similar(X, eltype(T))
    denormalize!(Y, X, T; kwargs...)
    return Y
end

end # module
