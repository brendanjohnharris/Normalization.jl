module UnitfulExt
using Unitful
import Normalization: normalize, normalize!, denormalize, denormalize!, AbstractNormalization, params, params!, dims, NotFitError, mapdims!, forward, inverse, estimators

# Extend Normalizations.jl to unitful data
# Note that in-place normalization is not defined for unitful arrays unless the normalization doesn't change the units.

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

# function inverse(::Type{N}) where {N<:AbstractNormalization{<:Quantity}}
#     u = unit(eltype(N))
#     function _inverse(ps...,)
#         ps = map(ustrip, ps)
#         inverse(forward(N)(ps...))
#     end
# end

function denormalize(X::AbstractArray, T::AbstractNormalization{<:Quantity}; kwargs...)
    Y = similar(X, eltype(T))
    denormalize!(Y, X, T; kwargs...)
    return Y
end

end # module
