module DataFramesExt
using DataFrames
using Normalization

import Normalization: fit!, fit, normalize!, denormalize!, negdims, NormUnion

function fit!(T::AbstractNormalization, Y::AbstractDataFrame; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    𝒳 = eltype(X)
    𝒯 = eltype(T)
    𝒳 == 𝒯 || throw(error("$T with eltype $𝒯 type does not match data type $𝒳"))

    fit!(T, X; kwargs...)
end

function fit(T::Union{<:AbstractNormalization,<:Type{<:AbstractNormalization}}, Y::AbstractDataFrame; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    fit(T, X; kwargs...)
end

function normalize!(Y::AbstractDataFrame, T::AbstractNormalization; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    normalize!(X, T; kwargs...)
    Y .= X
    return nothing
end

function denormalize!(Y::AbstractDataFrame, T::AbstractNormalization; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    denormalize!(X, T; kwargs...)
    Y .= X
    return nothing
end

(T::AbstractNormalization)(X::AbstractDataFrame) = normalize(X, T)

end # module
