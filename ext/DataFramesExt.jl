module DataFramesExt
using DataFrames
using Normalization

import Normalization: fit!, fit, normalize!, denormalize!, negdims

function fit!(T::AbstractNormalization, Y::AbstractDataFrame; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    𝒳 = eltype(X)
    𝒯 = eltype(T)
    𝒳 == 𝒯 || throw(error("$T with eltype $𝒯 type does not match data type $𝒳"))

    fit!(T, X; kwargs...)
    # dims = isnothing(dims) ? (1:ndims(X)) : dims
    # dims = length(dims) > 1 ? sort!(dims) : dims
    # _dims = negdims(dims, ndims(X)) |> Tuple
    # psz = size(X) |> collect
    # psz[[dims...]] .= 1
    # T.dims = dims
    # T.p = reshape.(map.(T.𝑝, (eachslice(X; dims=_dims, drop=false),)), psz...)
    # nothing
end

function fit(T::Union{<:AbstractNormalization,<:Type{<:AbstractNormalization}}, Y::AbstractDataFrame; kwargs...)
    X = DataFrames.Tables.matrix(Y)
    fit(T, X; kwargs...)
    # dims = isnothing(dims) ? (1:ndims(X)) : dims
    # dims = length(dims) > 1 ? sort!(dims) : dims
    # _dims = negdims(dims, ndims(X)) |> Tuple
    # psz = size(X) |> collect
    # psz[[dims...]] .= 1
    # T = @set T.dims = dims
    # T = @set T.p = reshape.(map.(T.𝑝, (eachslice(X; dims=_dims, drop=false),)), psz...)
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
