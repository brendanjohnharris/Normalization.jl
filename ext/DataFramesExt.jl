module DataFramesExt
using DataFrames
using Normalization
using Normalization.Accessors
using Normalization.JuliennedArrays

import Normalization: fit!, fit, normalize!, denormalize!

# Here we just treat a table/dataframe as a matrix. We have the option to restrict the normalization to a subset of columns?

function fit!(T::AbstractNormalization, Y::AbstractDataFrame; dims=nothing)
    X = DataFrames.Tables.matrix(Y)
    𝒳 = eltype(X)
    𝒯 = eltype(T)
    @assert 𝒳 == 𝒯 "$𝒯 type does not match data type ($𝒳)"
    dims = isnothing(dims) ? (1:ndims(X)) : dims
    length(dims) > 1 && sort!(dims)
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T.dims = dims
    T.p = reshape.(map.(T.𝑝, (JuliennedArrays.Slices(X, dims...),)), psz...)
    nothing
end
function fit(T::AbstractNormalization{Nothing}, Y::AbstractDataFrame; dims=nothing)
    X = DataFrames.Tables.matrix(Y)
    dims = isnothing(dims) ? (1:ndims(X)) : dims
    length(dims) > 1 && sort!(dims)
    psz = size(X) |> collect
    psz[[dims...]] .= 1
    T = @set T.dims = dims
    T = @set T.p = reshape.(map.(T.𝑝, (JuliennedArrays.Slices(X, dims...),)), psz...)
end

function normalize!(Y::AbstractDataFrame, T::AbstractNormalization)
    X = DataFrames.Tables.matrix(Y)
    normalize!(X, T)
    Y .= X
end

function denormalize!(Y::AbstractDataFrame, T::AbstractNormalization)
    X = DataFrames.Tables.matrix(Y)
    denormalize!(X, T)
    Y .= X
end

end # module
