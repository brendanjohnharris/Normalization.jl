module DimensionalDataExt
using DimensionalData
using Normalization
import Normalization: __mapdims!, rootenergy
import DimensionalData: Dimension
import DimensionalData.Dimensions.Lookups: isregular

function Normalization._mapdims!(zs::Slices{<:AbstractDimArray}, f, xs::Slices{<:AbstractDimArray}, ys::NTuple{N,<:AbstractArray}) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        __mapdims!(parent(zs[i]), f, parent(xs[i]), y)
    end
end

# * Unit power for regular dim arrays
const RegularIndex = Dimensions.LookupArrays.Sampled{T,R} where {T,R<:AbstractRange}
const RegularDims = Tuple{Vararg{<:Dimension{T}}} where {T<:RegularIndex}
const RegularArray = AbstractDimArray{T,N,S} where {T,N,S<:RegularDims}

function rootenergy(x::RegularArray)
    dx = map(step, lookup(x)) |> prod # Total step element volume
    sqrt(sum(map(Normalization.square, x)) * dx)
end
end # module
