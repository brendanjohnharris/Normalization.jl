module DimensionalDataExt
using DimensionalData
using Normalization
import Normalization.__mapdims!

function Normalization._mapdims!(zs::Slices{<:AbstractDimArray}, f, xs::Slices{<:AbstractDimArray}, ys::NTuple{N,<:AbstractArray}) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        __mapdims!(parent(zs[i]), f, parent(xs[i]), y)
    end
end

end # module
