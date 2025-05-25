module DimensionalDataExt
using DimensionalData
using Normalization

function Normalization._mapdims!(f, xs::Slices{<:AbstractDimArray}, ys::NTuple{N,<:AbstractArray}) where {N}
    @sync Threads.@threads for i in eachindex(xs) #
        y = ntuple((j -> @inbounds ys[j][i]), Val(N)) # Extract parameters for nth slice
        @inbounds map!(f(map(only, y)...), parent(xs[i]), parent(xs[i]))
    end
end

end # module
