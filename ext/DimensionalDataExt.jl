module DimensionalDataExt
using DimensionalData
using Normalization

function Normalization._mapdims!(f, xs::Slices{<:AbstractDimArray}, ys::NTuple{N,<:AbstractArray}) where {N}
    Threads.@threads for i in eachindex(xs)
        y = getindex.(ys, i)
        map!(f(only.(y)...), parent(xs[i]), parent(xs[i]))
    end
end

end # module
