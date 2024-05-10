module DimensionalDataExt
using DimensionalData
using Normalization

function Normalization._mapdims!(f, xs::Slices{<:AbstractDimArray}, ys)
    Threads.@threads for i in eachindex(xs)
        f(getindex(xs, i) |> parent, getindex.(ys, i)...)
    end
end


end # module
