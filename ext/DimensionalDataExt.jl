module DimensionalDataExt
using DimensionalData
using Normalization

function Normalization._mapdims!(f, idxs, x::AbstractDimArray, y...)
    Threads.@threads for i ∈ idxs
        selectslice = x -> view(x, i...)
        f(selectslice.((x.data, y...))...)
    end
end

end # module
