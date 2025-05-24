module UnitfulExt
using Unitful
import Normalization: normalize, normalize!, denormalize, denormalize!, AbstractNormalization

# Extend Normalizations.jl to unitful data
# Note that in-place normalization is not defined for unitful arrays unless the normalization doesn't change the units.

function normalize(X::AbstractArray{<:Quantity}, T::AbstractNormalization; kwargs...)
    Y = deepcopy(X) |> AbstractArray{Any}
    normalize!(Y, T; kwargs...)
    identity.(Y)
end

function denormalize(Y::AbstractArray, T::AbstractNormalization{<:Quantity}; kwargs...)
    N = deepcopy(N)
    map(params(N)) do p
        p .= ustrip(p)
    end
    X = denormalize(Y, N; kwargs...)
    X * (unit ∘ eltype)(T)
end

function denormalize(Y::AbstractArray{<:Quantity}, T::AbstractNormalization{<:Quantity}; kwargs...)
    @error "Denormalization of unitful arrays currently not supported"
end


end # module
