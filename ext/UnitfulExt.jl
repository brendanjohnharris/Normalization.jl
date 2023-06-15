module UnitfulExt
using Unitful
using Accessors
import Normalization: normalize, normalize!, denormalize, denormalize!, NormUnion, AbstractNormalization

# Extend Normalizations.jl to unitful data
# Note that in-place normalization is not defined for unitful arrays unless the normalization doesn't change the units.

normalize(X::AbstractArray{<:Quantity}, T::NormUnion; kwargs...) = (Y=deepcopy(X)|>AbstractArray{Any}; normalize!(Y, T; kwargs...); identity.(Y))

function denormalize(Y::AbstractArray, T::AbstractNormalization{<:Quantity}; kwargs...)
    N = @set T.p = ustrip.(T.p)
    X = denormalize(Y, N; kwargs...)
    X*(unit∘eltype)(T)
end

function denormalize(Y::AbstractArray{<:Quantity}, T::AbstractNormalization{<:Quantity}; kwargs...)
    @error "Denormalization of unitful arrays currently not supported"
end


end # module
