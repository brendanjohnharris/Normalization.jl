Base.@kwdef mutable struct ZScore <: AbstractNormalization
    dims
    p::Union{Nothing, NTuple{2, AbstractArray}} = nothing
    𝑝::NTuple{2, Function} = (mean, std)
    𝑓::Function = (x, 𝜇, 𝜎)->(x .- 𝜇)./𝜎
    𝑓⁻¹::Function = (y, 𝜇, 𝜎) -> y.*𝜎 .+ 𝜇
end
