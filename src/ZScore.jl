mutable struct ZScore <: AbstractNormalization
    dims
    p::Union{Nothing, NTuple{2, AbstractArray}}
    𝑝::NTuple{2, Function}
    𝑓::Function
    𝑓⁻¹::Function

    function ZScore(dims, p=nothing)
        isnothing(p) || (all(x->x==p[1], length.(p)) && error("Inconsistent parameter dimensions"))
        𝑝 = (mean, std)
        𝑓 = (x, 𝜇, 𝜎)->(x .- 𝜇)./𝜎
        𝑓⁻¹ = (y, 𝜇, 𝜎) -> y.*𝜎 .+ 𝜇
        new(dims, p, 𝑝, 𝑓, 𝑓⁻¹)
    end
end
