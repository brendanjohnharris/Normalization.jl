using Statistics
import Accessors: @o

export ZScore,
    HalfZScore,
    Sigmoid,
    MinMax,
    Center,
    UnitEnergy,
    OutlierSuppress

halfstd(x, args...; kwargs...) = std(x, args...; kwargs...) ./ convert(eltype(x), sqrt(1 - (2 / π)))

# * invertible normalizations
zscore(𝜇, 𝜎) = @o (_ - 𝜇) / 𝜎 # * But this needs to be mapped over SCALAR 𝜇
sigmoid(𝜇, 𝜎) = @o inv(1 + exp(-(_ - 𝜇) / 𝜎))
minmax(l, u) = @o (_ - l) / (u - l)
center(𝜇) = @o _ - 𝜇
unitenergy(r𝐸) = Base.Fix2(/, r𝐸) # For unitful consistency, the sorted parameter is the root energy

# * noninvertible normalizations
function outliersuppress(𝜇, 𝜎)
    thr = 5.0
    function _outliersuppress(x)
        o = x - 𝜇
        if abs(o) > thr * 𝜎
            return 𝜇 + sign(o) * thr * 𝜎
        else
            return x
        end
    end
end

# * Construct Normalizations
@_Normalization ZScore (mean, std) zscore
@_Normalization HalfZScore (minimum, halfstd) zscore
@_Normalization Sigmoid (mean, std) sigmoid
@_Normalization MinMax (minimum, maximum) minmax
@_Normalization Center (mean,) center

energy(x) = map(square, x) |> sum |> sqrt
@_Normalization UnitEnergy (energy,) unitenergy

@_Normalization OutlierSuppress (mean, std) outliersuppress
