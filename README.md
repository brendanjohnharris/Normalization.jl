# Normalization.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10613385.svg)](https://zenodo.org/doi/10.5281/zenodo.10613385)
[![Build Status](https://github.com/brendanjohnharris/Normalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/Normalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/Normalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/Normalization.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FNormalization&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/Normalization)

This package allows you to easily normalize an array over any combination of dimensions, with a bunch of methods (z-score, sigmoid, centering, minmax, etc.) and modifiers (robust, mixed, NaN-safe).

## Usage

Each normalization method is a subtype of `AbstractNormalization`.
Each `AbstractNormalization` subtype has its own `estimators` and `forward` methods that define how parameters are calculated and the normalization formula.
Each `AbstractNormalization` instance contains the concrete parameter values for a normalization, fit to a given input array.

You can work with `AbstractNormalization`s as either types or instances.
The type approach is useful for concise code, whereas the instance approach is useful for performant mutations.
In the examples below we use the `ZScore` normalization, but the same syntax applies to all `Normalization`s.

### Fit to a type
```julia
    X = randn(100, 10)
    N = fit(ZScore, X; dims=nothing) # eltype inferred from X
    N = fit(ZScore{Float32}, X; dims=nothing) # eltype set to Float32
    N isa AbstractNormalization && N isa ZScore # Returns a concrete AbstractNormalization
```

### Fit to an instance
```julia
    X = randn(100, 10)
    N = ZScore{Float64}(; dims=2) # Initializes with empty parameters
    N isa AbstractNormalization && N isa ZScore # Returns a concrete AbstractNormalization
    !isfit(N)

    fit!(N, X; dims=1) # Fit normalization in-place, and update the `dims`
    Normalization.dims(N) == 1
```

## Normalization and denormalization
With a fit normalization, there are two approaches to normalizing data: in-place and
out-of-place.
```julia
    _X = copy(X)
    normalize!(_X, N) # Normalizes in-place, updating _X
    Y = normalize(X, N) # Normalizes out-of-place, returning a new array
    normalize(X, ZScore; dims=1) # For convenience, fits and then normalizes
```
For most normalizations, there is a corresponding denormalization that
transforms data to the original space.
```julia
    Z = denormalize(Y, N) # Denormalizes out-of-place, returning a new array
    Z ≈ X
    denormalize!(Y, N) # Denormalizes in-place, updating Y
```

Both syntaxes allow you to specify the dimensions to normalize over. For example, to normalize each 2D slice (i.e. iterating over the 3rd dimension) of a 3D array:
```julia
X = rand(100, 100, 10)
N = fit(ZScore, X; dims=[1, 2])
normalize!(X, N) # Each [1, 2] slice is normalized independently
all(std(X; dims=[1, 2]) .≈ 1) # true
```

## Normalization methods

Any of these normalizations will work in place of `ZScore` in the examples above:
| Normalization | Formula | Description |
|--|--| -- |
| `ZScore` | $(x - \mu)/\sigma$ | Subtract the mean and scale by the standard deviation (aka standardization) |
| `Sigmoid` | $(1 + \exp(-\frac{x-\mu}{\sigma}))^{-1}$ | Map to the interval $(0, 1)$ by applying a sigmoid transformation |
| `MinMax` | $(x-\inf{x})/(\sup{x}-\inf{x})$ | Scale to the unit interval |
| `Center` | $x - \mu$ | Subtract the mean |
| `UnitEnergy` | $x/\sum x^2$ | Scale to have unit energy |
| `HalfZScore` | $\sqrt{1-2/\pi} \cdot (x - \inf{x})/\sigma$ | Normalization to the standard half-normal distribution |
| `OutlierSuppress` | $\max(\min(x, \mu + 5\sigma), \mu - 5\sigma)$ | Clip values outside of $\mu \pm 5\sigma$ |


## Normalization modifiers
What if the input data contains NaNs or outliers?
We provide `AbstractModifier` types that can wrap an `AbstractNormalization` to modify its behavior.

Any concrete modifier type `Modifier <: AbstractModifier` (for example, `NaNSafe`) can be applied to a concrete normalization type `Normalization <:AbstractNormalization`:
```julia
    N = NaNSafe{ZScore} # A combined type with a free `eltype` of `Any`
    N = NaNSafe{ZScore{Float64}} # A concrete `eltype` of `Float64`
```
Any `AbstractNormalization` can be used in the same way as an `AbstractModifier`.

### NaN-safe normalizations
If the input array contains any `NaN` values, the ordinary normalizations given above will fit with `NaN` parameters and return `NaN` arrays.
To circumvent this, any normalization can be made '`NaN`-safe', meaning it ignores `NaN` values in the input array, using the `NaNSafe` modifier.

### Robust modifier
The `Robust` modifier can be used with any `AbstractNormalization` that has mean and standard deviation parameters.
The `Robust` modifier converts the `mean` to `median` and `std` to `iqr/1.35`, giving a normalization that is less sensitive to outliers.

### Mixed modifier
The `Mixed` modifier defaults to the behavior of `Robust` but uses the regular parameters (`mean` and `std`) if the `iqr` is 0.

## Properties and traits
The following are common methods defined for all `AbstractNormalization` subtypes and instances.

### Type traits
- `Normalization.estimators(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the estimators `N` as a tuple of functions
- `forward(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the forward normalization function (e.g. $x$ -> $x - \mu / \sigma$ for the `ZScore`)
- inverse(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization}})` returns the inverse normalization function e.g. `forward(N)(ps...) |> InverseFunctions.inverse`
- `eltype(N::Union{<:AbstractNormalization,Type{<:AbstractNormalization})` returns the eltype of the normalization parameters

### Concrete properties
- `Normalization.dims(N::<:AbstractNormalization)` returns the dimensions of the normalization. The dimensions are determined by `dims` and correspond to the mapped slices of the input array.
- `params(N::<:AbstractNormalization)` returns the parameters of `N` as a tuple of arrays. The dimensions of arrays are the complement of `dims`.
- `isfit(N::<:AbstractNormalization)` checks if all parameters are non-empty

<!-- ### New normalizations

Finally, there is also a macro to define your own normalization (honestly you could just make the `struct` directly). For example, the `ZScore` is defined as:
```julia
@_Normalization ZScore (mean, std)  (x, 𝜇, 𝜎) -> x .= (x .- 𝜇)./𝜎  #=
                                 =# (y, 𝜇, 𝜎) -> y .= y.*𝜎 .+ 𝜇
```
Here, the first argument is a name for the normalization, the second is a tuple of parameter functions, the third is a vectorised, in-place function of an array `x` and any parameters, and the fourth is a function for the inverse transformation. -->
