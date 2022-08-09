# Normalization.jl

[![Build Status](https://github.com/brendanjohnharris/Normalization.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/Normalization.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/Normalization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/Normalization.jl)

This package allows you to easily normalize an array over any dimensions.
It also provides a bunch of normalization methods, such as the z-score, sigmoid, robust, mixed, and NaN-safe normalizations.

## Usage

Each normalization method is a subtype of `AbstractNormalization`. Instances of a normalization, including any parameters (such as the mean of a dataset) are stored in a variable of the `AbstractNormalization` type.
For example, to normalize a 2D array using the `ZScore` normalization method (or any other `<: AbstractNormalization`) over all dimensions:
```
X = rand(100, 100)
N = ZScore(X) # A normalization fit to X, NOT the normalized array
N = ZScore()(X) # An alternative to the line above
Y = N(X) # The normalized array
Z = N(rand(100, 100)) # Apply a normalization with parameters fit to X on a new array
```

There is also an alternative, preferred, syntax:
```
using Statistics
N = fit(ZScore, X)
Y = normalize(X, N)
normalize!(X, N) # In place, writing over X
```

A normalization can also be reversed:
```
_X = denormalize(X, N) # Apply the inverse normalization
denormalize!(X, N) # Or do the inverse in place
```

Both syntaxes allow you to specify the dimensions to normalize over. For example, to normalize each 2D slice (i.e. iterating over the 3rd dimension) of a 3D array:
```
X = rand(100, 100, 10)
N = fit(ZScore, X; dims=[1, 2])
normalize!(X, N) # Each [1, 2] slice is normalized independently
all(std(X; dims=[1, 2]) .≈ 1) # true
```

## Normalization methods

Any of these normalizations will work in place of `ZScore` in the examples above:
| Normalization | Formula | Description |
|--|--| -- |
| `ZScore` | $(x - \mu)/\sigma$ | Subtract the mean and scale by the standard deviation (aka standardisation) |
| `Sigmoid` | $(1 + \exp(-\frac{x-\mu}{\sigma}))^{-1}$ | Map to the interval $[-1, 1]$ by applying a sigmoid transformation |
| `MinMax` | $(x-\inf{x})/(\sup{x}-\inf{x})$ | Scale to the unit interval |


### Robust normalizations
This package also defines robust versions of any normalization methods that have $\mu$ (the mean) and $\sigma$ (the standard deviation) parameters. 
`Robust` including `RobustZScore` and `RobustSigmoid`, use the `median` and `iqr/1.35` rather than the `mean` and `std` for a normalization that is less sensitive to outliers.
There are also `Mixed` methods, such as `MixedZScore` and `MixedSigmoid`, that default to the `Robust` versions but use the regular parameters (`mean` and `std`) if the `iqr` is 0.

### NaN-safe normalizations

If the input array contains any `NaN` values, the normalizations given above will fit with `NaN` parameters and return `NaN` arrays. To circumvent this, any normalization can be made '`NaN`-safe', meaning it ignores `NaN` values in the input array. Using the `ZScore` example:
```
N = nansafe(ZScore)
fit!(N, X)
Y = N(X)
```

### New normalizations

Finally, there is also a macro to define your own normalization (honestly you could just make the `struct` directly). For example, the `ZScore` is defined as:
```
@_Normalization ZScore (mean, std)  (x, 𝜇, 𝜎) -> x .= (x .- 𝜇)./𝜎  #=
                                 =# (y, 𝜇, 𝜎) -> y .= y.*𝜎 .+ 𝜇
```
Here, the first argument is a name for the normalization, the second is a tuple of parameter functions, the third is vectorised function of an array `x` and any parameters, and the fourth is a function for the inverse transformation.
