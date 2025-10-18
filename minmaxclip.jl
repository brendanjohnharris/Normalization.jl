using Normalization
using Test

begin # * Normal data
    x = [1.0, 2.0, 3.0, 4.0]
    y1 = normalize(x, MinMax)

    N = fit(MinMaxClip, x)
    y2 = normalize(x, N)
    @test y1 == y2
    @test extrema(y2) == (0.0, 1.0)


    x = [-1.0, 5.0]
    y3 = normalize(x, N)
    @test y3 == [0.0, 1.0] # Values outside fo original range are clipped
end

begin # * Pathological data
    x = fill(7.0, 3)
    y1 = normalize(x, MinMax)

    N = fit(MinMaxClip, x)
    y2 = normalize(x, N)
    @test all(isnan, y1)
    @test all(==(0.5), y2)

    x = [-1.0, 5.0]
    y3 = normalize(x, N)
    @test y3 == [0.5, 0.5] # Any values will get mapped to 0.5
end
