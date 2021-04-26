module ReducedSysLyapVals
export ReducedSystem, getLyapunovData

import Pkg

Pkg.add("PyCall")
using PyCall

Pkg.add("DynamicalSystems")
using DynamicalSystems

pushfirst!(PyVector(pyimport("sys")."path"), "")
so =  pyimport("SystOsscills")



@inline @inbounds function ReducedSystem(u, p, t)
    w = p[1]; α = p[2]; β = p[3]; ρ = p[4]
    du  = so.FourBiharmonicPhaseOscillators(w,α,β,ρ)
    temp = du.getReducedSystem(u)
    return SVector{3}(temp[1], temp[2], temp[3])
end

# Jacobian:
@inline @inbounds function ReducedSystemJac(u, p, t)
    w,α,β,ρ = p
    du = so.FourBiharmonicPhaseOscillators(w,α,β,ρ)
    return @SMatrix [du.getReducedSystemJac(u)]
end

function getLyapunovData(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ = params
    a4d = ContinuousDynamicalSystem(ReducedSystem, rand(3), [0.5,a,b,r], ReducedSystemJac)
    λ = lyapunov(a4d, 10000.0, u0 = [startPtX, startPtY, startPtZ], dt = 0.1, Ttr = 10.0)
    return[i, j, a, b, r, λ]
end
end
