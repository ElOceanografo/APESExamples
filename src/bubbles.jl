using UnderwaterAcoustics
using SDWBA

struct Bubble{T}
    R::T
    f_res::T
    δ::T
end

function Bubble(R; depth=1.0, δ=0.0)
    f_res = bubbleresonance(R, depth)
    return Bubble(promote(R, f_res, δ)...)
end

function SDWBA.backscatter_xsection(b::Bubble, freq)
    return 4π * b.R^2 / ((b.f_res^2 / freq^2 - 1)^2 + b.δ^2)
end
SDWBA.target_strength(b::Bubble, freq) = 10log10(backscatter_xsection(b, freq))


struct ProlateSpheroid{T}
    a::T
    b::T
    esr::T
    ρf::T
    ξ::T
    f_res::T
end

"""
    `ProlateSpheroid(a, b, P₀=101.3, g=9.81, ρl=1026.0, ρf=ρl*1.04))`


* `a`, `b` : major and minor axes of spheroid, in m
* `P₀` : atmospheric pressure (default: 101.3 Pa)
* `g` : gravitational acceleration (default on Earth: 9.81 m s⁻²)
* `ρl` : density of seawater (default 1026.0 kg m⁻³)
* `ρf` : flesh density (default: ρl * 1.04)
* `ξ` : flesh viscosity (default: 5 kg m⁻¹ s⁻¹)
"""
function ProlateSpheroid(a, b; depth=1.0, P₀=101.3, g=9.81, ρl=1026.0, ρf=ρl*1.05, ξ=5.0)
    a < b || throw(ArgumentError("`a` is the minor axis, must be < b"))
    esr = (b * a^2)^(1/3)
    f_res = f_res_prolate_spheroid(a, b; depth=depth, P₀=P₀, g=g, ρl=ρl, ρf=ρf)
    return ProlateSpheroid(promote(a, b, esr, ρf, ξ, f_res)...)
end

function ζ(a, b)
    ϵ = a / b
    return sqrt(2*(1-ϵ^2)^0.25) / ϵ^(1/3) * log((1+sqrt(1-ϵ^2)) / (1-sqrt(1-ϵ^2)))
end

"""
    `f_res_prolate_spheroid(a, b; P₀=101.3, g=9.81, ρl=1026.0, ρf=ρl*1.04)`

Calculate resonant frequncy of a gas-filled prolate spheroid embedded in flesh, based on:
* `a`, `b` : minor and major axes of spheroid, in m
* `P₀` : atmospheric pressure (default: 101.3 Pa)
* `g` : gravitational acceleration (default on Earth: 9.81 m s^-2)
* `ρl` : density of seawater (default 1026.0 kg m^-3)
* `ρf` : flesh density (default: ρl * 1.04)
"""
function f_res_prolate_spheroid(a, b; depth=1.0, P₀=101.3, g=9.81, ρl=1026.0, ρf=ρl*1.04)
    P = P₀ + ρl * g * depth
    γ = 1.4
    esr = (b * a^2)^(1/3)
    return sqrt(ζ(a, b)^2 *  (3 * γ * P) / (4π^2 * esr^2 * ρf))
end

"""
    `damping(f, f_res, c, ξ, ρ_f)`

Calculate damping factor for a gas-filled spher(oid) embedded in flesh based on:
* `f` : frequency of incident sound waves
* `f_res` : resonant frequency of sphere(oid)
* `c` : speed of sound in surrounding water
* `ξ` : viscosity of flesh surrounding gas inclusion
* `ρ_f` : density of flesh
"""
function damping(f, f_res, esr, c, ξ, ρ_f)
    return 1 / ((2π*esr*f^2) / (f_res * c) + ξ / (π * esr^2 * f_res * ρ_f))
end

function SDWBA.backscatter_xsection(ps::ProlateSpheroid, f, c)
    H = damping(f, ps.f_res, ps.esr, c, ps.ξ, ps.ρf)
    t1 = ps.esr^2
    t2 = ps.f_res^2 / (f^2 * H^2)
    t3 = (ps.f_res / f - 1)^2
    return t1 / (t2 + t3)
end
SDWBA.target_strength(ps::ProlateSpheroid, f, c) = 10log10(backscatter_xsection(ps, f, c))
#
# a = 0.003
# b = 0.01
# ps = ProlateSpheroid(a, b, depth=100)
# backscatter_xsection(ps, 120e3, 1500)
# target_strength(ps, 120e3, 1500)
#
# using Plots
# p = plot()
# for depth in [10, 50, 100, 250, 500]
#     ps = ProlateSpheroid(a, b, depth=depth)
#     plot!(p, f -> target_strength(ps, f*1e3, 1500), 1, 200, label="$depth m")
# end
# vline!(p, [18, 38, 70, 120, 200], color=:grey, label="")


# b2 = Bubble(5e-3, δ=0.1, depth=500)
# plot(f -> target_strength(b1, 1e3*f), 0.5, 250, xscale=:log10, label="1 mm")
# plot!(f -> target_strength(b2, 1e3*f), 0.5, 250, label="5 mm")
# vline!([18, 38, 70, 120, 200], label="")
#
# freqs = [18, 38, 70, 120, 200]
# plot(freqs, [target_strength(b1, 1e3*f) for f in freqs], marker=:o, label="1 mm")
# plot!(freqs, [target_strength(b2, 1e3*f) for f in freqs], marker=:o, label="5 mm")
