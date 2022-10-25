using ProbabilisticEchoInversion
using Turing
using Random
using SDWBA
using Dierckx
using LinearAlgebra
using Plots, StatsPlots, Plots.PlotMeasures
using DataFrames, DataFramesMeta, CategoricalArrays


include(joinpath(@__DIR__, "../src/bubbles.jl"))

Random.seed!(12345)

freqs_bb = [15:25; 35:38; 50:87; 100:150]
freqs_nb = [18, 38, 70, 120]
freqs_plot = 10:160
i_nb = indexin(freqs_nb, freqs_bb)

depth = 300.0
b_hake = Bubble(0.02, depth=depth, δ=2)
b_myctophid = Bubble(0.00061, depth=depth, δ=0.3)
b_siphonophore = Bubble(0.00058, depth=depth, δ=0.2)
sergestid = resize(Models.krill_mcgeehee, 0.045)
freqs_squid = [5,    18,  38,   70,    90,   120,   150, 200] .+ 5
TS_squid = [-45, -33, -34.7, -36.5, -37., -37.25, -37.4, -37.5] .+ 10log10(0.2^2)
spline_squid = Spline1D(freqs_squid, TS_squid)

ts_hake(freq) = target_strength(b_hake, freq*1e3)
ts_myctophid(freq) = target_strength(b_myctophid, freq*1e3)
ts_siph(freq) = target_strength(b_siphonophore, freq*1e3)
ts_sergestid(freq) = target_strength(sergestid, freq*1e3, 1470)
ts_squid(freq) = spline_squid(freq)

TS_bb = [ts_hake.(freqs_bb) ts_myctophid.(freqs_bb) ts_squid.(freqs_bb) ts_sergestid.(freqs_bb) ts_siph.(freqs_bb)]
TS_nb = TS_bb[i_nb, :]
TS_plot = [ts_hake.(freqs_plot) ts_myctophid.(freqs_plot) ts_squid.(freqs_plot) ts_sergestid.(freqs_plot) ts_siph.(freqs_plot)]

Σ_bb = exp10.(TS_bb / 10)
Σ_nb = exp10.(TS_nb / 10)
Σ_plot = exp10.(TS_plot / 10)

cond(Σ_bb)
cond(Σ_nb)

labels = ["Hake" "Myctophid" "Squid" "Sergestid" "Siphonophore"]
pal = [1 2 3 4 5]
p1 = plot(freqs_plot, TS_plot, labels=labels, color=pal, legend=:bottomright,
    xlabel="Frequency (kHz)", ylabel="TS (dB re m²)", title="A)",
    titlealign=:left)
scatter!(p1, freqs_bb, TS_bb, label="", color=pal, markerstrokewidth=0)
vline!(p1, freqs_nb, linestyle=:dot, color=:grey, label="")


Random.seed!(123)
η_bb = 0.5
η_nb = 0.5
N = [0.25, 10, 0.0, 170, 7] / 100 # density (per 100 m³, divide by 100 to get m⁻³).  No squid.
Sv_bb = 10log10.(Σ_bb * N) .+ η_bb .* randn(length(freqs_bb))
sv_bb = exp10.(Sv_bb / 10)
Sv_nb = 10log10.(Σ_nb * N) .+ η_nb .* randn(length(freqs_nb))
# sv_nb = sv_bb[i_nb]
exp10.(Sv_nb / 10)
Sv_plot = 10log10.(Σ_plot * N)

p2 = plot(freqs_plot, Sv_plot, color=:black, label="Theoretical",
    xlabel="Freq (kHz)", ylabel="Sv (dB re m⁻¹)", titlealign=:left, title="B)")
scatter!(p2, freqs_bb, Sv_bb, marker=:o, label="Broadband", color=1)
scatter!(p2, freqs_nb, Sv_nb, markersize=8, color=2, label="Narrowband")
plot(p1, p2, size=(800, 400), margin=10px)
savefig(joinpath(@__DIR__, "plots/meso_mix_scenario_Sv.png"))


#=
Setting up the different inference scenarios.
=#
# Vague, but slightly restrictive priors
T = Distribution{Univariate, Continuous}
# μprior2 = max.(5N, 0.01)
# prior2 = T[truncated(Normal(0, μ), 0, Inf) for μ in μprior2]
μprior2 = max.(0.5N, 0.01)
prior2 = T[Normal(log10.(μ), 1.0) for μ in μprior2]

# Add estimate of siphonophore density from ROV survey
prior3 = copy(prior2)
# prior3[5] = truncated(Normal(N[5], 0.1N[5]), 0, Inf)
prior3[5] = Normal(log10.(N[5]), 0.06)

## Add estimates of sergestid density from ROV survey
prior4 = copy(prior3)
# prior4[4] = truncated(Normal(N[4], 0.1N[4]), 0, Inf)
prior4[4] = Normal(log10(N[4]), 0.06)

## Did eDNA sampling, found no squid
prior5 = copy(prior4)
# prior5[3] = Uniform(0, 1e-9)
prior5[3] = Normal(log10(1e-9), 0.01)

## Case where *only* did eDNA sampling, so absence of squid is only other info available
prior6 = copy(prior2)
# prior6[3] = Uniform(0, 1e-9)
prior6[3] = Normal(log10(1e-9), 0.01)

priors = [prior2, prior4, prior6, prior5]
scenarios = ["Acoustics\nonly", "+Video", "+eDNA", "Video\n+eDNA"]

@model function echomodel(data, params)
    Σbs = exp10.(params.TS ./ 10) # convert TS to σbs
    cv ~ Exponential(1.0)
    logn ~ arraydist(params.priors)
    n = exp10.(logn)
    sv_pred = Σbs * n
    # ηsv = max.(cv .* sv_pred, sqrt(eps()))
    # data.backscatter .~ Normal.(sv_pred, ηsv)
    data.backscatter .~ Normal.(10log10.(sv_pred), cv)
end

data_bb = (backscatter=Sv_bb, freqs=freqs_bb, coords=(depth,))
data_nb = (backscatter=Sv_nb, freqs=freqs_nb, coords=(depth,))


# 5000 samples is overkill, done for nice plotting
solver = MCMCSolver(sampler=NUTS(1000, 0.9), nchains=4, nsamples=2000)
chains = map(priors) do p
    params_nb = (TS = TS_nb, priors=p)
    params_bb = (TS = TS_bb, priors=p)
    chain_nb = solve(data_nb, echomodel, solver, params_nb)
    chain_bb = solve(data_bb, echomodel, solver, params_bb)
    return (nb = chain_nb, bb = chain_bb)
end

# MCMC diagnostic plots
for (i, chn) in enumerate(chains)
    for bandwidth in [:nb, :bb]
        p = plot(chn[bandwidth])
        str = replace.(scenarios, "\n" => "_")[i]
        savefig(joinpath(@__DIR__, "plots/diagnostic_posterior_$(bandwidth)_scenario_$(i)_$(str).png"))
    end
end

posteriors = map(enumerate(chains)) do (i, chn)
    df_bb = DataFrame((Array(group(chn.bb, :logn))), vec(labels))
    df_bb[!, :bandwidth] .= "Broadband"
    df_nb = DataFrame((Array(group(chn.nb, :logn))), vec(labels))
    df_nb[!, :bandwidth] .= "Narrowband"
    df = [df_bb; df_nb]
    df[!, :scenario] .= scenarios[i]
    return df
end

posteriors = @chain vcat(posteriors...) begin
    DataFrames.stack( Not([:scenario, :bandwidth]), variable_name=:scatterer, value_name=:density)
    @transform(:scenario = CategoricalArray(:scenario, levels=scenarios, ordered=true))
end


yl = @by(posteriors, :scatterer, :q = quantile((:density), 0.98)).q
# yl = [0.01, 0.75, 0.025, 10, 0.5] * 5
yl = [.01, 0.5, 0.5, 10, 1]
subplots = map(enumerate(unique(posteriors.scatterer))) do (i, scatterer)
    post = @subset(posteriors, :scatterer .== scatterer)

    prior_sample = [exp10.(rand(prior[i], 1_000)) for prior in priors]
    p = violin(prior_sample, color=:grey, alpha=0.3, ylabel="Density (# m⁻³)", ylims=(0, yl[i]))
    @df @subset(post, :bandwidth .== "Narrowband") violin!(p, :scenario, exp10.(:density), group=:scenario,
        color=2, side=:left, linewidth=0, legend=false, title=scatterer)
    @df @subset!(post, :bandwidth .== "Broadband") violin!(p, :scenario, exp10.(:density), side=:right, 
        linewidth=0, color=1)
    hline!(p, [(N[i])], color=:black, linestyle=:dash, label="")
    
    # ylims!(p, 0, yl[i])
    return p
end
push!(subplots,
    plot(ones(1,3), labels=["Prior" "Narrowband" "Broadband"], linewidth=10,
         color=[:grey 2 1], legend=:left, xaxis=false, xticks=false,
         yaxis=false, yticks=false, ylabel="", legend_font_pointsize=10))
plot(subplots..., size=(1200, 600), margin=5mm)
savefig(joinpath(@__DIR__, "plots/meso_mix_posteriors.png"))


plot(
    corrplot(exp.(Array(group(chains[1].bb, :logn)))),
    corrplot(exp.(Array(group(chains[3].bb, :logn)))),
    size=(1500, 500)
)

(mean(Array(group(chains[4].nb, :n)), dims=1) .- N') ./ N'
(mean(Array(group(chains[4].bb, :n)), dims=1) .- N') ./ N'

sqrt.(mean((Array(group(chains[4].nb, :n)) .- N').^2, dims=1)) ./ N'
sqrt.(mean((Array(group(chains[4].bb, :n)) .- N').^2, dims=1)) ./ N'

