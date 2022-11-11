using ProbabilisticEchoInversion
using Turing
using Random
using SDWBA
using Dierckx
using LinearAlgebra
using Plots, StatsPlots, Plots.PlotMeasures
using DataFrames, DataFramesMeta, CategoricalArrays


include(joinpath(@__DIR__, "../src/bubbles.jl"))

freqs_bb = [15:25; 35:38; 50:87; 100:150]
freqs_nb = [18, 38, 70, 120]
freqs_plot = 10:160
i_nb = indexin(freqs_nb, freqs_bb)

depth = 300.0
# b_hake = Bubble(0.02, depth=depth, δ=2)
freqs_hake = [18, 38, 70, 120, 200]
TS_hake = [-34.6, -35.0, -35.6, -36.6, -38.5]
spline_hake = Spline1D(freqs_hake, TS_hake)
b_myctophid = Bubble(0.00065, depth=depth, δ=0.3)
b_siphonophore = Bubble(0.00058, depth=depth, δ=0.2)
sergestid = resize(Models.krill_mcgeehee, 0.045)
freqs_squid = [5,    18,  38,   70,    90,   120,   150, 200] .+ 5
TS_squid = [-45, -33, -34.7, -36.5, -37., -37.25, -37.4, -37.5] .+ 10log10(0.2^2)
spline_squid = Spline1D(freqs_squid, TS_squid)

ts_hake(freq) = spline_hake(freq)# target_strength(b_hake, freq*1e3)
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
N = [0.25, 10, 0.0, 170, 7] # density (per 100 m³, divide by 100 to get m⁻³).  No squid.

sv_bb = mean(rand.(Rayleigh.(Σ_bb * N / sqrt(π/2))) for _ in 1:10)
Sv_bb = 10log10.(sv_bb)
sv_nb = sv_bb[i_nb]
Sv_nb = Sv_bb[i_nb]
Sv_plot = 10log10.(Σ_plot * N)

p2 = plot(freqs_plot, Sv_plot .- 20, color=:black, label="Theoretical",
    xlabel="Freq (kHz)", ylabel="Sv (dB re m⁻¹)", titlealign=:left, title="B)")
scatter!(p2, freqs_bb, Sv_bb .- 20, marker=:o, label="Broadband", color=1)
scatter!(p2, freqs_nb, Sv_nb .- 20, markersize=8, color=2, label="Narrowband")
plot(p1, p2, size=(800, 400), margin=10px)
savefig(joinpath(@__DIR__, "plots/meso_mix_scenario_Sv.png"))


#=
Setting up the different inference scenarios.
=#
# Vague, but slightly restrictive priors
T = Distribution{Univariate, Continuous}
# μprior2 = max.(5N, 0.01)
# prior2 = T[truncated(Normal(0, μ), 0, Inf) for μ in μprior2]
μprior1 = log10.(max.(N, 0.25))
prior1 = T[Normal(μ, 2.0) for μ in μprior1]

# Add estimate of sergestid and siphonophore density from ROV survey
prior2 = copy(prior1)
prior2[4] = Normal(log10(N[4]), 0.06)
prior2[5] = Normal(log10.(N[5]), 0.06)

# Did eDNA sampling, found no squid
prior3 = copy(prior1)
prior3[3] = Normal(log10(1e-9), 0.01)

# Video and eDNA sampling
prior4 = copy(prior2)
prior4[3] = Normal(log10(1e-9), 0.01)

priors = [prior1, prior2, prior3, prior4]
scenarios = ["Acoustics\nonly", "+Video", "+eDNA", "Video\n+eDNA"]

@model function echomodel(data, params)
    Σbs = exp10.(params.TS ./ 10) # convert TS to σbs
    ϵ ~ Exponential(1.0)
    logn ~ arraydist(params.priors)
    n = exp10.(logn)
    Sv_pred = 10log10.(Σbs * n)
    data.backscatter .~ Normal.(Sv_pred, ϵ)
    return 10log10.(params.Σ_plot * n) .+ ϵ .* randn(size(params.Σ_plot, 1))
end

data_bb = (backscatter=Sv_bb, freqs=freqs_bb, coords=(depth,))
data_nb = (backscatter=Sv_nb, freqs=freqs_nb, coords=(depth,))


# 10,00 samples is overkill, done for nice plotting
solver = MCMCSolver(sampler=NUTS(0.9), nchains=4, nsamples=2500, kwargs=(progress=true,))
chains = map(priors) do p
    params_nb = (TS = TS_nb, priors=p, Σ_plot = Σ_plot)
    params_bb = (TS = TS_bb, priors=p, Σ_plot = Σ_plot)
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


yl = [quantile((df.density), [0.01, 1]) for df in groupby(posteriors, :scatterer)]

subplots = map(enumerate(unique(posteriors.scatterer))) do (i, scatterer)
    post = @subset(posteriors, :scatterer .== scatterer)
    prior_sample = [rand(prior[i], 10_000) for prior in priors]
    p = violin(prior_sample, color=:grey, alpha=0.3, ylabel="Density (# m⁻³)", ylims=yl[i])
    @df @subset(post, :bandwidth .== "Narrowband") violin!(p, :scenario, :density, group=:scenario,
        color=2, side=:left, linewidth=0, legend=false, title=scatterer)
    @df @subset!(post, :bandwidth .== "Broadband") violin!(p, :scenario, :density, side=:right, 
        linewidth=0, color=1)
    hline!(p, [log10(N[i])], color=:black, linestyle=:dash, linewidth=2, label="")
    return p
end
push!(subplots,
    plot(ones(1,3), labels=["Prior" "Narrowband" "Broadband"], linewidth=10,
         color=[:grey 2 1], legend=:left, xaxis=false, xticks=false,
         yaxis=false, yticks=false, ylabel="", legend_font_pointsize=10))
plot(subplots..., size=(1200, 600), margin=5mm)
savefig(joinpath(@__DIR__, "plots/meso_mix_posteriors.png"))

plot(
    corrplot(exp.(Array(group(chains[1].nb, :logn)))),
    corrplot(Array(group(chains[3].nb, :logn))),
    size=(1500, 500)
)


ppreds = map(1:4) do i
    params_nb = (TS = TS_nb, priors=priors[i], Σ_plot=Σ_plot)
    params_bb = (TS = TS_bb, priors=priors[i], Σ_plot=Σ_plot)
    (bb = generated_quantities(echomodel(data_bb, params_bb), chains[i].bb) |> vec,
     nb = generated_quantities(echomodel(data_nb, params_nb), chains[i].nb) |> vec)
end

post_pred_plots = map(1:4) do i
    p = scatter(freqs_bb, Sv_bb, markersize=2, color=:black, 
        xlabel="Frequency (kHz)", ylabel="Sᵥ (dB re m⁻¹)",
        title=scenarios[i], label="Data", legend=:bottomright)
    plot!(p, freqs_plot, vec(mapslices(x -> quantile(x, 0.025), hcat(ppreds[i].nb...), dims=2)),
        fillrange=vec(mapslices(x -> quantile(x, 0.975), hcat(ppreds[i].nb...), dims=2)),
        alpha=0.5, color=2, label="")
    plot!(p, freqs_plot, mean(ppreds[1].nb), color=2, label="Narrowband")
    plot!(p, freqs_plot, vec(mapslices(x -> quantile(x, 0.025), hcat(ppreds[i].bb...), dims=2)),
        fillrange=vec(mapslices(x -> quantile(x, 0.975), hcat(ppreds[i].bb...), dims=2)),
        alpha=0.5, color=1, label="")
    plot!(p, freqs_plot, mean(ppreds[1].bb), color=1, label="Broadband")
    return p
end
plot(post_pred_plots..., size=(800, 500))
savefig(joinpath(@__DIR__, "plots/posterior_predictive.png"))