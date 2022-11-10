using ProbabilisticEchoInversion
using Turing
using Random
using SDWBA
using Plots, StatsPlots, Plots.PlotMeasures
using DataFrames, DataFramesMeta

include(joinpath(@__DIR__, "../src/bubbles.jl"))

Random.seed!(123)

freqs_bb = [15:25; 35:38; 50:87; 100:150; 170:240]
freqs_nb = [18, 38, 70, 120, 200]
freqs_plot = 10:250
i_nb = indexin(freqs_nb, freqs_bb)

depth = 200.0
a = 0.0005
δ = 0.3
fish = Bubble(a, depth=depth, δ=δ)
krill = resize(Models.krill_mcgeehee, 0.025)
# EBS/GOA average values from Lucca et al. 2021
krill.g .= 1.019
krill.h .= 1.032

ts_fish(freq) = target_strength(fish, freq*1e3)
ts_krill(freq) = target_strength(krill, freq*1e3, 1470)

TS_sim = [ts_fish.(freqs_bb) ts_krill.(freqs_bb)]#
Σ_sim = exp10.(TS_sim / 10)
TS_plot = [ts_fish.(freqs_plot) ts_krill.(freqs_plot)]
Σ_plot = exp10.(TS_plot / 10)

tsplot = plot(freqs_plot, TS_plot, label=["Fish" "Krill"], legend=:bottomright,
    xlabel="Frequency (kHz)", ylabel="TS (dB re. m⁻¹)");
vline!(tsplot, freqs_nb, linestyle=:dot, color=:black, label="");
scatter!(tsplot, freqs_bb, TS_sim, color=[1 2], label="", markerstrokewidth=0)

N1 = [0.001, 1e3]
N2 = [0.2, 4e2]
N3 = [0.5, 10]

scenarios = [N1, N2, N3]

svs = map(scenarios) do N
    TS = TS_sim
    Σ = exp10.(TS_sim / 10)
    sv_pred = Σ * N
    σ = sv_pred / sqrt(π/2) # convert mean sv to Rayleigh parameter σ
    # 
    return mean(rand.(Rayleigh.(σ)) for _ in 1:3)
end
Svs = [10log10.(sv) for sv in svs]
svs_nb = [sv[i_nb] for sv in svs]
Svs_nb = [Sv[i_nb] for Sv in Svs]

Svs_plot = [10log10.(Σ_plot * N) for N in scenarios]
pal = [2 3 1]
svplot = plot(freqs_plot, Svs_plot, legend=:bottomright, label="", c=pal,
    xlabel="Frequency (kHz)", ylabel="Sv (dB re m⁻¹)");
vline!(svplot, freqs_nb, linestyle=:dot, color=:black, label="");
scatter!(svplot, freqs_bb, Svs, legend=:bottomright, label="", c=pal, markershape=[:circle :square :utriangle],
    markerstrokewidth=0, markersize=2);
scatter!(svplot, freqs_nb, Svs_nb, color=pal, markershape=[:circle :square :utriangle], 
    markersize=5, markerstrokewidth=2, labels=["Krill-dominated" "Mix" "Fish-dominated"])
plot(tsplot, svplot, size=(1000, 400), margin=20px)
savefig(joinpath(@__DIR__, "plots/fish_krill_Sv.png"))

data_bb = [(backscatter=Sv, freqs=freqs_bb, coords=(depth,))
    for Sv in Svs]
data_nb = [(backscatter=data.backscatter[i_nb], freqs=freqs_bb[i_nb], coords=(depth,))
    for data in data_bb]


μprior = [0.1, 200]
params_bb = (TS_const = ts_krill.(freqs_bb), δ = δ, μprior=μprior)
params_nb = (TS_const = ts_krill.(freqs_nb), δ = δ, μprior=μprior)

@model function echomodel(data, params)
    depth = data.coords[end]
    a ~ truncated(Normal(5e-4, 3.5e-4), 0.0001, 0.001)
    b = Bubble(a, depth=data.coords[1], δ=params.δ)
    TS_fish = map(f -> target_strength(b, f.*1e3), data.freqs)
    TS = [TS_fish params.TS_const]
    Σbs = exp10.(TS ./ 10)
    logn ~ arraydist(Normal.(log10.(params.μprior), 2.0))
    n = exp10.(logn)
    Sv_pred = 10log10.(Σbs * n)
    ϵ ~ Exponential(1.0)
    data.backscatter .~ Normal.(Sv_pred, ϵ)
    return 10log10.(Σ_plot * n)
end

solver = MCMCSolver(sampler=NUTS(0.8), nchains=4, nsamples=2500)

chains = map(zip(data_nb, data_bb)) do data
    sv_nb, sv_bb = data
    chain_nb = solve(sv_nb, echomodel, solver, params_nb)
    chain_bb = solve(sv_bb, echomodel, solver, params_bb)
    return (nb = chain_nb, bb = chain_bb)
end

for i in 1:3
    chn_nb = chains[i].nb
    chn_bb = chains[i].bb
    p1 = plot(chn_nb, palette=:Blues);
    plot!(p1, chn_bb, palette=:Oranges);
    p2 = scatter(exp10.(chn_nb[Symbol("logn[1]")]), exp10.(chn_nb[Symbol("logn[2]")]), chn_nb[:a],
        xlabel="n[1]", ylabel="n[2]", zlabel="a", markerstrokewidth=0, alpha=0.5, palette=:Blues) ;
    scatter!(p2, exp10.(chn_bb[Symbol("logn[1]")]), exp10.(chn_bb[Symbol("logn[2]")]), chn_bb[:a],
        markerstrokewidth=0, alpha=0.5, palette=:Oranges);
    plot(p1, p2, size=(1500, 800))
    savefig(joinpath(@__DIR__, "plots/diagnostic_posterior_$(i).png"))
end

titles = ["Krill-dominated", "Mixture", "Fish-dominated"]
xlimits = [[1e-6, 1], [1e-6, 1], [1e-6, 1]]
ylimits = [[0, 2000], [0, 2000], [0, 200]]
post_plots = map(enumerate(chains)) do (i, chn)
    a_nb = reshape(chn.nb[:a], :, 1)
    n_nb = exp10.(reshape(Array(group(chn.nb, :logn)), :, 2)) 
    a_bb = reshape(chn.bb[:a], :, 1)
    n_bb = exp10.(reshape(Array(group(chn.bb, :logn)), :, 2)) 

    a_plot = plot(truncated(Normal(0.5, 0.35), 0.01, 1.5), 
        fill=(0, "#aaaaaa"), color="#aaaaaa", label="Prior")
    density!(a_plot, 1e3*a_nb, fill=true, alpha=0.75, label="Narrowband",
        xlabel="ESR (mm)", ylabel="PDF", title=titles[i], color=1,
        legend = i==3 ? :topright : :none)
    density!(a_plot, 1e3*a_bb, fill=true, alpha=0.75, label="Broadband", color=2)
    vline!(a_plot, [1e3*a], color=:black, linestyle=:dot, label="")
    ylims!(a_plot, (0, 15))

    ff =  range(xlimits[i]..., length=100)
    kk =  range(ylimits[i]..., length=100)
    prior = [prod(pdf.(Normal.(log10.(μprior), 1.0), log10.([x, y])))
        for x in ff, y in kk]
    n_plot = contourf(ff, kk, prior', c=:Greys_3, alpha=0.1, legend=:none)

    n_plot = scatter!(n_nb[:, 1], n_nb[:, 2], markerstrokewidth=0, alpha=0.25, 
        legend=false, markersize=1.5, color=1,
        xlabel="Fish m⁻³", ylabel="Krill m⁻³")
    scatter!(n_plot, n_bb[:, 1], n_bb[:, 2], markerstrokewidth=0, alpha=0.25,
        markersize=1.5, color=2)
    vline!(scenarios[i][[1]], color=:black, linestyle=:dot)
    hline!(scenarios[i][[2]], color=:black, linestyle=:dot)
    xlims!(n_plot, xlimits[i]...)
    ylims!(n_plot, ylimits[i]...)
    post_plot = plot(a_plot, n_plot, layout=(2,1))
end
plot(post_plots..., layout=(1, 3), size=(1000, 600), margin=15px)
savefig(joinpath(@__DIR__, "plots/fish_krill_posteriors.png"))


df = map(enumerate(chains)) do tup
    i, chn = tup
    dfnb = DataFrame(chn.nb)
    dfnb[!, :bandwidth] .= "narrowband"
    dfbb = DataFrame(chn.bb)
    dfbb[!, :bandwidth] .= "broadband"
    df = vcat(dfbb, dfnb)
    df[!, :scenario] .= i
    return df
end
df = vcat(df...)
truth = DataFrame(scenario = repeat(1:3, inner=2), 
                  scatterer = repeat(["fish", "krill"], outer=3),
                  ntrue=vcat(scenarios...))
comparison = @chain df begin
    select(:scenario, :bandwidth, :a, "logn[1]" => :lognfish, "logn[2]" => :lognkrill)
    @by([:scenario, :bandwidth],
        :a = mean(:a),
        :fish = mean(exp10.(:lognfish)),
        :krill = mean(exp10.(:lognkrill)))
    DataFrames.stack([:fish, :krill], variable_name=:scatterer, value_name=:n)
    leftjoin(truth, on=[:scenario, :scatterer])
    @transform(:error = :n .- :ntrue)
    @transform(:rel_error = :error ./ :ntrue)
end

rhats = map(enumerate(chains)) do tup
    i, chns = tup
    df_nb = chns.nb |> ess_rhat |> DataFrame |> df -> unstack(df[:, [:parameters, :rhat]], :parameters, :rhat)
    df_nb[!, :bandwidth] .= "Narrowband"
    df_bb = chns.bb |> ess_rhat |> DataFrame |> df -> unstack(df[:, [:parameters, :rhat]], :parameters, :rhat)
    df_bb[!, :bandwidth] .= "Broadband"
    [df_nb; df_bb]
end
rhats = vcat(rhats...)

ppreds = map(1:3) do i
    generated_quantities(echomodel(data_bb[i], params_bb), chains[i].bb) |> vec
end
for i in 1:3
    ppred = ppreds[i]
    scatter(data_bb[i].freqs, data_bb[i].backscatter, markerstrokewidth=0, markersize=2, 
        color=:black, label="Data", legend=:bottomright)
    plot!(freqs_plot, quantile(ppred, 0.025), fillrange=quantile(ppred, 0.975), color=1, alpha=0.5, 
        label="Posterior predictions\nmean and 95% interval")
    plot!(freqs_plot, mean(ppred), color=1, linewidth=2, label="")
    plot!(freqs_plot, Svs_plot[i], color=2, linewidth=2, label="Truth")
    savefig(joinpath(@__DIR__, "plots/posterior_$(i)_predictive.png"))
end