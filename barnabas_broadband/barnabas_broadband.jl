using Distributed
addprocs(topology=:master_worker, exeflags="--project=$(Base.active_project())")
using DataFrames, CSV
using StatsPlots, StatsPlots.PlotMeasures
using ColorSchemes
using Dierckx
using StatsBase
using ImageFiltering

@everywhere begin
    using ProbabilisticEchoInversion
    using SDWBA
    include(joinpath(@__DIR__, "../src/bubbles.jl"))
    using MAT
    using DimensionalData
    using DimensionalData.Dimensions: @dim, YDim, XDim
    using Dates
    using Turing
    using Distributions
    using Optim
end

#=
constants and helper functions
=#
const MATLAB_EPOCH = Dates.DateTime(-0001, 12, 31)
const MS_PER_DAY = 24 * 60 * 60 * 1000
datenum2datetime(datenum) = MATLAB_EPOCH + Millisecond(round(Int, datenum * MS_PER_DAY))
datetime2datenum(dt) = (dt - MATLAB_EPOCH).value / MS_PER_DAY
dBmean(x; args...) = 10log10.(mean(exp10.(x ./ 10); args...))

@everywhere begin
    @dim F YDim "Frequency (kHz)"
    @dim Z YDim "Depth (m)"
    @dim D XDim "Distance (km)"
end


#=
Loading data
=#
data = matread(joinpath(@__DIR__, "data/processed_spectra.mat"))

echo = allowmissing(data["Sv"])
heatmap(dropdims(maximum(echo, dims=3), dims=3))
# Eliminate bottom echoes using a simple threshold
for i in axes(echo, 1), j in axes(echo, 2)
    if maximum(echo[i, j, :]) > -30
        echo[i, j, :] .= missing
    end
end

echo = DimArray(echo,
    (Z(data["Z"]),
        Ti(datenum2datetime.(data["Ti"]) .- Hour(8)),
    # (Z(data["Z"]),
    #     Ti(datenum2datetime.(data["Ti"][1:3:end]) .- Hour(8)),
        F(data["F"])))
const freqs = data["F"]
const freqs_nb = [18, 38, 70, 120, 200]
freqs_plot = 15:250
heatmap(echo[F(Near(120))], yflip=true, clims=(-90, -32))

echo = echo[Ti(DimensionalData.Between(DateTime(2021, 6, 19, 22, 35), DateTime(2021, 6, 20, 3)))]
tticks = DateTime(2021, 6, 19, 23):Minute(30):DateTime(2021, 6, 20, 1, 50)
tlabels = Dates.format.(tticks, "HH:MM")
heatmap(echo[F(Near(38))], yflip=true, xticks=(tticks, tlabels))



intervals = CSV.read(joinpath(@__DIR__, "data/intervals.csv"), DataFrame)
plot(intervals.Lon_M, intervals.Lat_M)


krill = resize(Models.krill_conti, 0.025)
# values from Lucca et al. 2021
krill.g .= 1.019
krill.h .= 1.032
TS_krill = [target_strength(krill, f*1e3, 1468.0) for f in freqs]
TS_krill_plot = [target_strength(krill, f*1e3, 1468.0) for f in freqs_plot]

spline_fish = Spline1D(freqs_nb, [-34.6, -35.0, -35.6, -36.6, -38.5])
TS_fish = spline_fish(freqs)
TS_fish_plot = spline_fish(freqs_plot)

labels = ["Large fish" "Krill" "Bubble"]
pars = (
    TS =    [TS_fish TS_krill],
    prior = [Normal(-3, 3),
            Normal(-2, 3),
            Normal(-2, 3)],
    n_in_band = [1; fill(5, 5); fill(22, 22); fill(45, 45); fill(65, 65)],
    TS_fish_plot = TS_fish_plot,
    TS_krill_plot = TS_krill_plot,
    freqs_plot = freqs_plot
)

@everywhere begin
    @model function echomodel(data, params)
        ϵ ~ Exponential(1.0)
        a ~ Uniform(0.05e-3, 0.5e-3)
        δ = 0.5
        bubble = Bubble(a, depth=data.coords[1], δ=δ)
        TS_bubble = [target_strength(bubble, f*1e3) for f in data.freqs]
        TS = [params.TS TS_bubble]
        Σ = exp10.(TS ./ 10)
        logn ~ arraydist(params.prior)
        n = exp10.(logn)
        Sv_pred = 10log10.(Σ * n)

        for i in findall(!ismissing, data.backscatter)
            data.backscatter[i] ~ Normal(Sv_pred[i], ϵ * sqrt(params.n_in_band[i]))
        end

        # generated quantities for posterior prediction
        TS_bubble_plot = [target_strength(bubble, f*1e3) for f in params.freqs_plot]
        TS_plot = [params.TS_fish_plot params.TS_krill_plot TS_bubble_plot]
        Sv_pred_plot = 10log10.(exp10.(TS_plot ./ 10) * n)
        return Sv_pred_plot
    end
end

solver_map = MAPSolver(optimizer=BFGS(), verbose=false)
solution_map = apes(echo, echomodel, solver_map, params=pars, distributed=true)

fmissing(x, f) = ismissing(x) ? missing : f(x)
heatmap(fmissing.(solution_map, x -> coef(x)[:a]), yflip=true)
heatmap(fmissing.(solution_map, x -> coef(x)[Symbol("logn[2]")]), yflip=true)

solver_mcmc = MAPMCMCSolver(verbose=false, optimizer=BFGS(), nsamples=500, nchains=4)
solution_mcmc = apes(echo, echomodel, solver_mcmc, params=pars, distributed=true)

p1 = heatmap(fmissing.(solution_mcmc, chn -> median(1e3chn[:a])), yflip=true, 
    c=cgrad(:hawaii, rev=true), title="A", title_align=:left,
    colorbar_title="Bubble radius (mm)", size=(600, 350))
p2 = heatmap(fmissing.(solution_mcmc, chn -> std(1e3chn[:a])), yflip=true, c=:viridis,
    title="B", title_align=:left, colorbar_title="Bubble radius C.V.")
plot(p1, p2, size=(1000, 300), margin=15px, xticks=(tticks, tlabels), background_color_inside="#888888")
savefig(joinpath(@__DIR__, "plots/bubble_esr.png"))

titles = ["A. Large fish", "B. Krill", "C. Bubble-like"]
meanplots = map(1:3) do i
    μ = fmissing.(solution_mcmc, chn -> mean((chn[Symbol("logn[$(i)]")])))
    ul = quantile(skipmissing(vec(μ)), 0.99)
    ll = ul - 3
    cl = (ll, ul)
    xticks = i == 3 ? (tticks, tlabels) : false
    xlabel = i == 3 ? "Time" : ""
    p = heatmap(μ, colorbartitle="log₁₀($(labels[i]) m⁻³)", yflip=true,
        xlabel=xlabel, xticks=xticks, clims=cl, background_color_inside="#888888",
        title=titles[i], title_align=:left)
    return p
end

logcv(x) = std(exp10.(x)) / mean(exp10.(x))
cvplots = map(1:3) do i
    cv = fmissing.(solution_mcmc, chn -> logcv(chn[Symbol("logn[$(i)]")]))
    cl = tuple(quantile(skipmissing(vec(cv)), [0.001, 0.99])...)
    xticks = i == 3 ? (tticks, tlabels) : false
    xlabel = i == 3 ? "Time" : ""
    p = heatmap(cv, colorbartitle="$(labels[i]) C.V.", yflip=true,
        xlabel=xlabel, xticks=xticks, c=:viridis, clims=(0, 3), background_color_inside="#888888")
    return p
end
plot(
    plot(meanplots..., layout=(3, 1)),
    plot(cvplots..., layout=(3, 1)),
    layout=(1, 2), size=(1200, 700), margin=15px
) 
savefig(joinpath(@__DIR__, "plots/posteriors.png"))


p_epsilon = heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:ϵ])), yflip=true, c=:viridis,
    xticks=(tticks, tlabels), colorbar_title="Residual error (dB)", title="A.")
p_logprob = heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:lp])), yflip=true, c=:viridis, clims=(-350, -125),
    xticks=(tticks, tlabels), colorbar_title="Mean log-probability", title="B.")

rhats = fmissing.(solution_mcmc, chn -> maximum(abs.(DataFrame(ess_rhat(chn)).rhat .- 1)))
sum(skipmissing(rhats .< 1.04)) / sum(.!ismissing.(rhats))
p_rhat = heatmap(rhats, yflip=true, c=:viridis, clims=(0, 5), xticks=(tticks, tlabels), title="C.")
plot(p_epsilon, p_logprob, p_rhat, layout=(2,2), title_align=:left, size=(1000, 600))
savefig(joinpath(@__DIR__, "plots/diagnostics.png"))

# plots for worst-case scenario for convergence
iworst = argmax(skipmissing(rhats))
chn = solution_mcmc[iworst]
println(DataFrame(ess_rhat(chn)))
plot(chn)
savefig(joinpath(@__DIR__, "plots/chains_worst.png"))

m = echomodel(collect(iterspectra(echo))[iworst], pars)
post_pred = generated_quantities(m, solution_mcmc[iworst])
p_post_pred = scatter(echo[iworst, :], marker=:o, color=:black, ylabel="Sᵥ (dB re m⁻¹)");
for i in 1:4
    plot!(p_post_pred, freqs_plot, hcat(post_pred[1:50, i]...), color=i, alpha=0.3, label="")
end
p_post_pred
savefig(p_post_pred, joinpath(@__DIR__, "plots/post_pred_worst.png"))