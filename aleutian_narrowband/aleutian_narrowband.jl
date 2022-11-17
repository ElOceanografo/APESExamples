using Distributed
addprocs(8, topology=:master_worker, exeflags="--project=$(Base.active_project())")
using CSV
using StatsBase
@everywhere begin
    using ProbabilisticEchoInversion
    using Turing
    using Optim
    using Random
    using SDWBA
    using LinearAlgebra
    using Plots, StatsPlots, Plots.PlotMeasures
    using DataFrames, DataFramesMeta, CategoricalArrays
    using DimensionalData
    using DimensionalData.Dimensions: @dim, YDim, XDim

    include(joinpath(@__DIR__, "../src/bubbles.jl"))
    const freqs = [18, 38, 70, 120, 200]
end
@everywhere begin # for some reason this doesn't work when it's in the @everywhere block above
    @dim F YDim "Frequency (kHz)"
    @dim Z YDim "Depth (m)"
    @dim D XDim "Distance (km)"
end

dfs = map(freqs) do f
    echo = CSV.read(joinpath(@__DIR__, "data/DY2104_T6_shelfbreak_$(f)_kHz.csv"),
        DataFrame)
    echo = @chain echo begin
        @subset(:Layer_depth_min .< :Exclude_below_line_range_mean)
        @transform(:freq = f,
                   :Standard_error = :Standard_deviation ./ :C_good_samples)
        @select(:depth = :Layer_depth_min,
                :distance = :Dist_M,
                :Interval,
                :Layer,
                :freq,
                :Sv = :Sv_mean,
                :Standard_error)
        @transform(:sv = exp10.(:Sv/10))
    end
end
echodata = vcat(dfs...)
allowmissing!(echodata)
noise_floor = -120
echodata.Sv[echodata.Sv .< noise_floor] .= noise_floor
echodata[(echodata.Sv .> 0), [:sv, :Sv]] .= missing
replace!(echodata.Sv, -Inf => missing)

echodata.distance = (echodata.distance .- minimum(echodata.distance)) / 1e3
echodata = @subset(echodata,  5 .< :depth .< 400, :distance .< 30)
depths = disallowmissing(sort(unique(echodata.depth)))
distances = sort(unique(echodata.distance))
intervals = sort(unique(echodata.Interval))

Sv = map(freqs) do f 
    @chain echodata begin
        @subset(:freq .== f)
        unstack(:depth, :distance, :Sv)
        @orderby(:depth)
        select(Not(:depth))
        Array()
    end
end
Sv = DimArray(cat(Sv..., dims=3), (Z(depths), D(distances), F(freqs)))
sv = exp10.(Sv ./ 10)

bottom = @chain dfs[1] begin
    @subset(5 .< :depth .< 400)
    unstack(:depth, :distance, :Sv)
    @orderby(:depth)
    select(Not(:depth))
    Array()
    DimArray((Z(depths), D(distances)))
end
bottom .*= 0
bottom[ismissing.(bottom)] .= 1
b = [RGBA(0.5, 0.5, 0.5, x) for x in bottom]

heatmap(Sv[F(At(70))], yflip=true, background_color_inside="#333333")
plot!(distances, depths, b, yflip=true, aspect_ratio=0.03, xlims=(0, 18), ylims=(5, 395))

annotation_x = [16,  5,   9,    12,  10]
annotation_y = [40,  30,  90,  220, 290]
annotation_text = text.(["A", "B", "C",  "D", "E"], color=:white, 12)

sv_plots = map(freqs) do f
    if f == 200
        xticks=true
        xlabel="Distance (km)"
    else
        xticks=false
        xlabel=""
    end
    
    p = heatmap(Sv[F(At(f))], yflip=true, title="", xticks=xticks, xlabel=xlabel, clim=(-90, -60),
        # background_color_inside="#333333",
        topmargin=0px, bottommargin=0px, colorbar_title="Sᵥ ($(f) kHz)") 
    annotate!(p, annotation_x, annotation_y, annotation_text)
    plot!(distances, depths, b, yflip=true, aspect_ratio=0.01, xlims=(0, 18), ylims=(10, 400))
end
plot(sv_plots..., layout=(5, 1), size=(800, 900), leftmargin=20px)
savefig(joinpath(@__DIR__, "plots/echograms.png"))

@everywhere begin

    @model function echomodel(data, params)
        ϵ ~ Exponential(0.5)
        a ~ Uniform(1e-5, 2e-3)
        δ = 0.5
        bubble = Bubble(a, depth=data.coords[1], δ=δ)
        TS_bubble = [target_strength(bubble, f*1e3) for f in data.freqs]
        TS = [params.TS TS_bubble]
        Σ = exp10.(TS ./ 10)
        logn ~ arraydist(params.prior)
        n = exp10.(logn)
        Sv_pred = 10log10.(Σ * n)

        for i in eachindex(data.freqs)
            if ! ismissing(data.backscatter[i])
                data.backscatter[i] ~ censored(Normal(Sv_pred[i], ϵ), params.noise_floor, Inf)
            end
        end

        return Sv_pred
    end

end

krill = resize(Models.krill_mcgeehee, 0.025)
# values from Lucca et al. 2021
krill.g .= 1.019
krill.h .= 1.032
TS_krill = [target_strength(krill, f*1e3, 1468.0) for f in freqs]
TS_fish = [-34.6, -35.0, -35.6, -36.6, -38.5]
TS_myctophid = [-73.0, -58.0, -65, -67.2, -70.0]

labels = ["Large fish" "Myctophids" "Krill" "Bubble"]
pars = (TS =    [TS_fish TS_myctophid TS_krill], 
        prior = [Normal(-3, 3),  # big fish
                 Normal(-2, 3),  # myctophid
                 Normal(-2, 3),  # krill
                 Normal(-2, 3)], # small bubble-like
        noise_floor = noise_floor)

solver_map = MAPSolver(optimizer=SimulatedAnnealing(), verbose=false)
solution_map = apes(Sv, echomodel, solver_map, params=pars, distributed=true)

fmissing(x, f) = ismissing(x) ? missing : f(x)

heatmap(fmissing.(solution_map, x -> coef(x)[:a]), yflip=true, clims=(0, 0.0007))
heatmap(fmissing.(solution_map, x -> coef(x)[Symbol("logn[2]")]), yflip=true,
    clims=(-10, 0))

solver_mcmc = MCMCSolver(verbose=false, nsamples=1000, nchains=4)
solution_mcmc = apes(Sv, echomodel, solver_mcmc, params=pars, distributed=true)

thresh_mask = fmissing.(solution_mcmc, chn -> mean(chn[Symbol("logn[4]")]) .> -6 ? 1.0 : missing)
p1 = heatmap(fmissing.(solution_mcmc, chn -> mean(1e3chn[:a])) .* thresh_mask, yflip=true, 
    c=cgrad(:hawaii, rev=true), clims=(0.2, 1.2), 
    colorbar_title="Bubble radius (mm)", size=(600, 350))
    plot!(p1, distances, depths, b, yflip=true, aspect_ratio=0.04, xlims=(0, 18), ylims=(5, 400))
annotate!(p1, annotation_x, annotation_y, annotation_text)

p2 = heatmap(fmissing.(solution_mcmc, chn -> std(1e3chn[:a])) .* thresh_mask, yflip=true, c=:viridis,
    colorbar_title="Bubble radius C.V.")
plot!(p2, distances, depths, b, yflip=true, aspect_ratio=0.04, xlims=(0, 18), ylims=(5, 400))
annotate!(p2, annotation_x, annotation_y, annotation_text)
plot(p1, p2, size=(1000, 350), margin=15px)
savefig(joinpath(@__DIR__, "plots/bubble_esr.png"))


meanplots = map(1:4) do i
    μ = fmissing.(solution_mcmc, chn -> mean(chn[Symbol("logn[$(i)]")]))
    ul = quantile(skipmissing(vec(μ)), 0.999)
    ll = ul - 4
    cl = (ll, ul)
    xticks = i == 4 ? true : false
    xlabel = i == 4 ? "Distance (km)" : ""
    p = heatmap(μ, colorbartitle="log₁₀($(labels[i]) m⁻³)", yflip=true,
        xlabel=xlabel, xticks=xticks, clims=cl, background_color_inside="#888888")
    annotate!(p, annotation_x, annotation_y, annotation_text)
    return p
end

logcv(x) = std(exp10.(x)) / mean(exp10.(x))
cvplots = map(1:4) do i
    cv = fmissing.(solution_mcmc, chn -> logcv(chn[Symbol("logn[$(i)]")]))
    cl = tuple(quantile(skipmissing(vec(cv)), [0.001, 0.999])...)
    xticks = i == 4 ? true : false
    xlabel = i == 4 ? "Distance (km)" : ""
    p = heatmap(cv, colorbartitle="$(labels[i]) C.V.", yflip=true,
        xlabel=xlabel, xticks=xticks, c=:viridis, clims=(0, 15), background_color_inside="#888888")
    annotate!(p, annotation_x, annotation_y, annotation_text)
    return p
end
plot(
    plot(meanplots..., layout=(4, 1)),
    plot(cvplots..., layout=(4, 1)),
    layout=(1, 2), size=(1200, 1000), margin=15px
) 
savefig(joinpath(@__DIR__, "plots/posteriors.png"))

heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:ϵ])), yflip=true, c=:viridis)
savefig(joinpath(@__DIR__, "plots/epsilon.png"))
heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:lp])), yflip=true, c=:viridis)
savefig(joinpath(@__DIR__, "plots/mean_log_probability.png"))

rhats = fmissing.(solution_mcmc, chn -> maximum(abs.(DataFrame(ess_rhat(chn)).rhat .- 1)))
heatmap(rhats, yflip=true, clims=(0, 0.1), c=:viridis)
savefig(joinpath(@__DIR__, "plots/max_rhat.png"))