using Distributed
addprocs(8, topology=:master_worker, exeflags="--project=$(Base.active_project())")
using CSV
@everywhere begin
    using ProbabilisticEchoInversion
    using Turing
    using Optim
    using Random
    using SDWBA
    # using Dierckx
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
echodata[(echodata.Sv .> 0), [:sv, :Sv]] .= missing
replace!(echodata.Sv, -Inf => missing)
echodata[(echodata.freq .== 70) .& (echodata.depth .> 475), [:sv, :Sv]] .= missing
echodata[(echodata.freq .== 120) .& (echodata.depth .> 350), [:sv, :Sv]] .= missing
echodata[(echodata.freq .== 200) .& (echodata.depth .> 250), [:sv, :Sv]] .= missing
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
sv = 10 .^ (Sv ./ 10)
heatmap(Sv[F(At(38))], yflip=true, clim=(-90, -60))

sv_plots = map(freqs) do f
    if f == 200
        xticks=true
        xlabel="Distance (km)"
    else
        xticks=false
        xlabel=""
    end
    
    heatmap(Sv[F(At(f))], yflip=true, clim=(-90, -60), title="", xticks=xticks, xlabel=xlabel,
        topmargin=0px, bottommargin=0px, colorbar_title="Sᵥ ($(f) kHz)") 
end
plot(sv_plots..., layout=(5, 1), size=(800, 900), leftmargin=20px)
savefig(joinpath(@__DIR__, "plots/echograms.png"))

@everywhere begin

    @model function echomodel(data, params)
        cv ~ Exponential(0.1)
        a ~ Uniform(1e-5, 1e-2)#truncated(Normal(2e-3, 4e-3), 0, 0.21)
        δ = 0.3
        bubble = Bubble(a, depth=data.coords[1], δ=δ)
        TS_bubble = [target_strength(bubble, f*1e3) for f in freqs]
        TS = [params.TS TS_bubble]
        Σ = exp10.(TS ./ 10)
        logn ~ arraydist(params.prior)
        n = exp10.(logn)
        sv_pred = Σ * n
        ηsv = max.(cv * data.backscatter, sqrt(eps()))

        for i in eachindex(data.freqs)
            if ! ismissing(data.backscatter[i])
                data.backscatter[i] ~ Normal(sv_pred[i], ηsv[i])
            end
        end

        return 10log10.(sv_pred)
    end

end

krill = resize(Models.krill_mcgeehee, 0.025)
# estimates from Lucca et al. 2021
krill.g .= 1.019
krill.h .= 1.032
TS_krill = [target_strength(krill, f*1e3, 1468.0) for f in freqs]
TS_fish = [-34.6, -35.0, -35.6, -36.6, -38.5]
TS_myctophid = [-73.0, -58.0, -65, -67.2, -70.0]

pars = (TS =    [TS_fish TS_myctophid TS_krill], 
        prior = [Normal(-3, 3),  # big fish
                 Normal(-2, 3),  # myctophid
                 Normal(-2, 3),  # krill
                 Normal(-2, 3)]) # small bubble-like

solver_map = MAPSolver(optimizer=SimulatedAnnealing(), verbose=false)

solution_map = apes(sv, echomodel, solver_map, params=pars, distributed=true)

fmissing(x, f) = ismissing(x) ? missing : f(x)

heatmap(fmissing.(solution_map, x -> coef(x)[Symbol("logn[4]")]), yflip=true)


solver_mcmc = MCMCSolver(verbose=false)
solution_mcmc = apes(sv, echomodel, solver_mcmc, params=pars, distributed=true)

threshold_mask = mapspectra(x -> all(x.backscatter .> -90), Sv)

heatmap(fmissing.(solution_mcmc, chn -> median(1e3chn[:a])), yflip=true)
heatmap(fmissing.(solution_mcmc, chn -> std(1e3chn[:a])), yflip=true, c=:viridis, clims=())
# heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:δ])), yflip=true)

plot(
    heatmap(fmissing.(solution_mcmc, chn -> median(chn[Symbol("logn[1]")])), colorbartitle="log₁₀(large fish m⁻³)",
        xlabel="", xticks=false, clims=(-6, -3)),
    heatmap(fmissing.(solution_mcmc, chn -> median(chn[Symbol("logn[2]")])), colorbartitle="log₁₀(myctophid m⁻³)",
        xlabel="", xticks=false, clims=(-4, -1)),
    heatmap(fmissing.(solution_mcmc, chn -> median(chn[Symbol("logn[3]")])), colorbartitle="log₁₀(krill m⁻³)",
        xlabel="", xticks=false, clims=(-4, -1)),
    heatmap(fmissing.(solution_mcmc, chn -> median(chn[Symbol("logn[4]")])), colorbartitle="log₁₀(small bubble m⁻³)",
        clim=(-7, -6)),
    yflip=true, layout=(4, 1), size=(600, 600), 
)


plot(
    heatmap(fmissing.(solution_mcmc, chn -> std(chn[Symbol("logn[1]")])), colorbartitle="log₁₀(large fish m⁻³)",
        xlabel="", xticks=false, c=:viridis),
    heatmap(fmissing.(solution_mcmc, chn -> std(chn[Symbol("logn[2]")])), colorbartitle="log₁₀(myctophid m⁻³)",
        xlabel="", xticks=false, c=:viridis),
    heatmap(fmissing.(solution_mcmc, chn -> std(chn[Symbol("logn[3]")])), colorbartitle="log₁₀(krill m⁻³)",
        xlabel="", xticks=false, c=:viridis),
    heatmap(fmissing.(solution_mcmc, chn -> std(chn[Symbol("logn[4]")])), colorbartitle="log₁₀(small bubble m⁻³)",
        c=:viridis),
    yflip=true, layout=(4, 1), size=(600, 600),
)


heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:cv])), yflip=true, c=:viridis, clims=(0.11, 0.93))
heatmap(fmissing.(solution_mcmc, chn -> mean(chn[:lp])), yflip=true, c=:viridis, clims=(50, 70))