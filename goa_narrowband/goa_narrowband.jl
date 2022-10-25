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
        cv ~ Exponential(0.5)
        a ~ truncated(Normal(2e-3, 2e-3), 0, 0.21)
        δ ~ Uniform(0.1, 1.5)
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
    end

end

krill = resize(Models.krill_mcgeehee, 0.025)
TS_krill = [target_strength(krill, f*1e3, 1468.0) for f in freqs]
TS_fish = [-34.6, -35.0, -35.6, -36.6, -38.5]


solver_map = MAPSolver(optimizer=SimulatedAnnealing(), verbose=false)
pars = (TS =    [TS_fish TS_krill], 
        prior = [Normal(-2, 2), Normal(0, 2), Normal(0, 2)])
solution_map = apes(sv[1:50, :, :], echomodel, solver_map, params=pars, distributed=true)
heatmap([coef(x)[Symbol("logn[2]")] for x in solution_map], yflip=true)


solver_mcmc = MCMCSolver(verbose=false, nsamples=500)
solution_mcmc = apes(sv, echomodel, solver_mcmc, params=pars, distributed=true)

heatmap([ismissing(chn) ? missing : median(chn[:a]) for chn in solution_mcmc], yflip=true)