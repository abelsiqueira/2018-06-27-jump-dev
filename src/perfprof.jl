using CUTEst, BenchmarkProfiles, Plots, Optimize, MiniLogging
pyplot()

MiniLogging.basic_config(MiniLogging.INFO)
optlog = get_logger("optimize")

function perfprof()
  pnames = CUTEst.select(max_var = 10_000, contype=:unc)
  sort!(pnames)
  problems = (CUTEstModel(p) for p in pnames)
  solvers = Dict{Symbol,Function}(:lbfgs => lbfgs, :trunk => trunk)
  bmark_args = Dict(:max_f => 10_000, :max_time => 30.0)
  stats, p = bmark_and_profile(solvers, problems, bmark_args=bmark_args)
  png(p, "perfprof")
end

perfprof()
