using Optimize, CUTEst, NLPModels, BenchmarkProfiles, Plots, MiniLogging, Ipopt
pyplot()

MiniLogging.basic_config(MiniLogging.INFO)
optlog = get_logger("optimize")

function ipopt_wrapper(nlp :: AbstractNLPModel; kwargs...)
  model = NLPtoMPB(nlp, IpoptSolver(print_level=0))
  Δt = time()
  status = MathProgBase.optimize!(model)
  Δt = time() - Δt
  status = status == 0 ? :first_order : :unknown
  return GenericExecutionStats(status, nlp, solution=model.inner.x,
                               objective=model.inner.obj_val,
                               elapsed_time=Δt)
end

function runcutest()
  solvers = Dict{Symbol,Function}(:Ipopt => ipopt_wrapper,
                                  :Optimize => lbfgs)
  pnames = sort(CUTEst.select(max_var=1000, min_var=100, contype=:unc))
  bmark_args = Dict(:atol => 1e-8, :rtol => 0.0, :max_f => 10_000, :max_time => 30.0)

  problems = (CUTEstModel(p) for p in pnames)
  stats, p = bmark_and_profile(solvers, problems, bmark_args=bmark_args)
  png(p, "vs-ipopt-sum-counters-large")
  stats, p = bmark_and_profile(solvers, problems, bmark_args=bmark_args,
                               cost=stat->stat.elapsed_time)
  png(p, "vs-ipopt-time-large")
end

runcutest()
