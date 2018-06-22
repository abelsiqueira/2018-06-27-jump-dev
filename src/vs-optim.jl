using Optimize, CUTEst, NLPModels, BenchmarkProfiles, Plots, Optim, MiniLogging
pyplot()

MiniLogging.basic_config(MiniLogging.INFO)
optlog = get_logger("optimize")

function optim_method(nlp :: AbstractNLPModel; kwargs...)
  f(x) = obj(nlp, x)
  g!(storage, x) = grad!(nlp, x, storage)
  Δt = time()
  output = optimize(f, g!, nlp.meta.x0, LBFGS(m = 5),
                    Optim.Options(g_tol = 1e-8,
                                  iterations = 10_000_000,
                                  f_calls_limit = 10000)
                   )
  Δt = time() - Δt
  status = output.g_converged ? :first_order : :unknown
  return GenericExecutionStats(status, nlp, solution=output.minimizer,
                               objective=output.minimum, dual_feas=output.g_residual,
                               iter=output.iterations, elapsed_time=Δt)
end

function runcutest()
  solvers = Dict{Symbol,Function}(:Optim => optim_method,
                                  :Optimize => lbfgs)
  pnames = sort(CUTEst.select(max_var=10000, min_var=100, contype=:unc))
  bmark_args = Dict(:atol => 1e-8, :rtol => 0.0, :max_f => 10_000, :max_time => 30.0)

  problems = (CUTEstModel(p) for p in pnames)
  stats, p = bmark_and_profile(solvers, problems, bmark_args=bmark_args)
  png(p, "vs-optim-sum-counters-large")
  stats, p = bmark_and_profile(solvers, problems, bmark_args=bmark_args,
                               cost=stat->stat.elapsed_time)
  png(p, "vs-optim-time-large")
end

runcutest()
