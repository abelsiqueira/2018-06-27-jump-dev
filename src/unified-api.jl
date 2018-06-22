using CUTEst, NLPModels, JuMP

function newton(nlp :: AbstractNLPModel)
  x = copy(nlp.meta.x0)
  fx = obj(nlp, x)
  gx = zeros(nlp.meta.nvar)
  grad!(nlp, x, gx)
  while norm(gx) > 1e-4
    Hx = Symmetric(hess(nlp, x), :L)
    d = -Hx \ gx
    if dot(d, gx) >= 0.0
      d = -gx
    end
    xt = x + d
    ft = obj(nlp, xt)
    slope = dot(d, gx)
    t = 1.0
    while !(ft < fx + 1e-2 * t * slope)
      t *= 0.25
      xt = x + t * d
      ft = obj(nlp, xt)
    end
    x .= xt
    fx = ft
    grad!(nlp, x, gx)
  end
  return x, fx, gx
end

function runcutest()
  nlp = CUTEstModel("ROSENBR")
  x, fx, gx = newton(nlp)
  println("x = $x")
  println("fx = $fx")
  println("gx = $gx")
end
