using Plots, Krylov
pyplot(size=(600,600))

function foo()
  B = [2 1; 1 8] * 0.15
  g = - B * ones(2)
  t = linspace(0, 1, 100)

  σ = 1.05
  q(x,y) = dot([x;y], 0.5 * B * [x;y] + g)
  for (i,Δ) = enumerate([0.5; 1.2; 1.5])
    contour(2 * t - 0.5, 2 * t - 0.5, q, aspect_ratio=:equal, leg=false)
    dC = -g * (dot(g, g) / dot(g, B * g))
    plot!([0; dC[1]], [0; dC[2]], c=:red, l=:arrow, ann=(dC[1] * σ,dC[2] * σ,"Unc. Cauchy"))
    dN = -(B \ g)
    plot!([0; dN[1]], [0; dN[2]], c=:red, l=:arrow, ann=(dN[1] * σ,dN[2] * σ,"Newton"))
    plot!([dC[1]; dN[1]], [dC[2]; dN[2]], c=:red, l=:dash, ann=(dN[1] * σ,dN[2] * σ,"Newton"))
    x = cg(B, -g, radius=Δ)[1]
    scatter!([x[1]], [x[2]], ms=5, c=:black)
    plot!(Δ * cos.(2π * t), Δ * sin.(2π * t), c=:green, l=:dash, lw=2)
    xlims!(-0.5, 1.5); ylims!(-0.5, 1.5)
    png("krylov$i")
  end
end

foo()
