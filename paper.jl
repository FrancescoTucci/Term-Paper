type LakeModel3 #setting out a scaffold with a given type for each object
    lambda :: Float64
    alpha :: Float64
    g :: Float64
    pi :: Float64
    delta :: Float64
    epsilon1 :: Float64

    A :: Matrix{Float64}
    A_hat :: Matrix{Float64}
end

function LakeModel3(;lambda=0.2, alpha=0.05, g=0.005, pi=0.70,delta=0.25,
  epsilon1=0.10)


    A = [ (1-lambda)*(1-epsilon1*pi) alpha delta;
        lambda (1-alpha) 0;
        (1-lambda)*epsilon1*pi 0 (1-delta)]
    A_hat = A .* (1+g)  #transition matrices for employed, unemployed,prisoners rates and stocks


    return LakeModel3(lambda, alpha,g, pi, delta, epsilon1, A, A_hat)
end
lm=LakeModel3()
N_0 = 300      # Population
e_0 = 0.89   # Initial employment rate
u_0 = 0.09 # Initial unemployment rate
p_0 =0.02 # Initial prisoners rate
T = 50 #Number of periods
E_0 = e_0 * N_0 #Corresponding stocks
U_0 = u_0 * N_0
P_0 = p_0 * N_0
X_0 = [U_0; E_0; P_0] #initial vector of stocks
x_0 = [u_0; e_0;p_0] #initial vector of rates



function rate_steady_state(lm::LakeModel3,x=x_0) #iterations over the transitions
  #to find the steady state rate for employ. and unemploy. and prison population

    tol= 1e-6
    error = tol+1
    while (error > tol)
        new_x = lm.A * x
        error = maxabs(new_x - x)
        x = new_x
    end
    return x
end
s=rate_steady_state(lm)






function simulate_stock_path(lm::LakeModel3, X0::Vector{Float64}, T::Int) #returns the sequence of the values
  #for employmed,unemployed and prisoners stocks (N.B. not the rates!!) from initial values for the three, stored in X0
    X_path = Array(Float64, 3, T)
    X = copy(X0)
    for t in 1:T
        X_path[:, t] = X
        X = lm.A_hat * X
    end
    return X_path
end



function simulate_rate_path(lm::LakeModel3, x0::Vector{Float64}, T::Int)#same concept for the respective rates
    x_path = Array(Float64, 3, T)
    x = copy(x0)
    for t in 1:T
        x_path[:, t] = x
        x = lm.A * x
    end
    return x_path
end
using Plots
pyplot()

X_path = simulate_stock_path(lm, X_0, T)

titles = ["Unemployment" "Employment" "Prisoners" "Population"]
dates = collect(1:T)

x1 =X_path[1, :]
x2 =  X_path[2, :]
x3 =X_path[3,:]
x4= x1 .+ x2 .+ x3

plot(dates, Vector[x1, x2, x3, x4], title=titles, layout=(4, 1), legend=:none) #Plotting the dynamic of aggregate stocks

x_path = simulate_rate_path(lm, x_0, T)

titles = ["Unemployment rate" "Employment rate" "Prison Population"]
dates = collect(1:T)

plot(dates, x_path', layout=(3, 1),
title=titles, legend=:none) #Plotting the dynamic of the rates
hline!(s', layout=(3, 1), color=:red, linestyle=:dash) #Showing convergence towards the steady state

using QuantEcon
using StatsBase

srand(42)
lm1 = LakeModel3(g=0)

alpha, lambda, delta, pi, epsilon1 = lm1.alpha, lm1.lambda, lm1.delta,lm1.pi,lm1.epsilon1
       P = [(1 - lambda)*(1-epsilon1*pi) lambda (1 - lambda)*(epsilon1*pi);
           alpha (1 - alpha) 0;
           delta 0 (1-delta)] #transition matrix for the outcomes faced by each individual

       mc = MarkovChain(P, [1; 2; 3]) #using the command from QuantEcon
                                      # 1=unemployed, 2=employed, 3= prison
       xbar = rate_steady_state(lm) #taking the steady states from up above
function iterations(h::Int64) #function to show that the time spent in each state for the individual
                             # to the variables' steady states the more we consider a longer interval of time
 mc = MarkovChain(P, [1; 2; 3])
  T=collect(1000:20000)
v=Array{Float64}(19001,3)
b=zeros(3)

   for (i, t) in enumerate(T)
    o = simulate(mc, t ; init=h)
    l=countmap(o)
    for e in 1:3
    b[e] = l[e]
    end
    v[i,:]=b./t

   end
return v
 end
 mod=iterations(1) #starting unemployed
 r1=mod[:,1]
 r2=mod[:,2]
 r3=mod[:,3]
 using LaTeXStrings
 plot(T, [r1,r2,r3],lw=2,label=[L"$U$" L"$E$" L"$P$"], title="State proba. if starting unemployed" )
 hline!((xbar)', color=[:red :black :purple], label=[L"$ss_U$" L"$ss_E$" L"$ss_P$"],linestyle=:dash)
 mod1=iterations(2) #starting employed
 n1=mod1[:,1]
 n2=mod1[:,2]
 n3=mod1[:,3]
 plot(T, [n1,n2,n3],lw=2,label=[L"$U$" L"$E$" L"$P$"], title="State proba. if starting employed" )
 hline!((xbar)', color=[:red :black :purple], label=[L"$ss_U$" L"$ss_E$" L"$ss_P$"],linestyle=:dash)
mod2=iterations(3) #starting in prison
m1=mod2[:,1]
m2=mod2[:,2]
m3=mod2[:,3]
plot(T, [m1,m2,m3],lw=2,label=[L"$U$" L"$E$" L"$P$"], title="State proba. if starting in prison" )
hline!((xbar)', color=[:red :black :purple], label=[L"$ss_U$" L"$ss_E$" L"$ss_P$"],linestyle=:dash)

#evolution of the steady states with respect to some policy changes

function enforcement() #proba of enforcement pi
  n=50
  pi=linspace(0.2,0.8,n)
  pi_vec=collect(linspace(0.2,0.8,n))
  ss=Array{Float64}(50,3)

  for i in 1:50
    ss[i,:]=rate_steady_state(LakeModel3(pi=pi_vec[i]))
  end
  return ss
end
enforcement()
ss_u=enforcement()[:,1]
ss_e=enforcement()[:,2]
ss_p=enforcement()[:,3]

plot(pi_vec,[ss_u,ss_e,ss_p],lw=2, ylims=(0.0,1.0), label=[L"$SS_u$" L"$SS_e$" L"$SS_p$"], #Plotting how the steady states change according to different level of enforcement
title= "Steady states - enforcement rate")
function crime_rate() #proba. of committing a crime
  n=50
  eps_vec=collect(linspace(0.05,0.5,n))
  ss=Array{Float64}(50,3)

  for i in 1:50
    ss[i,:]=rate_steady_state(LakeModel3(epsilon1=eps_vec[i]))
  end
  return ss
end

crime_rate()
ss_u=crime_rate()[:,1]
ss_e=crime_rate()[:,2]
ss_p=crime_rate()[:,3]

plot(eps_vec,[ss_u,ss_e,ss_p],lw=2, ylims=(0.0,1.0), label=[L"$SS_u$" L"$SS_e$" L"$SS_p$"], #Plotting how the steady state evolves with the probability of committing a crime
title= "Steady states - crime rate")
function release() #proba. of being released from prison
  n=50
  delta_vec=collect(linspace(0.05,0.7,n))
  ss=Array{Float64}(50,3)

  for i in 1:50
    ss[i,:]=rate_steady_state(LakeModel3(delta=delta_vec[i]))
  end
  return ss
end
release()
ss_u=release()[:,1]
ss_e=release()[:,2]
ss_p=release()[:,3]

plot(eps_vec,[ss_u,ss_e,ss_p],lw=2, ylims=(0.0,1.0), label=[L"$SS_u$" L"$SS_e$" L"$SS_p$"], #Plotting how the steady state evolves with different levels of release rate
title= "Steady states - release rate")

function jobfinding() #proba. of finding a job
  n=50
  lambda_vec=collect(linspace(0.05,0.35,n))
  ss=Array{Float64}(50,3)

  for i in 1:50
    ss[i,:]=rate_steady_state(LakeModel3(lambda=lambda_vec[i]))
  end
  return ss
end
jobfinding()
ss_u=jobfinding()[:,1]
ss_e=jobfinding()[:,2]
ss_p=jobfinding()[:,3]

plot(lambda_vec,[ss_u,ss_e,ss_p],lw=2, ylims=(0.0,1.0), label=[L"$SS_u$" L"$SS_e$" L"$SS_p$"], #Plotting how the steady state evolves with different levels of release rate
title= "Steady states - job finding rate")

function breakingshock() #proba. of losing a job
  n=50
  alpha_vec=collect(linspace(0.01,0.20,n))
  ss=Array{Float64}(50,3)

  for i in 1:50
    ss[i,:]=rate_steady_state(LakeModel3(alpha=alpha_vec[i]))
  end
  return ss
end
breakingshock()
ss_u=breakingshock()[:,1]
ss_e=breakingshock()[:,2]
ss_p=breakingshock()[:,3]

plot(alpha_vec,[ss_u,ss_e,ss_p],lw=2, ylims=(0.0,1.0), label=[L"$SS_u$" L"$SS_e$" L"$SS_p$"], #Plotting how the steady state evolves with different levels of release rate
title= "Steady states - job loss rate")
