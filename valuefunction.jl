using Distributions






type McCallModel1
    alpha::Float64        # Job separation rate
    beta::Float64         # Discount rate
    pi::Float64           # Enforcement probability
    delta::Float64         #Probability of being released from prison
    gamma::Float64        #Proba of not facing either an offer for a job or an opportunity to commit a crime, see QuantEcon
    b::Float64        #unemployment benefits
    x::Float64            #Prisoners' benefit
    c_vec::Vector{Float64} #Possible crime values
    w_vec::Vector{Float64} # Possible wage values
    pw_vec::Vector{Float64} # Probabilities over w_vec
    pc_vec:: Vector{Float64} #Proba. over c_vec


    const n = 60                                   # n possible outcomes for wage
    const default_w_vec = linspace(1, 5, n)   # wages between 1 and 5
    const a, b = 600, 400                          # shape parameters
    const dist = BetaBinomial(n-1, a, b)     #distribution of wages, taken from QuantEcon
    const default_pw_vec = pdf(dist)    #pdf over the distribution above
    const default_c_vec = linspace(0, 6, n)   #crime values between 0 and 6
    const dist1 = BetaBinomial(n-1, 200, 800)  #same type of distribution but more skewed on the left
    const default_pc_vec = pdf(dist1) #pdf over previous distribution


function McCallModel(alpha=0.05,beta=0.90, gamma=0.25,delta= 0.25,x=0.01,pi= 0.7, b=0.5,
   w_vec=default_w_vec, pw_vec=default_pw_vec, c_vec=default_c_vec, pc_vec=default_pc_vec)

        return new(alpha, beta, pi , delta,gamma, b, x , c_vec, w_vec, pw_vec, pc_vec)

end
end
mcm=McCallModel() #returns the benchmark model

function update_bellman!(mcm, V::Vector{Float64}, V_new::Vector{Float64}, #function  to update the bellman in every step
  U::Vector{Float64},U_new::Vector{Float64}, P::Float64)
 alpha, beta, pi, x ,delta,b,gamma= mcm.alpha, mcm.beta,mcm.pi, mcm.x, mcm.delta, mcm.b,mcm.gamma

for (w_idx, w) in enumerate(mcm.w_vec)# w_idx indexes the vector of possible wages
        V_new[w_idx] = w + beta * (1 - alpha) * V[w_idx] + beta * alpha * sum( U .* mcm.pc_vec)
        end
for (c_idx, c) in enumerate(mcm.c_vec)#same indexing for crimes
        U_new[c_idx] = b + c + beta  * (1-gamma) * (sum(max(U[c_idx], V) .* mcm.pw_vec) +
        sum(max(U[c_idx], (1-pi) * U + pi * P) .* mcm.pc_vec)) + beta * gamma * U[c_idx]
        end

       P_new = x + beta * delta * sum( U .* mcm.pc_vec) + beta * (1-delta) * P

 return  V_new, U_new,P_new
end

function VFI() #value function iteration starting from initial values
tol=1e-5
V=ones(n)
V_new=similar(V)
U=ones(n)
U_new=ones(n)
P=1.0
for iter in 1:20000
  V_new=update_bellman!(mcm, V, V_new , U,  U_new ,P)[1]
  U_new=update_bellman!(mcm, V, V_new , U,  U_new ,P)[2]
  P_new = update_bellman!(mcm, V, V_new , U, U_new ,P)[3]

error1=maximum(abs(V - V_new))
error2=maximum(abs(U - U_new))
error3=abs(P - P_new)
error= max(error1,error2,error3)
# check convergence
if error < tol #&& maxabs(U.-U_new) < tol && maxabs(P.-P_new)<tol

println("Found solution after $iter iterations")

return V_new, U_new, P_new
elseif iter==20000
warn("No solution found after $iter iterations")
return V_new, U_new, P_new
end
V= V_new
U=U_new
P=P_new # update guess
end
end

V, U ,P= VFI() #No convergence!!
