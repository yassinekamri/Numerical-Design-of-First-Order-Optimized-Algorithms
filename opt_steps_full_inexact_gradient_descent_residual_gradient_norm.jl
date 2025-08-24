# ========================================================================================================================
# Full (use of past gradients for updates) Inexact gradient descent — Step-Size Optimization via Linearization method
# objective : min_{i in 1:N} |∇f(x_i)|^2
# -------------------------------------------------------------------------------------------------------------------------
# Julia code to optimize the step-sizes of inexact gradient descent using the
# linearization method described in:
#   Y. Kamri, J. M. Hendrickx, and F. Glineur.
#   "Numerical Design of Optimized First-Order Algorithms." arXiv, 2025.
#   Link: https://arxiv.org/abs/2507.20773
#
#
# Dependencies:
#   - JuMP, MosekTools, Mosek
#   - LinearAlgebra, SparseArrays
#   - ProgressBars (optional)
#   - JLD2 (optional)
# ==========================================================================================================================

# Imports
using JuMP, MosekTools, Mosek
using LinearAlgebra
using ProgressBars
using JLD2

# -------------------------------------------------------------------------------------------------
# Compute xbar, gbarn dbar for inexact gradient descent
# -------------------------------------------------------------------------------------------------
# Populates the vectors xbar, gbar and dbar which represent respectively iterates,
# the associated gradients and the approximations for each gradients: |d_i - g_i| <= epsilon |g_i|.
function computations_x_g_d!(N,gamma,xbar,gbar,dbar)


    fill!(xbar,0.0)
    fill!(gbar,0.0)
    fill!(dbar,0.0)

    dimG = N+ 2 + N
    dimF = N+ 1


    for i = 1:dimF
        gbar[i + 1, i] = 1
    end
    
    for i = 1:N
        dbar[N+2+i, i] = 1
    end
    # iterative updates of full inexact gradient descent
    for i in 1:N+2
        U = zeros(dimG, 1)
        U[1, 1] = 1
        if i == 1
            x = vec(U)
        elseif i == N+2
            x = vec(zeros(dimG, 1))
        else
            iter = U
            for j = 1:i-1
                iter = iter .- gamma[i-1,j]*dbar[:,j]
            end
            x = vec(iter)
        end
        xbar[:,i] = x
    end
end

# ---------------------------------------------------------------------------
# Primal PEP formulation
# ---------------------------------------------------------------------------
# Computes the primal formulation of the PEP for inexact gradient descent
function pep_primal_inexact_norm_grad(gamma,xbar,gbar,dbar,fbar,epsilon)
    N = size(gamma,1)
    dimG = N+2 + N
    dimF = N+1
    L = 1
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrix such that:
    # |x_k|^2 = xbar[:, k]'  * G * xbar[:, k]
    # |g_k|^2 = gbar[:, k]'  * G * gbar[:, k]
    # g_k * x_j = gbar[:, k]' * G * xbar[:, j]
    @variable(model, G[1:dimG, 1:dimG], PSD)
    @variable(model, F[1:dimF])
    @variable(model,t)

     # Initial condition for inexact gradient descent
    @constraint(model, F'*(fbar[:,1] - fbar[:,N+2] ) <= 1)

    # interpolation constraints for smooth convex functions
    for i = 1:N + 2
        for j = 1:N + 2
            if i != j
                xi = xbar[:,i]
                xj = xbar[:,j]
                gi = gbar[:, i]
                gj = gbar[:, j]
                fi = fbar[:, i]
                fj = fbar[:, j] 
                cond = F'*(fj-fi) + gj'*G*(xi-xj) + 0.5/L * (gi-gj)'*G*(gi-gj)
                @constraint(model, cond <= 0)
            end
        end
    end

    # specific constraints for inexact gradient descent |d_i - g_i| <= epsilon |g_i|
    for i = 1:N
        di = dbar[:,i]
        gi = gbar[:,i]
        cond = (di-gi)'*G*(di-gi) - epsilon^2*gi'*G*gi
        @constraint(model, cond <= 0)
    end

     # objective: min_{i in 1:N} |∇f(x_i)|^2
    @objective(model, Max, t)
    for i in 1:N+1
        @constraint(model, t <= gbar[:,i]'*G*gbar[:,i])
    end

    optimize!(model)

    objective = objective_value(model)
    G_val = value.(G)

    return objective, G_val
end

# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for inexact gradient descent 
function pep_dual_inexact_norm_grad(gamma,xbar,gbar,dbar,fbar,epsilon)


    N = size(gamma,1)
    dimG = N+2 + N
    dimF = N+1

    L = 1
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, tau >= 0)
    @variable(model, lb[1:N+2, 1:N+2]) # dual variables associated to the interpolation conditions
    @variable(model, mu[1:N]) # dual variables associated to the inequalities |d_i - g_i| <= epsilon |g_i|
    @variable(model, beta[1:N+1]) # dual variables associated to constraints enforcing min_{i in 1:N} |∇f(x_i)|^2 as the objective

    # construction of the SDP constraint (mat >= 0)
    mat = -gbar[:,N+2]*gbar[:,N+2]'
    cond = -tau*(fbar[:, 1] - fbar[:,N+2])
    for i = 1:N + 2
        for j = 1:N + 2
            if i != j
                xi = xbar[:,i]
                xj = xbar[:,j]
                gi = gbar[:, i]
                gj = gbar[:, j]
                fi = fbar[:, i]
                fj = fbar[:, j]
                A = gj * (xi - xj)' + (xi - xj) * gj' + (1 / L) * (gi - gj) * (gi - gj)'
                AA = (fj - fi)
                mat += 0.5 * lb[i, j] * A
                cond -= lb[i, j] * AA
                @constraint(model, lb[i, j] >= 0)
            end
        end
    end
   
    cond1 = 0
    for i in 1:N+1
        mat -= beta[i]* gbar[:,i] * gbar[:,i]'
        cond1 += beta[i]
        @constraint(model, beta[i] >= 0)
    end
    @constraint(model, cond1 .== 1)

    for i = 1:N
        di = dbar[:,i]
        gi = gbar[:,i]
        A = (di-gi)*(di-gi)' - epsilon^2*(gi*gi')
        mat += mu[i]*A
        @constraint(model, mu[i] >= 0)
    end
    
    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)
    @objective(model, Min, tau)

    optimize!(model)

    objective = objective_value(model)
    lb_val = value.(lb)
    mu_val = value.(mu)
    mat_val = value.(mat)
    beta_val = value.(beta)
    return objective, lb_val,mu_val,mat_val,beta_val
end

# ---------------------------------------------------------------------------------------------
# Derivatives of the SDP matrices of the dual PEP with respect to dual variables and step-sizes
# ---------------------------------------------------------------------------------------------
# Computes the derivatives of the SDP matrix in the dual PEP w.r.t. dual variables lambda
function derivative_lambda(i,j,xbar,gbar)
    L = 1
    xi = xbar[:, i]
    xj = xbar[:, j]
    gi = gbar[:, i]
    gj = gbar[:, j]
    return 0.5 * ( gj * (xi - xj)' + (xi - xj) * gj' + (1 / L) * (gi - gj) * (gi - gj)'  )
end
# Computes the derivatives of the SDP matrix in the dual PEP w.r.t. dual variables mu
function derivative_mu(i,dbar,gbar,epsilon)
    di = dbar[:,i]
    gi = gbar[:,i]
    return (di-gi)*(di-gi)' - epsilon^2*(gi*gi')
end

# Computes the derivative of the i-th iterate with respect to the step-size gamma_{j,k}
function dxi_dgammajk(N,i,j,k,dbar)
    dimG =  N + 2 + N
    if j == i-1 && k <= i-1
        return - dbar[:, k]
    else
        return zeros(dimG,1)
    end
end
# Computes the derivatives of the SDP matrix in the dual PEP w.r.t. to the step sizes gamma
function derivative_gamma!(N,t,k,dbar,gbar,lb,mat_gamma)
    fill!(mat_gamma, 0.0)
    dx_dgamma_cache = [dxi_dgammajk(N,i,t,k,dbar) for i in 1:N+2]
    for i in 1:(N + 2)
        for j in 1:(N + 2)
            if i != j
                gi = gbar[:, i]
                gj = gbar[:, j]
                dxi = dx_dgamma_cache[i]
                dxj = dx_dgamma_cache[j]
                mat_gamma .+= 0.5 * lb[i,j] * ( gj * (dxi - dxj)' + (dxi - dxj) * gj' )
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Linearized subproblem for step-size optimization
# ---------------------------------------------------------------------------
# Constructs and solves the linearized PEP subproblem to compute an update
# for the step-sizes gamma. For details, see:
# https://arxiv.org/abs/2507.20773

function linearized_pep_inex(K,gamma_init, xbar, gbar, dbar, fbar, epsilon, delta, mat_gamma)

    computations_x_g_d!(K,gamma_init,xbar,gbar,dbar)
    obj, lb_val, mu_val, mat_val, beta_val = pep_dual_inexact_norm_grad(gamma_init, xbar, gbar, dbar, fbar, epsilon)
   

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-10))
    set_silent(model)

    
    @variable(model, tau)
    @variable(model, lb[1:K+2, 1:K+2])
    @variable(model, mu[1:K])
    @variable(model, gamma[1:K,1:K])
    @variable(model, beta[1:K+1])
    @variable(model, s)

    mat = mat_val
   
    cond = -tau*(fbar[:, 1] - fbar[:,K+2])
    @constraint(model, tau + obj >= 0)
    for i in 1:K+2
        for j in 1:K+2
            if i != j
                cond -= lb[i,j] * ( fbar[:,j] - fbar[:,i] )
                mat += lb[i,j]*derivative_lambda(i,j,xbar,gbar)
                @constraint(model, lb[i,j] + lb_val[i,j] >= 0)

            end
        end
    end

    for i in 1:K
        for j in 1:K
            derivative_gamma!(K,i,j,dbar,gbar,lb_val,mat_gamma)
            mat .+= gamma[i,j] * mat_gamma 
        end 
    end

    for i in 1:K 
        mat += mu[i] * derivative_mu(i,dbar,gbar,epsilon)
        @constraint(model, mu[i] + mu_val[i] >= 0)
    end
    cond1 = 0
    for i in 1:K+1
        mat -= beta[i]* gbar[:,i] * gbar[:,i]'
        cond1 += beta[i]
        @constraint(model, beta[i] + beta_val[i] >= 0)
    end
    @constraint(model, cond1 .== 0)

    x = vcat(vec(gamma), vec(lb))
    @constraint(model, [s; x] in SecondOrderCone())
    @constraint(model, s <= delta)
    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)
    @objective(model, Min, tau)

    optimize!(model)

    tau_val = value(tau)
    dual_vars = value.(lb)
    mu_vars = value.(mu)
    gamma_val = value.(gamma)
    return tau_val, dual_vars, mu_vars, gamma_val
end

# ---------------------------------------------------------------------------
# Full optimization algorithm with trust-region updates
# ---------------------------------------------------------------------------
# Implements the iterative linearization method with a trust-region strategy
# for updating step-sizes gamma.
function inner_iteration(gamma, delta, epsilon, max_iters=1000, tol=1e-4)
    # Preallocate memory
    K = size(gamma,1)
    xbar = zeros(2*K+2, K+2)
    gbar = zeros(2*K+2, K+2)
    dbar = zeros(2*K+2,K)
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1))  
    mat_gamma = zeros(2*K+2,2*K+2)
   
    
    for i in tqdm(1:max_iters)
        # Linearized subproblem to compute an update
        tau_val,_,_, gamma_val = linearized_pep_inex(K,gamma, xbar, gbar, dbar, fbar, epsilon, delta, mat_gamma)

        # Stopping criterion
        if norm(gamma_val) <= tol
            break
        end
    
       
        computations_x_g_d!(K,gamma,xbar,gbar,dbar)
        obj, G = pep_dual_inexact_norm_grad(gamma, xbar, gbar, dbar, fbar, epsilon)

        computations_x_g_d!(K,gamma + gamma_val,xbar,gbar,dbar)
        obj1, G = pep_dual_inexact_norm_grad(gamma, xbar, gbar, dbar, fbar, epsilon)

        # Trust-region update strategy
        ratio = (obj1 - obj) / tau_val
        if (obj1 - obj) >= 0
            delta *= 0.5
        elseif (obj1 - obj) < 0 && ratio > 0.9
            delta *= 2
            gamma .+= gamma_val
        elseif (obj1 - obj) < 0 && ratio < 0.1
            delta *= 0.5
        elseif (obj1 - obj) < 0
            gamma .+= gamma_val
        end 
    end
    computations_x_g_d!(K,gamma,xbar,gbar,dbar)
    obj, G = pep_dual_inexact_norm_grad(gamma, xbar, gbar, dbar, fbar, epsilon)
    return obj, gamma, delta
end

#--------------- test -----------------------

K = 8 # Total number of steps

gamma = ones(K,K) # intial step sizes

epsilon = 0.3 # inexactness parameter

delta = 100 # initial trust region diameter

obj, gamma, delta = inner_iteration(gamma,delta,epsilon)