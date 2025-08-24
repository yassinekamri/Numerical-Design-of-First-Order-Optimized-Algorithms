# ============================================================================
# Full (using past gradients for updates) Inexact gradient descent — Step-Size Optimization via Linearization method
# objective : f(x_N) - f(x_*)
# ----------------------------------------------------------------------------
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
# ============================================================================

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
# Computes the primal formulation of the PEP for full inexact gradient descent
function pep_primal_inexact(gamma, xbar, gbar, dbar, fbar, epsilon)
    N = size(gamma, 1)
    dimG = N + 2 + N
    dimF = N + 1
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

   # Initial condition for inexact gradient descent
    U = xbar[:, 1]
    @constraint(model, U' * G * U <= 1) 

    # interpolation constraints for smooth convex functions
    @constraint(model, 
        [i = 1:N+2, j = 1:N+2; i != j], 
        F' * (fbar[:, j] - fbar[:, i]) + 
        gbar[:, j]' * G * (xbar[:, i] - xbar[:, j]) + 
        0.5 / L * (gbar[:, i] - gbar[:, j])' * G * (gbar[:, i] - gbar[:, j]) <= 0
    )

    # specific constraints for inexact gradient descent |d_i - g_i| <= epsilon |g_i|
    @constraint(model, 
        [i = 1:N], 
        (dbar[:, i] - gbar[:, i])' * G * (dbar[:, i] - gbar[:, i]) - 
        epsilon^2 * gbar[:, i]' * G * gbar[:, i] <= 0
    )

    # Define the objective max f(x_N) - f(x_*)
    @objective(model, Max, F' * fbar[:, N + 1])

    # Solve the optimization problem
    optimize!(model)

    # Retrieve results
    objective = objective_value(model)
    G_val = value.(G)

    return objective, G_val
end

# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for full inexact gradient descent 
function pep_dual_inexact(gamma, xbar, gbar, dbar, fbar, epsilon)
    N = size(gamma, 1)
    L = 1

    # Initialize JuMP model
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Declare variables
    @variable(model, tau >= 0)
    @variable(model, lb[1:N+2, 1:N+2] >= 0)  # dual variables associated to the interpolation conditions
    @variable(model, mu[1:N] >= 0)           # dual variables associated to the inequalities |d_i - g_i| <= epsilon |g_i|


    # construction of the SDP constraint (mat >= 0)
    mat = @expression(model, 
    tau * (xbar[:,1] * xbar[:,1]') + sum(
             0.5 * lb[i, j] * (
                gbar[:, j] * (xbar[:, i] - xbar[:, j])' +
                (xbar[:, i] - xbar[:, j]) * gbar[:, j]' +
                (1 / L) * (gbar[:, i] - gbar[:, j]) * (gbar[:, i] - gbar[:, j])'
            )
            for i in 1:N+2, j in 1:N+2 if i != j
        )
    )

    cond = @expression(model, 
    fbar[:, N + 1] -sum(
            lb[i, j] * (fbar[:, j] - fbar[:, i])
            for i in 1:N+2, j in 1:N+2 if i != j
        )
    )

    
    mat_mu = @expression(model, 
        sum(
            mu[i] * (
                (dbar[:, i] - gbar[:, i]) * (dbar[:, i] - gbar[:, i])' -
                epsilon^2 * (gbar[:, i] * gbar[:, i]')
            )
            for i in 1:N
        )
    )
    
    mat = mat + mat_mu
    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)

    # Objective
    @objective(model, Min, tau)

    # Solve the optimization problem
    optimize!(model)

    # Retrieve results
    objective = objective_value(model)
    lb_val = value.(lb)
    mu_val = value.(mu)
    mat_val = value.(mat)

    return objective, lb_val, mu_val, mat_val
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
    return 0.5 *( gj * (xi - xj)' + (xi - xj) * gj' + (1 / L) * (gi - gj) * (gi - gj)')
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
# Computes the derivatives of the SDP matrices in the dual PEP w.r.t. to the step sizes gamma
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
#   https://arxiv.org/abs/2507.20773
function linearized_pep_inex(K,gamma_init, xbar, gbar, dbar, fbar, epsilon, delta, grad_tau, mat_gamma)

    
    computations_x_g_d!(K,gamma_init,xbar,gbar,dbar)
    obj, lb_val, mu_val, mat_val = pep_dual_inexact(gamma_init, xbar, gbar, dbar, fbar, epsilon)
   

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-10))
    set_silent(model)

    @variable(model, tau)
    @variable(model, lb[1:K+2, 1:K+2])
    @variable(model, mu[1:K])
    @variable(model, gamma[1:K, 1:K])
    @variable(model, t)
    @variable(model, s)

    
    mat = @expression(model, mat_val + tau * grad_tau +  sum(lb[i, j] * derivative_lambda(i,j,xbar,gbar) for i=1:K+2, j=1:K+2) )
    mat_mu = @expression(model,sum(mu[i] * derivative_mu(i,dbar,gbar,epsilon) for i = 1:K) )
    mat = mat + mat_mu

    for i in 1:K
        for j in 1:K
            derivative_gamma!(K,i,j,dbar,gbar,lb_val,mat_gamma)
            mat .+= gamma[i,j] * mat_gamma 
        end 
    end

    @constraint(model, obj + tau >= 0)

    @expression(model, AA[i=1:K+2, j=1:K+2], fbar[:, j] - fbar[:, i])
    cond1 = @expression(model, tau * fbar[:, K+2] - sum(lb[i, j] * AA[i, j] for i=1:K+2, j=1:K+2 if i != j))

    @constraint(model, [i=1:K+2, j=1:K+2; i != j], lb[i, j] + lb_val[i, j] >= 0)
    @constraint(model, [i=1:K], mu[i] + mu_val[i] >= 0)

    x = vcat(vec(gamma), vec(lb))
    obj = @expression(model, tau)

    @constraint(model, [s; x] in SecondOrderCone())
    @constraint(model, s <= delta)
    @constraint(model, mat in PSDCone())
    @constraint(model, cond1 .== 0)
    @objective(model, Min, obj)

    optimize!(model)

    tau_val = value(tau)
    dual_vars = value.(lb)
    mu_vars = value.(mu)
    gamma_val = value.(gamma)
    return tau_val, dual_vars, mu_vars, gamma_val
end



function inner_iteration(gamma, delta, epsilon, max_iters=1000, tol=1e-4)
    # Preallocate memory
    K = size(gamma,1)
    xbar = zeros(2*K+2, K+2)
    gbar = zeros(2*K+2, K+2)
    dbar = zeros(2*K+2,K)
    hess = zeros(K^2 + (K+2)^2, K^2 + (K+2)^2)
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1))  
    grad_tau = zeros(2*K+2,2*K+2)
    mat_gamma = zeros(2*K+2,2*K+2)
    grad_tau[1,1] = 1
    for i in tqdm(1:max_iters)
        tau_val, _, _, gamma_val = linearized_pep_inex(K,gamma, xbar, gbar, dbar, fbar, epsilon, delta, grad_tau, mat_gamma)

        # Stopping criterion
        if norm(gamma_val) <= tol
            break
        end

        computations_x_g_d!(K,gamma,xbar,gbar,dbar)
        obj, G = pep_primal_inexact(gamma, xbar, gbar, dbar, fbar, epsilon)

        computations_x_g_d!(K,gamma + gamma_val,xbar,gbar,dbar)
        obj1, G = pep_primal_inexact(gamma, xbar, gbar, dbar, fbar, epsilon)
        ratio = (obj1 - obj) / tau_val

        # Trust-region update strategy
        if (obj1 - obj) >= 0
            delta *= 0.5
        elseif (obj1 - obj) < 0 && ratio > 0.9
            delta *= 2
            gamma .+= gamma_val
        elseif (obj1 - obj) < 0 && ratio < 0.1
            delta *= 0.5
        else
            gamma .+= gamma_val
        end


    end
    computations_x_g_d!(K,gamma,xbar,gbar,dbar)
    obj, G = pep_primal_inexact(gamma, xbar, gbar, dbar, fbar, epsilon)
    return obj, gamma, delta
end



#--------------- test -----------------------

K = 9 # Total number of steps

gamma = ones(K,K) # intial step sizes

epsilon = 0.1 # inexactness parameter

delta = 100 # initial trust region diameter

obj, gamma, delta = inner_iteration(gamma,delta,epsilon)