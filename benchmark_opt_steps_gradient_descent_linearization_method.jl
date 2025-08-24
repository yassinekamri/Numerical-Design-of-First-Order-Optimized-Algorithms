# =================================================================================
# Benchmark task Gradient descent — Step-Size Optimization via Linearization method
# ---------------------------------------------------------------------------------
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
# ==================================================================================

# Imports
using JuMP, MosekTools, Mosek
using LinearAlgebra
using ProgressBars
using JLD2

# -------------------------------------------------------------------------------------------------
# Compute xbar, gbarn dbar for inexact gradient descent
# -------------------------------------------------------------------------------------------------
# Populates the vectors xbar, gbar which represent respectively the iterates and
# the associated gradients
function computations_x_g!(xbar, gbar, gamma)
    K = length(gamma)
    dimF = K + 1

    fill!(xbar, 0.0)
    fill!(gbar, 0.0)

    for i in 1:dimF
        xbar[1, i] = 1
        gbar[i + 1, i] = 1
    end
    # iterative updates of inexact gradient descent
    for i in 1:K
        xbar[:, i + 1] .= xbar[:, i] .- gamma[i] .* gbar[:, i]
    end
end


# ---------------------------------------------------------------------------
# Primal PEP formulation
# ---------------------------------------------------------------------------
# Computes the primal formulation of the PEP for inexact gradient descent
function PEP_Primal_Memoryless(gamma, xbar, gbar, fbar)
    K = length(gamma)
    dimG = K + 2
    dimF = K + 1
    nbPts = K + 2
    L = 1 # smoohtness constant
    m = 0 # strong convexity parameter

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-7))
    set_silent(model)

    # Variables:
    # F: functional values f_i = F' * fbar[:,i] 
    # G: Gram matrix such that:
    # |x_k|^2 = xbar[:, k]'  * G * xbar[:, k]
    # |g_k|^2 = gbar[:, k]'  * G * gbar[:, k]
    # g_k * x_j = gbar[:, k]' * G * xbar[:, j]
    @variable(model, G[1:dimG, 1:dimG], PSD)
    @variable(model, F[1:dimF])

    # Initial condition for gradient descent
    @constraint(model, tr(xbar[:, 1] * xbar[:, 1]' * G) == 1)

    # Interpolations conditions for L-smooth mu-strongly convex functions (L =1 and mu = 0 here)
    @constraint(model, [i=1:nbPts, j=1:nbPts; i != j],
        F' * (fbar[:, j] - fbar[:, i]) + tr(0.5 * (        L / (L - m) * (gbar[:, j] * (xbar[:, i] - xbar[:, j])' + (xbar[:, i] - xbar[:, j]) * gbar[:, j]') +
        1 / (L - m) * (gbar[:, i] - gbar[:, j]) * (gbar[:, i] - gbar[:, j])' +
        m / (L - m) * (gbar[:, i] * (xbar[:, j] - xbar[:, i])' + (xbar[:, j] - xbar[:, i]) * gbar[:, i]') +
        L * m / (L - m) * (xbar[:, i] - xbar[:, j]) * (xbar[:, i] - xbar[:, j])') * G) <= 0)

    # the objective: f(x_N) - f(x_*)
    @objective(model, Max, F' * fbar[:, K + 1])

    
    optimize!(model)

    sol_time = solve_time(model)

    objective = objective_value(model)
    G_matrix = value.(G)

    return objective, G_matrix, sol_time
end

# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for inexact gradient descent 
function pep_dual_memoryless(gamma, xbar, gbar, fbar)
    
    N = length(gamma)
    L = 1
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-7))
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", 60.0)
    set_silent(model)

    
    @variable(model, tau >= 0)
    @variable(model, lb[1:N+2, 1:N+2]) # dual variables associated to the interpolation conditions

    # construction of the SDP constraint (mat >= 0)
    mat = @expression(model, tau * xbar[:, 1] * xbar[:, 1]' + sum(0.5 * lb[i, j] * (  gbar[:, j] * (xbar[:, i] - xbar[:, j])' + 
    (xbar[:, i] - xbar[:, j]) * gbar[:, j]' + 
    (1 / L) * (gbar[:, i] - gbar[:, j]) * (gbar[:, i] - gbar[:, j])' ) for i=1:N+2, j=1:N+2 if i != j))

    
    cond1 = @expression(model, fbar[:, N + 1] - sum(lb[i, j] * (fbar[:, j] - fbar[:, i]) for i=1:N+2, j=1:N+2 if i != j))


    @constraint(model, [i=1:N+2, j=1:N+2; i != j], lb[i, j] >= 0)

    
    @constraint(model, mat in PSDCone())
    @constraint(model, cond1 .== 0)

    @objective(model, Min, tau)

   
    optimize!(model)

    objective = objective_value(model)
    dual_vars = value.(lb)
    mat_val = value.(mat)

    sol_time = solve_time(model)

    return objective, dual_vars, mat_val, sol_time
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
    return 0.5*(gj * (xi - xj)' + (xi - xj) * gj' + (1 / L) * (gi - gj) * (gi - gj)')
end

# Computes the derivative of the i-th iterate with respect to the j-th step-size gamma_j
function dxi_dgammaj(N, i, j, gbar)
    dimG = N + 2
    if i <= j || i == dimG
        return zeros(dimG) 
    else
        return -gbar[:, j]
    end
end

# Computes the derivatives of the SDP matrix in the dual PEP w.r.t. to the step sizes gamma
function derivative_gamma!(mat_gamma,N, t, gbar, lb)
     
    fill!(mat_gamma,0.0)
    dxi_cache = [dxi_dgammaj(N, i, t, gbar) for i in 1:N+2]
    
    for i in 1:N+2
        dxi = dxi_cache[i]
        for j in 1:N+2
            if i != j
                gj = gbar[:, j]
                dxj = dxi_cache[j]
                d_diff = dxi - dxj
                mat_gamma .+= 0.5 * lb[i, j] * (gj * d_diff' + d_diff * gj')
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
function linearized_pep(gamma_init, xbar, gbar, fbar, delta, grad_tau, mat_gamma)

    
    computations_x_g!(xbar, gbar, gamma_init)
    obj, dual_vars, mat_val,time_dual = pep_dual_memoryless(gamma_init, xbar, gbar, fbar)

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => 1e-10))
    set_silent(model)

    @variable(model, tau)
    @variable(model, lb[1:K+2, 1:K+2])
    @variable(model, gamma[1:K])
    @variable(model, t)
    @variable(model, s)

    mat = @expression(model, mat_val + tau * grad_tau +  sum(lb[i, j] * derivative_lambda(i, j, xbar, gbar) for i=1:K+2, j=1:K+2) )


    for i in 1:K
    derivative_gamma!(mat_gamma, K, i, gbar, dual_vars)  
    mat += gamma[i] * mat_gamma  
    end

    @constraint(model, obj + tau >= 0)

    @expression(model, AA[i=1:K+2, j=1:K+2], fbar[:, j] - fbar[:, i])
    cond1 = @expression(model, tau * fbar[:, K+2] - sum(lb[i, j] * AA[i, j] for i=1:K+2, j=1:K+2 if i != j))

    @constraint(model, [i=1:K+2, j=1:K+2; i != j], lb[i, j] + dual_vars[i, j] >= 0)


    x = vcat(gamma, vec(lb))
    obj = @expression(model, tau)
    @constraint(model, [s; x] in SecondOrderCone())
    @constraint(model, s <= delta)
    @constraint(model, mat in PSDCone())
    @constraint(model, cond1 .== 0)
    @objective(model, Min, obj)

    
    optimize!(model)
    sol_time = solve_time(model)
    used_time = sol_time + time_dual

    tau_val = value(tau)
    dual_vars = value.(lb)
    gamma_val = value.(gamma)
    return tau_val, dual_vars, gamma_val, used_time
end



# ---------------------------------------------------------------------------
# Full optimization algorithm with trust-region updates
# ---------------------------------------------------------------------------
# Implements the iterative linearization method with a trust-region strategy
# for updating step-sizes gamma.
function inner_iteration(gamma, delta, max_iters=1000, tol=1e-7)
    # Preallocate memory
    start_time = time_ns()
    K = size(gamma,1)
    xbar = zeros(K+2, K+2)
    gbar = zeros(K+2, K+2)
    hess = zeros(K + (K+2)^2, K + (K+2)^2)
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1))  
    grad_tau = zeros(K+2,K+2)
    mat_gamma = zeros(K+2,K+2)
    grad_tau[1,1] = 1
    num_iter = 0
    elapsed_time = 0.0
    for i in tqdm(1:max_iters)
        # Linearized subproblem to compute an update
        tau_val,_, gamma_val,used_time = linearized_pep(gamma,xbar,gbar,fbar,delta,grad_tau,mat_gamma)
        elapsed_time += used_time
    
        computations_x_g!(xbar, gbar, gamma)
        obj, G = PEP_Primal_Memoryless(gamma, xbar, gbar, fbar)

        computations_x_g!(xbar, gbar, gamma + gamma_val)
        obj1, G = PEP_Primal_Memoryless(gamma + gamma_val, xbar, gbar, fbar)

        # stopping criterion
        if abs(obj1 - obj) <= tol && norm(gamma_val) < 1e-4
            break
        end

        # Trust-region update strategy
        ratio = (obj1 - obj) / tau_val
        if ratio > 0.9
            delta *= 2
            gamma .+= gamma_val
        elseif ratio < 0.1
            delta *= 0.5
        else
            gamma .+= gamma_val
        end

        num_iter = i
    end
    computations_x_g!(xbar,gbar,gamma)
    obj, G = PEP_Primal_Memoryless(gamma, xbar, gbar,fbar)
    return obj, gamma, delta, elapsed_time, num_iter
end


#--------------- test -----------------------

K = 13 # Total number of steps

gamma = ones(K) # intial step sizes

delta = 100 # initial trust region diameter

obj, gamma, delta = inner_iteration(gamma,delta)