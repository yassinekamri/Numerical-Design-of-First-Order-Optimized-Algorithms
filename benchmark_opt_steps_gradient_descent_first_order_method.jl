# =================================================================================
# Benchmark task Gradient descent — Step-Size Optimization via Fist-order Method
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
# Compute xbar, gbarn dbar for gradient descent
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
    # iterative updates of gradient descent
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
# Computation of the gradient of the worst-case given by the PEP w.r.t the step-sizes gamma
# ---------------------------------------------------------------------------------------------
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

# compute the gradient of the worst-case given by the PEP w.r.t the steps-sizes gamma
# see paper https://arxiv.org/abs/2507.20773 for more details on these computations
function gradient_step!(grad_vec,mat_gamma,K,gbar,dual_vars,G)
    fill!(grad_vec,0.0)
    for j in 1:K
        derivative_gamma!(mat_gamma, K, j, gbar, dual_vars)
        grad_vec[j] = tr(-G*mat_gamma)
    end
end

# -----------------------------------------------------------------------------
# Full optimization algorithm with Polyak step-sizes
# -----------------------------------------------------------------------------
# Implements the iterative first-order method with decreasing Polyak step-sizes
# for updating step-sizes gamma

function inner_iteration(K, iter= 1000)

    gamma = ones(K) + 0.1 * randn(K)
    xbar = zeros(K+2, K+2)
    gbar = zeros(K+2, K+2)
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1))  
    mat_gamma = zeros(K+2,K+2)
    grad_vec = zeros(K)
    elapsed_time = 0.0

    function_values = Float64[]
    step_sizes = [gamma]

    for i in tqdm(1:iter)
        computations_x_g!(xbar, gbar, step_sizes[i])
        fi, G, used_time = PEP_Primal_Memoryless(step_sizes[i], xbar, gbar, fbar)
        objective, dual_vars, _, used_time1 = pep_dual_memoryless(step_sizes[i], xbar, gbar, fbar)
        
        elapsed_time += (used_time + used_time1)

        push!(function_values, fi)
        

        gradient_step!(grad_vec, mat_gamma, K, gbar, dual_vars, G)

        fstar = minimum(function_values)
        norm_grad = norm(grad_vec)
        h = abs(fi - fstar + 1/i)/norm_grad
        push!(step_sizes, step_sizes[i] - h * grad_vec)

    end
    min_idx = argmin(function_values)
    min_function_value = function_values[min_idx]
    best_step_size = step_sizes[min_idx]
    return min_function_value, best_step_size, elapsed_time
end


#--------------- test ---------------------------------------

K = 4 # total number of iteration for gradient descent/number of step-sizes to optimize
min_function_value, best_step_size, elapsed_time = inner_iteration(K)