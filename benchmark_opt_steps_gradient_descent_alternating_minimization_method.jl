
# =============================================================================================
# Benchmark task Gradient descent — Step-Size Optimization via ALternating Minimization method
# ---------------------------------------------------------------------------------------------
# Julia code to optimize the step-sizes of inexact gradient descent using the
# linearization method described in:
#   Y. Kamri, J. M. Hendrickx, and F. Glineur.
#   "Numerical Design of Optimized First-Order Algorithms." arXiv, 2025.
#   Link: https://arxiv.org/abs/2507.20773
#
#
# Dependencies:
#   - JuMP, MosekTools, Mosek
#   - LinearAlgebra
#   - ProgressBars (optional)
#   - JLD2 (optional)
# ==============================================================================================

#imports
using JuMP, MosekTools, Mosek, JLD2
using LinearAlgebra, ProgressBars



# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual PEP for gradient descent 
# standard dual PEP where the step-sizes are fixed in advance
# first step of the alternating minimization
function pep_dual_memoryless(gamma)
    N = length(gamma)
    dimG = N + 2
    dimF = N + 1

    L = 1
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, tau >= 0)
    @variable(model, lb[1:dimG, 1:dimG])

    xbar = zeros(dimG, dimG)
    for i = 1:dimF
        xbar[1, i] = 1
    end

    gbar = zeros(dimG, dimG)
    for i = 1:dimF
        gbar[i + 1, i] = 1
    end

    for i = 1:N
        xbar[:, i + 1] = xbar[:, i] - gamma[i] * gbar[:, i]
    end

    fbar = Matrix{Float64}(I, dimF, dimF)
    fs = zeros(dimF, 1)
    fbar = hcat(fbar, fs)
    
    mat = tau * xbar[:, 1] * xbar[:, 1]'
    cond = fbar[:, N + 1]
    for i = 1:N + 2
        for j = 1:N + 2
            if i != j
                xi = xbar[:, i]
                xj = xbar[:, j]
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

    @constraint(model, mat in PSDCone())
    @constraint(model, cond .== 0)
    @objective(model, Min, tau)

    optimize!(model)

    objective = objective_value(model)
    dual_vars = value.(lb)

    return objective, dual_vars
end


# helper functions that computes the updates and generate the vector of associated gradient
# given step-sizes gamma
function get_steps_memoryless(i, N, gamma)
    dimG = N + 2
    dimF = N + 1
    gbar = zeros(dimG, dimG)
    for j = 1:dimF
        gbar[j + 1, j] = 1
    end

    U = zeros(dimG, 1)
    U[1, 1] = 1
    x = U
    if i > 1 && i < N + 2
        for k = 1:i-1
            x = x - gamma[k] * gbar[:, k]
        end
    elseif i == N + 2
        x = zeros(dimG, 1)
    end

    return x
end

# ---------------------------------------------------------------------------
# Second step of the alternating minimization
# ---------------------------------------------------------------------------
# similar to the dual PEP but the dual variables lambda are fixed given by the
# first step of the alternating minimization which is the standard dual PEP
# the variable here is the step-sizes gamma
function pep_dual_memoryless_outer_opt(lb, N)
    dimG = N + 2
    dimF = N + 1
    L = 1
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, gamma[1:N])
    @variable(model, t)
    
    gbar = zeros(dimG, dimG)
    for i = 1:dimF
        gbar[i + 1, i] = 1
    end

    fbar = Matrix{Float64}(I, dimF, dimF)
    fs = zeros(dimF, 1)
    fbar = [fbar fs]

    U = zeros(dimG, 1)
    U[1] = 1

    mat = t * (U * U')

    for i = 1:N+2
        for j = 1:N+2
            if i != j
                xi = get_steps_memoryless(i, N, gamma)
                xj = get_steps_memoryless(j, N, gamma)
                gi, gj = gbar[:, i], gbar[:, j]
                
                A = gj * (xi - xj)' + (xi - xj) * gj' + (1 / L) * (gi - gj) * (gi - gj)'
                mat += 0.5 * lb[i, j] * A
            end
        end
    end

    @constraint(model, mat in PSDCone())
    @constraint(model, t >= 0)

    @objective(model, Min, t)
    optimize!(model)

    objective = objective_value(model)
    steps = value.(gamma)

    return objective, steps
end


# ----------------------------------------
# Full Alternating Minimization algorithm
# ----------------------------------------
function inner_iteration(step, N, iter = 1000)

    tol = 1e-4
    gamma = step
    obj = Inf

    for j in tqdm(1:iter)
        obj, lb = pep_dual_memoryless(gamma)
        objective, steps = pep_dual_memoryless_outer_opt(lb, N)
        if norm(gamma - steps) < tol
            break
        end
        gamma = steps 
        objective, lb = pep_dual_memoryless(gamma)
        obj = objective
    end
    return obj , gamma
end


#--------------- test -------------------------

K = 2 # total number of steps

gamma = ones(K) # initial step sizes

obj , gamma = inner_iteration(gamma, K)