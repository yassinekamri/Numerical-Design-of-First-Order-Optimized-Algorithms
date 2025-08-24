# ============================================================================
# Cyclic Coordinate Descent — Step-Size Optimization via Linearization method
# ----------------------------------------------------------------------------
# Julia code to optimize the step-sizes of cyclic coordinate descent using the
# linearization method described in:
#   Y. Kamri, J. M. Hendrickx, and F. Glineur.
#   "Numerical Design of Optimized First-Order Algorithms." arXiv, 2025.
#   Link: https://arxiv.org/abs/2507.20773
#
# Related work:
#   Y. Kamri, F. Glineur, J. M. Hendrickx, and I. Necoara.
#   "On the Worst-Case Analysis of Cyclic Block Coordinate Descent Type Algorithms."
#   arXiv, 2025. Link: https://arxiv.org/abs/2507.16675
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
using SparseArrays
using JLD2


# ---------------------------------------------------------------------------
# Compute xbar and gbar for cyclic coordinate descent
# ---------------------------------------------------------------------------
# Populates the vectors xbar and gbar, which represent respectively the blocks
# of coordinates of the iterates of the algorithm and the associated gradients,
# given the step-sizes gamma, the number of steps K, and the number of blocks nblocks.
function compute_x_g!(gamma, K, nblocks, xbar, gbar)

    for i in 1:K+2
        for j in 1:nblocks
            xbar[i, j] .= 0.0 
            gbar[i,j] .= 0.0
        end
    end

    dimG = K + 2
    dimF = K + 1
    
    
    for i in 1:nblocks
        xbar[1,i][1] = 1
    end

    
    for j in 1:(K+2)
        for i in 1:nblocks
            if j <= K+1
                gbar[j,i][j+1] = 1
            end
        end
    end

    # Iterative updates of cyclic coordinate descent
    for i in 1:K+2
        iter = [zeros(dimG, 1) for j in 1:nblocks]
        if i == 1
            for t in 1:nblocks
                iter[t] = xbar[1,t]
            end  
        elseif i == K+2
            for t in 1:nblocks
                iter[t] = xbar[K+2,t]
            end   
        else
            for t in 1:nblocks
                iter[t] = xbar[1,t]
            end 
            for k = 1:i-1
                idx = mod(k, nblocks) + 1
                iter[idx] -= gamma[k] * gbar[k,idx]
            end
        end
        for t in 1:nblocks
            xbar[i,t] = iter[t]
        end
    end
end


# ---------------------------------------------------------------------------
# Primal PEP formulation
# ---------------------------------------------------------------------------
# Computes the primal formulation of the PEP for cyclic coordinate descent.
# See Kamri et al., "On the Worst-Case Analysis of Cyclic Block Coordinate
# Descent Type Algorithms" (2025) for theoretical background.
function pep_coords_primal(K, nblocks, xbar, gbar, fbar)
    dimG = K + 2  
    dimF = K + 1

    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Variables:
    #   F: functional values f_i = F' * fbar[:,i] 
    #   G: Gram matrices for each block of coordinates
    # |x^(i)_k|^2 = xbar[k, i]'  * G[i, :, :] * xbar[k, i]
    # |g^(i)_k|^2 = gbar[k, i]' * G[i, :, :] * gbar[k, i]
    # g^(i)_k * x^(i)_j = gbar[k, i]' * G[i, :, :] * xbar[j, i]
    @variable(model, F[1:dimF])
    @variable(model, G[1:nblocks, 1:dimG, 1:dimG])

  

    # Initial condition for the PEP # initial condition for the PEP see Setting INIT. in https://arxiv.org/abs/2507.16675
    @constraint(
        model,
        sum(L[i] * (xbar[1, i]' * G[i, :, :] * xbar[1, i])[1] for i in 1:nblocks) <= 1
    )

    # Interpolation conditions with all coordinate-wise smoothness constants = 1, see paper in https://arxiv.org/abs/2507.16675
    for i in 1:dimG
        for j in 1:dimG
            if i != j
                fi = fbar[:, i]
                fj = fbar[:, j]

                cond = F' * (fj - fi)

                for t in 1:nblocks
                    g_term = (gbar[j, t]' * G[t, :, :] * (xbar[i, t] - xbar[j, t]))[1]
                    cond += g_term
                end

                for t in 1:nblocks
                    grad_dist =  (gbar[j, t] - gbar[i, t])' * G[t, :, :] * (gbar[j, t] - gbar[i, t])
                    grad_dist_scalar = grad_dist[1] 
                    @constraint(model, cond + grad_dist_scalar <= 0)
                end
            end
        end
    end

    # PSD constraints
    for t in 1:nblocks
        @constraint(model, G[t,:,:] in PSDCone())
    end

    # Objective: maximize f_{K+1}
    @objective(model, Max, F' * fbar[:, K+1])

    optimize!(model)

    objective = objective_value(model)
    G_val = value.(G)

    return objective, G_val
end


# ---------------------------------------------------------------------------
# Dual PEP formulation
# ---------------------------------------------------------------------------
# Constructs and solves the dual problem associated with pep_coords_primal.
function pep_coords_dual(K, nblocks, xbar,gbar, fbar)
    dimG = K + 2  
    dimF = K + 1

    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, tau >= 0)
    @variable(model, lb[1:nblocks, 1:(K+2), 1:(K+2)])

    mat = [tau * xbar[1,j] * xbar[1,j]' for j in 1:nblocks]
    cond = fbar[:,K+1] 

    for i in 1:(K+2)
        for j in 1:(K+2)
            if i != j
                fi = fbar[:,i]
                fj = fbar[:,j]
                AA = fj - fi
                for t in 1:nblocks
                    @constraint(model, lb[t,i,j] >= 0)
                    cond -= lb[t,i,j] * AA
                    for k in 1:nblocks
                        if k != t
                            A = gbar[j,k] * (xbar[i,k] - xbar[j,k])' + (xbar[i,k] - xbar[j,k]) * gbar[j,k]'
                            mat[k] += 0.5 * lb[t,i,j] * A
                        else
                            A = gbar[j,t] * (xbar[i,t] - xbar[j,t])' + (xbar[i,t] - xbar[j,t]) * gbar[j,t]' +  (gbar[i,t] - gbar[j,t]) * (gbar[i,t] - gbar[j,t])'
                            mat[t] += 0.5 * lb[t,i,j] * A
                        end
                    end
                end
            end
        end
    end

    @constraint(model, cond == 0)
    for i in 1:nblocks
        @constraint(model, mat[i] in PSDCone())
    end

    @objective(model, Min, tau)

    optimize!(model)

    objective = objective_value(model)
    lb_val = value.(lb)
    mat_val = [zeros(K+2,K+2) for j in 1:nblocks]
    for i = 1:nblocks
        mat_val[i] = value.(mat[i])
    end

    return objective, lb_val, mat_val
end


# ---------------------------------------------------------------------------------------------
# Derivatives of the SDP matrices of the dual PEP with respect to dual variables and step-sizes
# ---------------------------------------------------------------------------------------------
# Computes the derivatives of the SDP matrices in the dual PEP w.r.t. dual variables lambda
function derivative_lambda!(i,j,t,xbar,gbar,mat_lambda)
    for i in 1:nblocks
        fill!(mat_lambda[i], 0.0)
    end
    if i != j 
        for k in 1:nblocks
            if k != t
                mat_lambda[k] = 0.5 * (gbar[j,k] * (xbar[i,k] - xbar[j,k])' + (xbar[i,k] - xbar[j,k]) * gbar[j,k]') 
            else
                mat_lambda[t] = 0.5 * ( gbar[j,t] * (xbar[i,t] - xbar[j,t])' + (xbar[i,t] - xbar[j,t]) * gbar[j,t]' + (gbar[i,t] - gbar[j,t]) * (gbar[i,t] - gbar[j,t])'  )
            end
        end
    end
end

# Computes the derivative of the i-th iterate with respect to the j-th step-size gamma_j
function dxi_dgammaj(K,i,j,gbar)
    dimG = K + 2
    grad = [zeros(dimG, 1) for j in 1:nblocks]

    if i != 1 && i != K+2 && j <= i - 1
        idx = mod(j, nblocks) + 1
        grad[idx] =  -gbar[j,idx]
    end
    return grad
end

# Computes the derivatives of the SDP matrices in the dual PEP w.r.t. to the step sizes gamma
function derivative_gamma!(l,gbar,lb,mat_gamma)
    for i in 1:nblocks
        fill!(mat_gamma[i],0.0)
    end
    for i in 1:K+2
        for j in 1:K+2
            if i != j
                dxi = dxi_dgammaj(K,i,l,gbar)
                dxj = dxi_dgammaj(K,j,l,gbar)
                for t in 1:nblocks
                    for k in 1:nblocks
                        mat_gamma[k] += 0.5 *lb[t,i,j] * (gbar[j,k] * (dxi[k] - dxj[k])' + (dxi[k] - dxj[k]) * gbar[j,k]')
                    end
                end
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
function linearized_pep_coords(gamma_init,nblocks,xbar,gbar,fbar,mat_gamma,mat_lambda,derivative_tau,delta)

    compute_x_g!(gamma_init, K,nblocks,xbar,gbar)
    obj, lb_val, mat_val = pep_coords_dual(K, nblocks, xbar,gbar, fbar)

    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, tau)
    @variable(model, lb[1:nblocks, 1:(K+2), 1:(K+2)] >= 0)
    @variable(model, gamma[1:K])
    @variable(model,s)

    @constraint(model, tau + obj >= 0)
    mat = [mat_val[i] + tau*derivative_tau for i in 1:nblocks]

    for i in 1:K
        derivative_gamma!(i,gbar,lb_val,mat_gamma)
        for k in 1:nblocks
            mat[k] = mat[k] + mat_gamma[k] * gamma[i]
        end
    end

    cond = fbar[:,K+2]
    for i in 1:K+2
        for j in 1:K+2
            if i != j
                fi = fbar[:,i]
                fj = fbar[:,j]
                AA = fj - fi
                for t in 1:nblocks
                    cond -= lb[t,i,j] * AA
                    @constraint(model, lb[t,i,j] + lb_val[t,i,j] >= 0)
                    derivative_lambda!(i,j,t,xbar,gbar,mat_lambda)
                    for l in 1:nblocks
                        mat[l] += lb[t, i, j] * mat_lambda[l]
                    end
                end
            end
        end
    end

    x = vcat(gamma,vec(lb))
    obj = tau 
    @constraint(model, [s ; x] in SecondOrderCone())
    @constraint(model, s <= delta)

    for i = 1:nblocks
        @constraint(model, mat[i] in PSDCone())
    end
    @constraint(model, cond .== 0)
    @objective(model, Min, obj)

    optimize!(model)

    tau_val = objective_value(model)
    dual_vars = value.(lb)
    gamma_val = value.(gamma)
    return tau_val,dual_vars,gamma_val
end


# ---------------------------------------------------------------------------
# Full optimization algorithm with trust-region updates
# ---------------------------------------------------------------------------
# Implements the iterative linearization method with a trust-region strategy
# for updating step-sizes gamma.

function inner_iteration(gamma, nblocks, delta, max_iters=1000, tol=1e-7)
    # Preallocate memory
    K = size(gamma,1)
    dimG = K + 2
    dimF = K + 1
    xbar = [zeros(dimG, 1) for _ in 1:(K+2), j in 1:nblocks]
    gbar = [zeros(dimG, 1) for _ in 1:(K+2), j in 1:nblocks]
    fbar = Matrix(I, K+1, K+1) 
    fbar = hcat(fbar, zeros(K+1, 1))  
    derivative_tau = zeros(K+2,K+2)
    derivative_tau[1,1] = 1
    mat_gamma = [zeros(K+2,K+2) for _ in 1:nblocks]
    mat_lambda = [zeros(K+2,K+2) for _ in 1:nblocks]
    
    for i in tqdm(1:max_iters)
        
        # Linearized subproblem to compute an update
        tau_val, _ , gamma_val = linearized_pep_coords(gamma,nblocks,xbar,gbar,fbar,mat_gamma,mat_lambda,derivative_tau,delta)
        

        # Stopping criterion
        if norm(gamma_val) <= tol
            break
        end

        # Trust-region update strategy
        compute_x_g!(gamma, K,nblocks,xbar,gbar)
        obj, _ , _ = pep_coords_dual(K, nblocks, xbar,gbar, fbar)

        compute_x_g!(gamma + gamma_val, K,nblocks,xbar,gbar)
        obj1, _ , _ = pep_coords_dual(K, nblocks, xbar,gbar, fbar)
        
        ratio = (obj1 - obj) / tau_val
        
        if ratio > 0.9
            delta *= 2
            gamma .+= gamma_val
        elseif ratio < 0.1
            delta *= 0.5
        else
            gamma .+= gamma_val
        end
    end
    compute_x_g!(gamma, K,nblocks,xbar,gbar)
    obj, _ , _ = pep_coords_dual(K, nblocks, xbar,gbar, fbar)
    return obj, gamma, delta
end


#--------------- test -----------------------

K = 4 # Total number of steps

nblocks = 2 # number of blocks 

gamma = ones(K) # intial step sizes

delta = 100 # intial value for the trust region diameter

obj,gamma, delta = inner_iteration(gamma, nblocks, delta)

