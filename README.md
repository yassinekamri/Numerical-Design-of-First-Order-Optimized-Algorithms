# Numerical Design of Optimized First-Order Algorithms

This repository contains code to reproduce the results of the following [paper](https://arxiv.org/abs/2507.20773):  

**Yassine Kamri, Julien M. Hendrickx, and François Glineur.** *Numerical Design of Optimized First-Order Algorithms.* arXiv preprint arXiv:2507.20773, 2025.

---

## Authors
- Yassine Kamri  
- Julien M. Hendrickx  
- François Glineur  

---

## Getting Started
The code is written in Julia and requires the [JuMP](https://jump.dev) optimization toolbox together with the [Mosek](https://www.mosek.com) SDP solver.

---

## Description of the Files

- **benchmark_opt_steps_gradient_descent_alternating_minimization_method.jl**  
  Reproduces the benchmark results of the article. Optimizes the step sizes of memoryless gradient descent over smooth convex functions using our **Alternating Minimization (AM) Method**.  

- **benchmark_opt_steps_gradient_descent_first_order_method.jl**  
  Reproduces the benchmark results of the article. Optimizes the step sizes of memoryless gradient descent over smooth convex functions using our **First-Order Method (FOM)**.  

- **benchmark_opt_steps_gradient_descent_linearization_method.jl**  
  Reproduces the benchmark results of the article. Optimizes the step sizes of memoryless gradient descent over smooth convex functions using our **Successive Linearization Method (SLM)**.  

- **opt_step_ccd_full.jl**  
  Optimizes the step sizes of full cyclic coordinate descent (using past gradient information for the updates) over coordinate-wise smooth convex functions using SLM.  

- **opt_steps_ccd.jl**  
  Optimizes the step sizes of cyclic coordinate descent over coordinate-wise smooth convex functions using SLM.  

- **opt_steps_full_inexact_gradient_descent_functional_accuracy.jl**  
  Optimizes the step sizes of full inexact gradient descent (using past gradient information for the updates) over smooth convex functions for the **functional accuracy criterion** using SLM.  

- **opt_steps_full_inexact_gradient_residual_gradient_norm.jl**  
  Optimizes the step sizes of full inexact gradient descent (using past gradient information for the updates) over smooth convex functions for the **residual gradient norm criterion** using SLM.  

- **opt_steps_inexact_gradient_descent_functional_accuracy.jl**  
  Optimizes the step sizes of inexact gradient descent over smooth convex functions for the **functional accuracy criterion** using SLM.  

- **opt_steps_inexact_gradient_descent_residual_gradient_norm.jl**  
  Optimizes the step sizes of inexact gradient descent over smooth convex functions for the **residual gradient norm criterion** using SLM.  
