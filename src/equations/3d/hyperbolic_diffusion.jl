
@doc raw"""
    HyperbolicDiffusionEquations3D

The linear hyperbolic diffusion equations in three space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations3D <: AbstractHyperbolicDiffusionEquations{3, 4}
  Lr::Float64
  Tr::Float64
  nu::Float64
  resid_tol::Float64
end

function HyperbolicDiffusionEquations3D()
  # diffusion coefficient
  nu = parameter("nu", 1.0)
  # relaxation length scale
  Lr = parameter("Lr", 1.0/(2.0*pi))
  # relaxation time
  Tr = Lr*Lr/nu
  # stopping tolerance for the pseudotime "steady-state"
  resid_tol = parameter("resid_tol", 1e-12)
  HyperbolicDiffusionEquations3D(Lr, Tr, nu, resid_tol)
end


get_name(::HyperbolicDiffusionEquations3D) = "HyperbolicDiffusionEquations3D"
varnames_cons(::HyperbolicDiffusionEquations3D) = @SVector ["phi", "q1", "q2", "q3"]
varnames_prim(::HyperbolicDiffusionEquations3D) = @SVector ["phi", "q1", "q2", "q3"]
default_analysis_quantities(::HyperbolicDiffusionEquations3D) = (:l2_error, :linf_error, :residual)


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_conditions_poisson_periodic(x, t, equation::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    q1  = 0.0
    q2  = 0.0
    q3  = 0.0
  else
    phi =          sin(2 * pi * x[1]) * sin(2 * pi * x[2]) * sin(2 * pi * x[3])
    q1  = 2 * pi * cos(2 * pi * x[1]) * sin(2 * pi * x[2]) * sin(2 * pi * x[3])
    q2  = 2 * pi * sin(2 * pi * x[1]) * cos(2 * pi * x[2]) * sin(2 * pi * x[3])
    q3  = 2 * pi * sin(2 * pi * x[1]) * sin(2 * pi * x[2]) * cos(2 * pi * x[3])
  end
  return @SVector [phi, q1, q2, q3]
end

function initial_conditions_poisson_nonperiodic(x, t, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
  else
    phi = 2.0*cos(pi*x[1])*sin(2.0*pi*x[2]) + 2.0 # ϕ
    q1  = -2.0*pi*sin(pi*x[1])*sin(2.0*pi*x[2])   # ϕ_x
    q2  = 4.0*pi*cos(pi*x[1])*cos(2.0*pi*x[2])    # ϕ_y
  end
  return @SVector [phi, q1, q2]
end

function initial_conditions_harmonic_nonperiodic(x, t, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
  else
    C   = 1.0/sinh(pi)
    phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
    q1  = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
    q2  = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
  end
  return @SVector [phi, q1, q2]
end

function initial_conditions_jeans_instability(x, t, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  # gravity equation: -Δϕ = -4πGρ
  # Constants taken from the FLASH manual
  # https://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel.pdf
  rho0 = 1.5e7
  delta0 = 1e-3
  #
  phi = rho0*delta0 # constant background pertubation magnitude
  q1  = 0.0
  q2  = 0.0
  return @SVector [phi, q1, q2]
end

function initial_conditions_eoc_test_coupled_euler_gravity(x, t, equation::HyperbolicDiffusionEquations3D)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C_grav = -4 * G / (3 * pi) # "3" is the number of spatial dimensions  # 2D: -2.0*G/pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A * sin(pi * (x[1] + x[2] + x[3] - t))
  # intialize with ansatz of gravity potential
  phi = C_grav * rho1
  q1  = C_grav * A * pi * cos(pi*(x[1] + x[2] + x[3] - t)) # = gravity acceleration in x-direction
  q2  = q1                                                 # = gravity acceleration in y-direction
  q3  = q1                                                 # = gravity acceleration in z-direction

  return @SVector [phi, q1, q2, q3]
end


function initial_conditions_sedov_self_gravity(x, t, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  # for now just use constant initial condition for sedov blast wave (can likely be improved)

  phi = 0.0
  q1  = 0.0
  q2  = 0.0
  return @SVector [phi, q1, q2]
end

# Apply source terms
function source_terms_poisson_periodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy)*sin(2πz) and f = -12νπ^2 sin(2πx)*sin(2πy)*sin(2πz)
  inv_Tr = inv(equation.Tr)
  C = -12.0*equation.nu*pi*pi

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, k, element_id]
    x2 = x[2, i, j, k, element_id]
    x3 = x[3, i, j, k, element_id]
    tmp1 = sin(2 * pi * x1)
    tmp2 = sin(2 * pi * x2)
    tmp3 = sin(2 * pi * x3)
    ut[1, i, j, k, element_id] -= C*tmp1*tmp2*tmp3
    ut[2, i, j, k, element_id] -= inv_Tr * u[2, i, j, k, element_id]
    ut[3, i, j, k, element_id] -= inv_Tr * u[3, i, j, k, element_id]
    ut[4, i, j, k, element_id] -= inv_Tr * u[4, i, j, k, element_id]
  end

  return nothing
end

function source_terms_poisson_nonperiodic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      ut[1, i, j, element_id] += 10 * pi^2 * cos(pi*x1) * sin(2.0*pi*x2)
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations3D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  inv_Tr = inv(equation.Tr)

  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    ut[2, i, j, k, element_id] -= inv_Tr * u[2, i, j, k, element_id]
    ut[3, i, j, k, element_id] -= inv_Tr * u[3, i, j, k, element_id]
    ut[4, i, j, k, element_id] -= inv_Tr * u[4, i, j, k, element_id]
  end

  return nothing
end

# The coupled EOC test does not require additional sources
function source_terms_eoc_test_coupled_euler_gravity(ut, u, x, element_id, t, n_nodes, equation::HyperbolicDiffusionEquations3D) # FIXME: ndims
  return source_terms_harmonic(ut, u, x, element_id, t, n_nodes, equation)
end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::HyperbolicDiffusionEquations3D)
  phi, q1, q2, q3 = u

  if orientation == 1
    f1 = -equation.nu*q1
    f2 = -phi/equation.Tr
    f3 = zero(phi)
    f4 = zero(phi)
  elseif orientation == 2
    f1 = -equation.nu*q2
    f2 = zero(phi)
    f3 = -phi/equation.Tr
    f4 = zero(phi)
  else
    f1 = -equation.nu*q3
    f2 = zero(phi)
    f3 = zero(phi)
    f4 = -phi/equation.Tr
  end

  return SVector(f1, f2, f3, f4)
end


@inline function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations3D)
  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = sqrt(equation.nu / equation.Tr)

  return 0.5 * (f_ll + f_rr - λ_max * (u_rr - u_ll))
end


@inline function flux_upwind(u_ll, u_rr, orientation, equation::HyperbolicDiffusionEquations3D)
  # Obtain left and right fluxes
  phi_ll, q1_ll, q2_ll, q3_ll = u_ll
  phi_rr, q1_rr, q2_rr, q3_rr = u_rr
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equation.nu/equation.Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (q1_rr - q1_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  elseif orientation == 2 # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q2_rr - q2_ll)
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (q3_rr - q3_ll)
  end

  return SVector(f1, f2, f3, f4)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, n_nodes, invjacobian, cfl,
                     equation::HyperbolicDiffusionEquations3D)
  dt = cfl * 2 / (invjacobian * sqrt(equation.nu/equation.Tr)) / n_nodes

  return dt
end


# Convert conservative variables to primitive
cons2prim(cons, equation::HyperbolicDiffusionEquations3D) =  cons

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
function cons2entropy(cons, n_nodes, n_elements, equation::HyperbolicDiffusionEquations3D)
  entropy = similar(cons)
  @. entropy[1, :, :, :, :] = cons[1, :, :, :, :]
  @. entropy[2:4, :, :, :, :] = equation.Lr^2 * cons[2:4, :, :, :, :]

  return entropy
end


# Calculate entropy for a conservative state `cons` (here: same as total energy)
@inline entropy(cons, equation::HyperbolicDiffusionEquations3D) = energy_total(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline function energy_total(cons, equation::HyperbolicDiffusionEquations3D)
  # energy function as found in equation (2.5.12) in the book "I Do Like CFD, Vol. 1"
  return 0.5*(cons[1]^2 + equation.Lr^2 * (cons[2]^2 + cons[3]^2 + cons[4]^2))
end
