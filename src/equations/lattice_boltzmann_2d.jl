
@doc raw"""
    LatticeBoltzmannEquation2D

The Lattice-Boltzmann equation
```math
\partial_t u_\alpha + v_{\alpha,1} \partial_1 u_\alpha + v_{\alpha,2} \partial_2 u_\alpha = 0
```
in two space dimensions.
"""
struct LatticeBoltzmannEquation2D{RealT<:Real, CollisionOp} <: AbstractLatticeBoltzmannEquation{2, 9}
  c::RealT
  c_s::RealT
  Ma::RealT
  u0::RealT
  Re::RealT
  L::RealT
  nu::RealT

  omega_alpha::SVector{9, RealT}
  v_alpha1::SVector{9, RealT}
  v_alpha2::SVector{9, RealT}

  collision_op::CollisionOp
end

function LatticeBoltzmannEquation2D(; Ma, Re, collision_op=collision_bg,
                                    c::Real=1, L::Real=1, u0=nothing, nu=nothing)
  # Sanity check that exactly one of Ma, u0 is not `nothing`
  if isnothing(Ma) && isnothing(u0)
    error("Mach number `Ma` and reference speed `u0` may not both be `nothing`")
  elseif !isnothing(Ma) && !isnothing(u0)
    error("Mach number `Ma` and reference speed `u0` may not both be set")
  end

  # Sanity check that exactly one of Re, nu is not `nothing`
  if isnothing(Re) && isnothing(nu)
    error("Reynolds number `Re` and visocsity `nu` may not both be `nothing`")
  elseif !isnothing(Re) && !isnothing(nu)
    error("Reynolds number `Re` and visocsity `nu` may not both be set")
  end

  # Calculate speed of sound
  c_s = c / sqrt(3)

  # Calculate missing quantities
  if isnothing(Ma)
    Ma = u0 / c_s
  elseif isnothing(u0)
    u0 = Ma * c_s
  end
  if isnothing(Re)
    Re = u0 * L / nu
  elseif isnothing(nu)
    nu = u0 * L / Re
  end

  # Promote to common data type
  Ma, Re, c, L, u0, nu = promote(Ma, Re, c, L, u0, nu)

  # Source for weights and speeds: https://cims.nyu.edu/~billbao/report930.pdf
  omega_alpha = @SVector [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
  v_alpha1 =    @SVector [ 0,  c,  0, -c,  0,  c, -c,  -c,  c]
  v_alpha2 =    @SVector [ 0,  0,  c,  0, -c,  c,  c,  -c, -c]

  LatticeBoltzmannEquation2D(c, c_s, Ma, u0, Re, L, nu,
                             omega_alpha, v_alpha1, v_alpha2,
                             collision_op)
end


get_name(::LatticeBoltzmannEquation2D) = "LatticeBoltzmannEquation2D"
varnames_cons(::LatticeBoltzmannEquation2D) = @SVector ["pdf"*string(i) for i in 1:9]
varnames_prim(::LatticeBoltzmannEquation2D) = @SVector ["rho", "v1", "v2", "p"]

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equation::LatticeBoltzmannEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LatticeBoltzmannEquation2D)
  @unpack u0 = equation
  return @SVector [2.0]
end


"""
    initial_condition_convergence_test(x, t, equation::LatticeBoltzmannEquation2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::LatticeBoltzmannEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans))
  return @SVector [scalar]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equation::LatticeBoltzmannEquation2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::LatticeBoltzmannEquation2D)
  if orientation == 1
    v_alpha = equation.v_alpha1
  else
    v_alpha = equation.v_alpha2
  end
  return v_alpha .* u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LatticeBoltzmannEquation2D)
  if orientation == 1
    v_alpha = equation.v_alpha1
  else
    v_alpha = equation.v_alpha2
  end
  return 0.5 * ( v_alpha .* (u_ll + u_rr) - abs.(v_alpha) .* (u_rr - u_ll) )
end


density(u, equation::LatticeBoltzmannEquation2D) = sum(u)


function velocity(u, orientation, equation::LatticeBoltzmannEquation2D)
  if orientation == 1
    v_alpha = equation.v_alpha1
  else
    v_alpha = equation.v_alpha2
  end
  
  return sum(v_alpha .* u)/density(u, equation)
end


function velocity(u, equation::LatticeBoltzmannEquation2D)
  @unpack v_alpha1, v_alpha2 = equation
  rho = density(u, equation)
  
  return SVector(sum(v_alpha1 .* u)/rho, sum(v_alpha2 .* u)/rho)
end


function local_maxwell_equilibrium(alpha::Real, rho, v1, v2, equation::LatticeBoltzmannEquation2D)
  @unpack omega_alpha, c_s, v_alpha1, v_alpha2 = equation

  va_v = v_alpha1[alpha]*v1 + v_alpha2[alpha]*v2
  cs_squared = c_s^2
  v_squared = v1^2 + v2^2

  return omega_alpha[alpha] * rho * (1 + va_v/cs_squared
                                       + va_v^2/(2*cs_squared^2)
                                       - v_squared/(2*cs_squared))
end


function local_maxwell_equilibrium(alpha, u, equation::LatticeBoltzmannEquation2D)
  rho = density(u, equation)
  v1, v2 = velocity(u, equation)

  return local_maxwell_equilibrium(u, alpha, rho, v1, v2, equation)
end


function local_maxwell_equilibrium(u, rho, v1, v2, equation::LatticeBoltzmannEquation2D)
  return SVector(local_maxwell_equilibrium(u, 1, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 2, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 3, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 4, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 5, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 6, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 7, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 8, rho, v1, v2, equation),
                 local_maxwell_equilibrium(u, 9, rho, v1, v2, equation))
end


function local_maxwell_equilibrium(u, equation::LatticeBoltzmannEquation2D)
  rho = density(u, equation)
  v1, v2 = velocity(u, equation)

  return local_maxwell_equilibrium(rho, v1, v2, equation)
end


function collision_bgk(u, dt, equation::LatticeBoltzmannEquation2D)
  @unpack c_s, nu = equation
  tau = nu / (c_s^2 * dt)
  return -(u - local_maxwell_equilibrium(u, equation))/(tau + 1/2)
end



@inline have_constant_speed(::LatticeBoltzmannEquation2D) = Val(true)

@inline function max_abs_speeds(equation::LatticeBoltzmannEquation2D)
  return SVector(1, 1) * equation.c
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")


# Calculate entropy for a conservative state `cons`
@inline entropy(u, equation::LatticeBoltzmannEquation2D) = error("not implemented") 


# Calculate total energy for a conservative state `cons`
@inline energy_total(u, equation::LatticeBoltzmannEquation2D) = error("not implemented")
