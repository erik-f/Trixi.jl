
function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1, max_λ2 = max_abs_speeds(equations)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::ParallelTreeMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), TreeMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::ParallelTreeMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), TreeMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::StructuredMesh,
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack size, coordinates_min, coordinates_max = mesh
  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]

  for element_x in 1:mesh.size[1], element_y in 1:mesh.size[2]
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element_x, element_y)
      λ1, λ2 = max_abs_speeds(u_node, equations)
      λ1 *= 0.5 * dy
      λ2 *= 0.5 * dx
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element_x, element_y]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::StructuredMesh,
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack size, coordinates_min, coordinates_max = mesh
  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]

  for element_x in 1:mesh.size[1], element_y in 1:mesh.size[2]
    max_λ1, max_λ2 = max_abs_speeds(equations)
    max_λ1 *= 0.5 * dy
    max_λ2 *= 0.5 * dx
    inv_jacobian = cache.elements.inverse_jacobian[element_x, element_y]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end
