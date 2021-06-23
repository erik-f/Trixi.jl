
# Extract outward-pointing normal vector (contravariant vector ±Ja^i, i = index) as SVector
# Note that this vector is not normalized
@inline function get_normal_vector(direction, cache, indices...)
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  orientation = div(direction + 1, 2)
  normal = get_contravariant_vector(orientation, contravariant_vectors, indices...)

  # Contravariant vectors at interfaces in negative coordinate direction are pointing inwards
  if direction in (1, 3, 5)
    normal *= -1
  end

  return normal
end


@inline ndofs(mesh::P4estMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_2d.jl")
include("dg_3d.jl")
