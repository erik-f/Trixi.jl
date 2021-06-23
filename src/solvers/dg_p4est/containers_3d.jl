# Note: This is an experimental feature and may be changed in future releases without notice.
mutable struct P4estTreeContainer{RealT<:Real} <: AbstractContainer
  # Same arrays as in P4estElementContainer, but on trees instead of elements.
  # By https://doi.org/10.1007/s10915-018-00897-9, these need to be interpolated onto the elements
  # to ensure FSP on non-conforming meshes in 3D.
  node_coordinates     ::Array{RealT, 5}
  jacobian_matrix      ::Array{RealT, 6}
  contravariant_vectors::Array{RealT, 6}
  inverse_jacobian     ::Array{RealT, 4}
end


# Create tree container and initialize tree data
function init_trees(mesh::P4estMesh{3}, basis)
  RealT = real(mesh)
  n_trees = ntrees(mesh)

  node_coordinates      = Array{RealT, 5}(undef, 3, nnodes(basis), nnodes(basis), nnodes(basis), n_trees)
  jacobian_matrix       = Array{RealT, 6}(undef, 3, 3, nnodes(basis), nnodes(basis), nnodes(basis), n_trees)
  contravariant_vectors = similar(jacobian_matrix)
  inverse_jacobian      = Array{RealT, 4}(undef, nnodes(basis), nnodes(basis), nnodes(basis), n_trees)

  tree_container = P4estTreeContainer{RealT}(
    node_coordinates, jacobian_matrix, contravariant_vectors, inverse_jacobian)

  interpolate_tree_node_coordinates!(node_coordinates, mesh, basis)

  for tree in 1:n_trees
    calc_jacobian_matrix!(jacobian_matrix, tree, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, tree, jacobian_matrix,
                                node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, tree, jacobian_matrix, basis)
  end

  return tree_container
end


function interpolate_tree_node_coordinates!(node_coordinates, mesh, basis)
  matrix = polynomial_interpolation_matrix(mesh.nodes, basis.nodes)

  for tree in 1:ntrees(mesh)
    multiply_dimensionwise!(
      view(node_coordinates, :, :, :, :, tree),
      matrix, matrix, matrix,
      view(mesh.tree_node_coordinates, :, :, :, :, tree)
    )
  end

  return node_coordinates
end


# Initialize data structures in element container
function init_elements!(elements,
                        mesh::P4estMesh{3},
                        basis::LobattoLegendreBasis,
                        trees::P4estTreeContainer)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  calc_node_coordinates!(node_coordinates, mesh, basis.nodes)

  interpolate_trees_to_elements!(jacobian_matrix, contravariant_vectors, inverse_jacobian,
                                 mesh, basis, trees)

  return elements
end

function init_elements!(elements, mesh::P4estMesh{3}, basis, cache)
  init_elements!(elements, mesh, basis, cache.trees)
end


# Interpolate tree_node_coordinates to each quadrant at the nodes of the specified basis
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{3},
                                basis::LobattoLegendreBasis)
  # Hanging nodes will cause holes in the mesh if its polydeg is higher
  # than the polydeg of the solver.
  @assert length(basis.nodes) >= length(mesh.nodes) "The solver can't have a lower polydeg than the mesh"

  calc_node_coordinates!(node_coordinates, mesh, basis.nodes)
end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{3},
                                nodes::AbstractVector)
  # Macros from p4est
  p4est_root_len = 1 << P4EST_MAXLEVEL
  p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

  trees = unsafe_wrap_sc(p8est_tree_t, mesh.p4est.trees)

  for tree in eachindex(trees)
    offset = trees[tree].quadrants_offset
    quadrants = unsafe_wrap_sc(p8est_quadrant_t, trees[tree].quadrants)

    for i in eachindex(quadrants)
      element = offset + i
      quad = quadrants[i]

      quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

      nodes_out_x = 2 * (quad_length * 1/2 * (nodes .+ 1) .+ quad.x / p4est_root_len) .- 1
      nodes_out_y = 2 * (quad_length * 1/2 * (nodes .+ 1) .+ quad.y / p4est_root_len) .- 1
      nodes_out_z = 2 * (quad_length * 1/2 * (nodes .+ 1) .+ quad.z / p4est_root_len) .- 1

      matrix1 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_x)
      matrix2 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_y)
      matrix3 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_z)

      multiply_dimensionwise!(
        view(node_coordinates, :, :, :, :, element),
        matrix1, matrix2, matrix3,
        view(mesh.tree_node_coordinates, :, :, :, :, tree)
      )
    end
  end

  return node_coordinates
end


# Interpolate tree_node_coordinates to each quadrant
function interpolate_trees_to_elements!(jacobian_matrix, contravariant_vectors, inverse_jacobian,
                                        mesh::P4estMesh{3},
                                        basis::LobattoLegendreBasis,
                                        tree_container)
  # Macros from p4est
  p4est_root_len = 1 << P4EST_MAXLEVEL
  p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

  trees = unsafe_wrap_sc(p8est_tree_t, mesh.p4est.trees)

  for tree in eachindex(trees)
    offset = trees[tree].quadrants_offset
    quadrants = unsafe_wrap_sc(p8est_quadrant_t, trees[tree].quadrants)

    for i in eachindex(quadrants)
      element = offset + i
      quad = quadrants[i]

      quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

      nodes_out_x = 2 * (quad_length * 1/2 * (basis.nodes .+ 1) .+ quad.x / p4est_root_len) .- 1
      nodes_out_y = 2 * (quad_length * 1/2 * (basis.nodes .+ 1) .+ quad.y / p4est_root_len) .- 1
      nodes_out_z = 2 * (quad_length * 1/2 * (basis.nodes .+ 1) .+ quad.z / p4est_root_len) .- 1

      matrix1 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_x)
      matrix2 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_y)
      matrix3 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_z)

      # Interpolate jacobian_matrix
      for dim in 1:3
        multiply_dimensionwise!(
          view(jacobian_matrix, dim, :, :, :, :, element),
          matrix1, matrix2, matrix3,
          view(tree_container.jacobian_matrix, dim, :, :, :, :, tree)
        )
      end

      # Correction factor
      jacobian_matrix[.., element] ./= 2^quad.level

      # Interpolate contravariant_vectors
      for dim in 1:3
        multiply_dimensionwise!(
          view(contravariant_vectors, dim, :, :, :, :, element),
          matrix1, matrix2, matrix3,
          view(tree_container.contravariant_vectors, dim, :, :, :, :, tree)
        )
      end

      # Correction factor
      contravariant_vectors[.., element] ./= 4^quad.level

      # Reshape arrays to allow usage of multiply_dimensionwise!
      inverse_jacobian_ = unsafe_wrap(Array, pointer(inverse_jacobian),
        (1, nnodes(basis), nnodes(basis), nnodes(basis), ncells(mesh)))

      n_nodes_mesh = length(mesh.nodes)
      tree_inverse_jacobian_ = unsafe_wrap(Array, pointer(tree_container.inverse_jacobian),
        (1, n_nodes_mesh, n_nodes_mesh, n_nodes_mesh, ntrees(mesh)))

      # Interpolate inverse_jacobian
      multiply_dimensionwise!(
        view(inverse_jacobian_, :, :, :, :, element),
        matrix1, matrix2, matrix3,
        view(tree_inverse_jacobian_, :, :, :, :, tree)
      )

      # Correction factor
      inverse_jacobian_[.., element] .*= 8^quad.level
    end
  end

  return jacobian_matrix
end


# Initialize node_indices of interface container
@inline function init_interface_node_indices!(interfaces::P4estInterfaceContainer{3},
                                              faces, orientation, interface_id)
  # Iterate over primary and secondary element
  for side in 1:2
    # Align interface at the primary element (primary element has surface indices (:i, :j)).
    # The secondary element needs to be indexed differently.
    if side == 1
      surface_index1 = :i
      surface_index2 = :j
    else
      surface_index1, surface_index2 = orientation_to_indices_p4est(faces[2], faces[1], orientation)
    end

    if faces[side] == 0
      # Index face in negative x-direction
      interfaces.node_indices[side, interface_id] = (:one, surface_index1, surface_index2)
    elseif faces[side] == 1
      # Index face in positive x-direction
      interfaces.node_indices[side, interface_id] = (:end, surface_index1, surface_index2)
    elseif faces[side] == 2
      # Index face in negative y-direction
      interfaces.node_indices[side, interface_id] = (surface_index1, :one, surface_index2)
    elseif faces[side] == 3
      # Index face in positive y-direction
      interfaces.node_indices[side, interface_id] = (surface_index1, :end, surface_index2)
    elseif faces[side] == 4
      # Index face in negative z-direction
      interfaces.node_indices[side, interface_id] = (surface_index1, surface_index2, :one)
    else # faces[side] == 5
      # Index face in positive z-direction
      interfaces.node_indices[side, interface_id] = (surface_index1, surface_index2, :end)
    end
  end

  return interfaces
end


# Initialize node_indices of boundary container
@inline function init_boundary_node_indices!(boundaries::P4estBoundaryContainer{3},
                                             face, boundary_id)
  if face == 0
    # Index face in negative x-direction
    boundaries.node_indices[boundary_id] = (:one, :i, :j)
  elseif face == 1
    # Index face in positive x-direction
    boundaries.node_indices[boundary_id] = (:end, :i, :j)
  elseif face == 2
    # Index face in negative y-direction
    boundaries.node_indices[boundary_id] = (:i, :one, :j)
  elseif face == 3
    # Index face in positive y-direction
    boundaries.node_indices[boundary_id] = (:i, :end, :j)
  elseif face == 4
    # Index face in negative z-direction
    boundaries.node_indices[boundary_id] = (:i, :j, :one)
  else # face == 5
    # Index face in positive z-direction
    boundaries.node_indices[boundary_id] = (:i, :j, :end)
  end

  return boundaries
end


# Initialize node_indices of mortar container
# faces[1] is expected to be the face of the small side.
@inline function init_mortar_node_indices!(mortars::P4estMortarContainer{3},
                                           faces, orientation, mortar_id)
  for side in 1:2
    # Align mortar at small side.
    # The large side needs to be indexed differently.
    if side == 1
      surface_index1 = :i
      surface_index2 = :j
    else
      surface_index1, surface_index2 = orientation_to_indices_p4est(faces[2], faces[1], orientation)
    end

    if faces[side] == 0
      # Index face in negative x-direction
      mortars.node_indices[side, mortar_id] = (:one, surface_index1, surface_index2)
    elseif faces[side] == 1
      # Index face in positive x-direction
      mortars.node_indices[side, mortar_id] = (:end, surface_index1, surface_index2)
    elseif faces[side] == 2
      # Index face in negative y-direction
      mortars.node_indices[side, mortar_id] = (surface_index1, :one, surface_index2)
    elseif faces[side] == 3
      # Index face in positive y-direction
      mortars.node_indices[side, mortar_id] = (surface_index1, :end, surface_index2)
    elseif faces[side] == 4
      # Index face in negative z-direction
      mortars.node_indices[side, mortar_id] = (surface_index1, surface_index2, :one)
    else # faces[side] == 5
      # Index face in positive z-direction
      mortars.node_indices[side, mortar_id] = (surface_index1, surface_index2, :end)
    end
  end

  return mortars
end


# Convert p4est orientation code to node indices.
# Return node indices that index "my side" wrt "other side",
# i.e., i and j are indices of other side.
function orientation_to_indices_p4est(my_face, other_face, orientation_code)
  # my_face and other_face are the face directions (zero-based)
  # of "my side" and "other side" respectively.
  # Face corner 0 of the face with the lower face direction connects to a corner of the other face.
  # The number of this corner is the orientation code in p4est.
  lower = my_face <= other_face

  # x_pos, y_neg, and z_pos are the directions in which the face has right-handed coordinates
  # when looked at from the outside.
  my_right_handed = my_face in (1, 2, 5)
  other_right_handed = other_face in (1, 2, 5)

  # If both or none are right-handed when looked at from the outside, they will have different
  # orientations when looked at from the same side of the interface.
  flipped = my_right_handed == other_right_handed

  # In the folowing illustrations, p4est's face corner numbering is shown.
  # ξ and η are the local coordinates of the respective face.
  # We're looking at both faces from the same side of the interface, so that "other side"
  # (in the illustrations on the left) has right-handed coordinates.
  if !flipped
    if orientation_code == 0
      # Corner 0 of other side matches corner 0 of my side
      #   2┌──────┐3   2┌──────┐3
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   0└──────┘1
      #     η            η
      #     ↑            ↑
      #     │            │
      #     └───> ξ      └───> ξ
      surface_index1 = :i
      surface_index2 = :j
    elseif ((lower && orientation_code == 2) # Corner 0 of my side matches corner 2 of other side
        || (!lower && orientation_code == 1)) # Corner 0 of other side matches corner 1 of my side
      #   2┌──────┐3   0┌──────┐2
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   1└──────┘3
      #     η            ┌───> η
      #     ↑            │
      #     │            ↓
      #     └───> ξ      ξ
      surface_index1 = :j_backwards
      surface_index2 = :i
    elseif ((lower && orientation_code == 1) # Corner 0 of my side matches corner 1 of other side
        || (!lower && orientation_code == 2)) # Corner 0 of other side matches corner 2 of my side
      #   2┌──────┐3   3┌──────┐1
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   2└──────┘0
      #     η                 ξ
      #     ↑                 ↑
      #     │                 │
      #     └───> ξ     η <───┘
      surface_index1 = :j
      surface_index2 = :i_backwards
    else # orientation_code == 3
      # Corner 0 of my side matches corner 3 of other side and
      # corner 0 of other side matches corner 3 of my side.
      #   2┌──────┐3   1┌──────┐0
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   3└──────┘2
      #     η           ξ <───┐
      #     ↑                 │
      #     │                 ↓
      #     └───> ξ           η
      surface_index1 = :i_backwards
      surface_index2 = :j_backwards
    end
  else # flipped
    if orientation_code == 0
      # Corner 0 of other side matches corner 0 of my side
      #   2┌──────┐3   1┌──────┐3
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   0└──────┘2
      #     η            ξ
      #     ↑            ↑
      #     │            │
      #     └───> ξ      └───> η
      surface_index1 = :j
      surface_index2 = :i
    elseif orientation_code == 2
      # Corner 0 of my side matches corner 2 of other side and
      # corner 0 of other side matches corner 2 of my side.
      #   2┌──────┐3   0┌──────┐1
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   2└──────┘3
      #     η            ┌───> ξ
      #     ↑            │
      #     │            ↓
      #     └───> ξ      η
      surface_index1 = :i
      surface_index2 = :j_backwards
    elseif orientation_code == 1
      # Corner 0 of my side matches corner 1 of other side and
      # corner 0 of other side matches corner 1 of my side.
      #   2┌──────┐3   3┌──────┐2
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   1└──────┘0
      #     η                 η
      #     ↑                 ↑
      #     │                 │
      #     └───> ξ     ξ <───┘
      surface_index1 = :i_backwards
      surface_index2 = :j
    else # orientation_code == 3
      # Corner 0 of my side matches corner 3 of other side and
      # corner 0 of other side matches corner 3 of my side.
      #   2┌──────┐3   2┌──────┐0
      #    │      │     │      │
      #    │      │     │      │
      #   0└──────┘1   3└──────┘1
      #     η           η <───┐
      #     ↑                 │
      #     │                 ↓
      #     └───> ξ           ξ
      surface_index1 = :j_backwards
      surface_index2 = :i_backwards
    end
  end

  return surface_index1, surface_index2
end
