"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

#from fenics import SubDomain, Point, PointSource, vertices, FunctionSpace, Function, edges, cells
import numpy as np
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# TODO: which type? possibly a numpy array or similar
# need python 3.12. Possibly add interpreter checks
type ppFunction = np.array # TODO: Check. For now assume that it is N x dim for vector fields, N x 1 for scalar fields

class CouplingBoundaryType(Enum):
    """
    Defines on which boundary we want the coupling.
    Used in initialization to extract vertex IDs and coords on the coupling boundary.
    """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class Vertices:
    """
    Vertices class provides a generic skeleton for vertices. A set of vertices has a set of IDs and
    coordinates as defined in PorePy.
    """

    def __init__(self):
        self._coordinates = None
        self._ids = None

    def set_ids(self, ids):
        self._ids = ids

    def set_coordinates(self, coords):
        self._coordinates = coords

    def get_ids(self):
        return copy.deepcopy(self._ids)
    
    def get_coordinates(self):
        return copy.deepcopy(self._coordinates)


class FunctionType(Enum):
    """
    Defines scalar- and vector-valued function.
    Used in assertions to check if a PorePy function is scalar or vector.
    """
    SCALAR = 0  # scalar valued function
    VECTOR = 1  # vector valued function


class CouplingMode(Enum):
    """
    Defines the type of coupling being used.
    Options are: Bi-directional coupling, Uni-directional Write Coupling, Uni-directional Read Coupling
    Used in assertions to check which type of coupling is done
    """
    BI_DIRECTIONAL_COUPLING = 4
    UNI_DIRECTIONAL_WRITE_COUPLING = 5
    UNI_DIRECTIONAL_READ_COUPLING = 6


# # TODO: implementation needed
# def determine_function_type(input_obj, dims):
#     """
#     Determines if the function is scalar- or vector-valued based on rank evaluation.

#     Parameters
#     ----------
#     input_obj : ppFunction
#         A PorePy ppFunction object
#     dims : int
#         Dimension of problem.

#     Returns
#     -------
#     tag : bool
#         0 if input_function is SCALAR and 1 if input_function is VECTOR.
#     """
#     assert isinstance(input_obj, ppFunction), "input_obj has to be of type ppFunction"

#     obj_dim = input_obj.shape[0] # TODO: modify depending of ppFunction type
#     if obj_dim == 0:
#         return FunctionType.SCALAR
#     elif obj_dim == 2 or obj_dim == 3:
#         assert obj_dim == dims
#         return FunctionType.VECTOR
#     else:
#         raise Exception("Error determining type of given ppFunction")


# TODO: implement
def convert_porepy_to_precice(porepy_function, ids):
    """
    Convert PorePy boundary data arrays into preCICE-compatible numpy array for all x and y coordinates on the boundary.
    Input is already numeric; map array to IDs.
    Parameters
    ----------
    porepy_function : ppFunction
        A PorePy function referring to a physical variable in the problem.
    ids: numpy array
        Array of indices of vertices on the coupling interface.

    Returns
    -------
    precice_data : array_like
        Array of PorePy function values at each point on the boundary.
    """
    #if not isinstance(porepy_function, ppFunction):
    #    raise Exception("Cannot handle data type {}".format(type(porepy_function)))

    precice_data = []

    ##################### FENICS
    if porepy_function.shape[0] > 1: ### maybe
        dims = ... # has to match the main 
        sampled_data = porepy_function.compute_vertex_values().reshape([dims, -1])
    else:
        sampled_data = porepy_function.compute_vertex_values()

    if len(ids):
        if porepy_function.function_space().num_sub_spaces() > 0:  # function space is VectorFunctionSpace
            for id in ids:
                precice_data.append(sampled_data[:, id])
        else:  # function space is FunctionSpace (scalar)
            for id in ids:
                precice_data.append(sampled_data[id])
    else:
        precice_data = np.array([])

    ####################################

    return np.array(precice_data)

def get_vertex_coords(model, coupling_subdomain):
    """
    Extracts vertices which porepy accesses and which lie on the given coupling domain.

    Parameters
    
    model : the pp.Model

    coupling_subdomain : CouplingBoundaryType to select which is the interface (north/south/west/east)

    Returns
    -------
    coords : numpy array. The coordinates of fenics vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    """
    subdomain = model.mdg.subdomains(dim = model.nd)[0]
    domain_sides = model.domain_boundary_sides(subdomain)

    side_map = {
        "n": domain_sides.north,
        "s": domain_sides.south,
        "w": domain_sides.west,
        "e": domain_sides.east,
    }

    
    assert len(coupling_subdomain) != 0, "No boundary specified!"

    coupling_sides = []
    coupling_sides = side_map[coupling_subdomain[0]]
    for c in coupling_subdomain:
        coupling_sides += side_map[c]

    # Here we need to export the coordinates of the coupling boundary to PreCICE.
    
    coords = subdomain.face_centers[:2, coupling_sides].T

    return np.array(coords)


# TODO
def get_coupling_boundary_edges(function_space, coupling_subdomain, ids, id_mapping):
    """
    Extracts edges of mesh which lie on the coupling boundary.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        FEniCS domain of the coupling interface region.
    ids: numpy_array
        Array of global IDs of vertices.
    id_mapping : python dictionary
        Dictionary mapping preCICE vertex IDs to PorePy vertex IDs.

    Returns
    -------
    vertices1_ids : numpy array
        Array of first vertex of each edge.
    vertices2_ids : numpy array
        Array of second vertex of each edge.
    edges_ids : numpy array
        Array of FEniCS edge local IDs.
    """

    def edge_is_on(subdomain, this_edge):
        """
        Check whether edge lies within subdomain
        """
        assert (len(list(vertices(this_edge))) == 2)
        return all([subdomain.inside(v.point(), True) for v in vertices(this_edge)])

    edge_vertices_ids = []
    fenics_edges_ids = []

    for edge in edges(function_space.mesh()):
        if edge_is_on(coupling_subdomain, edge):
            v1, v2 = list(vertices(edge))
            if v1.global_index() in ids and v2.global_index() in ids:
                edge_vertices_ids.append([id_mapping[v1.global_index()], id_mapping[v2.global_index()]])
                fenics_edges_ids.append(edge.index())

    edge_vertices_ids = np.array(edge_vertices_ids)
    fenics_edges_ids = np.array(fenics_edges_ids)

    return edge_vertices_ids, fenics_edges_ids

# TODO
def get_coupling_triangles(function_space, coupling_subdomain, fenics_edge_ids, id_mapping):
    """
    Extracts triangles of mesh which lie on the coupling region.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        FEniCS domain of the coupling interface region.
    fenics_edge_ids: numpy array
        Array with FEniCS IDs of coupling mesh edges

    Returns
    -------
    vertex_ids : numpy array
        Array of indices of vertices which make up triangles (3 per triangle)
    """

    def cell_is_in(subdomain, this_cell):
        """
        Check whether edge lies within subdomain
        """
        assert (len(list(vertices(this_cell))) == 3), "Only triangular meshes are supported"
        return all([subdomain.inside(v.point(), True) for v in vertices(this_cell)])

    vertex_ids = []
    for cell in cells(function_space.mesh()):
        if cell_is_in(coupling_subdomain, cell):
            e1, e2, e3 = list(edges(cell))
            if all(edge_ids in fenics_edge_ids for edge_ids in [e1.index(), e2.index(), e3.index()]):
                v1, v2 = vertices(e1)
                _, v3 = vertices(e2)
                assert (v3 != v1)
                assert (v3 != v2)
                vertex_ids.append([id_mapping[v1.global_index()],
                                   id_mapping[v2.global_index()],
                                   id_mapping[v3.global_index()]])

    return np.array(vertex_ids)
