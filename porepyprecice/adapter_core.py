"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

from fenics import SubDomain, Point, PointSource, vertices, FunctionSpace, Function, edges, cells
import numpy as np
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# TODO: which type? possibly a numpy array or similar
# need python 3.12. Possibly add interpreter checks
type ppFunction = np.array # modify

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


# TODO: implementation needed
def determine_function_type(input_obj, dims):
    """
    Determines if the function is scalar- or vector-valued based on rank evaluation.

    Parameters
    ----------
    input_obj : ppFunction
        A PorePy ppFunction object
    dims : int
        Dimension of problem.

    Returns
    -------
    tag : bool
        0 if input_function is SCALAR and 1 if input_function is VECTOR.
    """
    assert isinstance(input_obj, ppFunction), "input_obj has to be of type ppFunction"
    obj_dim = input_obj.shape[0] # TODO: modify depending of ppFunction type
    if obj_dim == 0:
        return FunctionType.SCALAR
    elif obj_dim == 2 or obj_dim == 3:
        assert obj_dim == dims
        return FunctionType.VECTOR
    else:
        raise Exception("Error determining type of given ppFunction")


# TODO: implement
def convert_porepy_to_precice(porepy_function, local_ids):
    """
    Convert PorePy boundary data arrays into preCICE-compatible numpy array for all x and y coordinates on the boundary.
    Input is already numeric; map array to IDs.
    Parameters
    ----------
    porepy_function : ppFunction
        A PorePy function referring to a physical variable in the problem.
    local_ids: numpy array
        Array of indices of vertices on the coupling interface.

    Returns
    -------
    precice_data : array_like
        Array of PorePy function values at each point on the boundary.
    """


    if not isinstance(porepy_function, ppFunction):
        raise Exception("Cannot handle data type {}".format(type(porepy_function)))

    precice_data = []

    if porepy_function.function_space().num_sub_spaces() > 0:
        dims = porepy_function.function_space().num_sub_spaces()
        sampled_data = porepy_function.compute_vertex_values().reshape([dims, -1])
    else:
        sampled_data = porepy_function.compute_vertex_values()

    if len(local_ids):
        if porepy_function.function_space().num_sub_spaces() > 0:  # function space is VectorFunctionSpace
            for lid in local_ids:
                precice_data.append(sampled_data[:, lid])
        else:  # function space is FunctionSpace (scalar)
            for lid in local_ids:
                precice_data.append(sampled_data[lid])
    else:
        precice_data = np.array([])

    return np.array(precice_data)


# TODO
def get_porepy_vertices(function_space, coupling_subdomain, dims):
    """
    Extracts vertices which porepy accesses and which lie on the given coupling domain, from a given
    function space.

    Parameters
    ----------
    function_space : FEniCS function space
        Function space on which the finite element problem definition lives.
    coupling_subdomain : FEniCS Domain
        Subdomain consists of only the coupling interface region.
    dims : int
        Dimension of problem.

    Returns
    -------
    lids : numpy array
        Array of local ids of fenics vertices.
    gids : numpy array
        Array of global ids of fenics vertices.
    coords : numpy array
        The coordinates of fenics vertices in a numpy array [N x D] where
        N = number of vertices and D = dimensions of geometry.
    """

    if not issubclass(type(coupling_subdomain), SubDomain):
        raise Exception("No correct coupling interface defined! Given coupling domain is not of type dolfin Subdomain")

    # Get mesh from FEniCS function space
    mesh = function_space.mesh()

    # Get coordinates and global IDs of all vertices of the mesh  which lie on the coupling boundary.
    # These vertices include shared (owned + unowned) and non-shared vertices in a parallel setting
    lids, gids, coords = [], [], []
    for v in vertices(mesh):
        if coupling_subdomain.inside(v.point(), True):
            lids.append(v.index())
            gids.append(v.global_index())
            coords.append([v.x(d) for d in range(dims)])

    return np.array(lids), np.array(gids), np.array(coords)

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
