"""
This module consists of helper functions used in the Adapter class. Names of the functions are self explanatory
"""

import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# need python 3.12. Possibly add interpreter checks
type ppFunction = np.array

class CouplingBoundaryType(Enum):
    """
    Defines on which boundary we want the coupling.
    Used in initialization to extract vertex IDs and coords on the coupling boundary.
    """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class CouplingMode(Enum):
    """
    Defines the type of coupling being used.
    Options are: Bi-directional coupling, Uni-directional Write Coupling, Uni-directional Read Coupling
    Used in assertions to check which type of coupling is done
    """
    BI_DIRECTIONAL_COUPLING = 4
    UNI_DIRECTIONAL_WRITE_COUPLING = 5
    UNI_DIRECTIONAL_READ_COUPLING = 6


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

    coords = subdomain.face_centers[:2, coupling_sides].T

    return np.array(coords)


def get_coupling_boundary_edges():
    """Extracts edges of mesh which lie on the coupling boundary."""

    raise NotImplementedError("This function is not implemented yet. It needs to be adapted to the new way of defining the coupling boundary") 

def get_coupling_triangles(function_space, coupling_subdomain, fenics_edge_ids, id_mapping):
    """
    Extracts triangles of mesh which lie on the coupling region.

    Returns
    -------
    vertex_ids : numpy array
        Array of indices of vertices which make up triangles (3 per triangle)
    """

    raise NotImplementedError("This function is not implemented yet. It needs to be adapted to the new way of defining the coupling boundary")
