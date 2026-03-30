"""
This module provides a mechanism to interpolate point data acquired from preCICE into PorePy Expressions.
The purpose of this module is to take point-based data from preCICE (for example, forces or displacements
on a boundary) and create an interpolant that can be evaluated at arbitrary points in the domain.

In FEniCS, this was done with UserExpression, which automatically works with the FEM assembly.
In PorePy, we don't have UserExpression, so we need pure numerical interpolators.
The interpolator needs to handle scalar or vector data.
"""

from .adapter_core import FunctionType
from scipy.interpolate import Rbf
import numpy as np
import logging


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CouplingExpression:
    """
    Base class to interpolate nodal/boundary data obtained from preCICE into a continuous functional representation
    suitable for use in PorePy simulations. This class supports both scalar- and vector-valued data and provides
    methods to evaluate the interpolated values at arbitrary points in the domain.

    Attributes
    ----------
    _vals : np.ndarray
        Nodal data from preCICE. Shape (N,) for scalar data or (N, D) for vector data, where N is the number
        of boundary points and D is the vector dimension.
    _coords : np.ndarray
        Coordinates of the boundary points. Shape (N, 2) for 2D data.
    _f : callable or list of callables
        Interpolant(s) generated from `_vals` and `_coords`. A single callable for scalar data, or a list
        of callables (one per component) for vector data.
    _dimension : int
        Spatial dimension of the coordinates (currently only 2D supported).
    _function_type : Enum (FunctionType)
        Indicates whether the data being interpolated is SCALAR or VECTOR.

    Methods
    -------
    update_boundary_data(vals, coords)
        Update the boundary data and recompute the interpolant.

    create_interpolant()
        Construct the interpolant(s) from the current boundary data.

    interpolate(x)
        Evaluate the interpolant at a given point `x`.

    is_scalar_valued()
        Check whether the function being interpolated is scalar-valued.

    is_vector_valued()
        Check whether the function being interpolated is vector-valued.
    """
    def __init__(self):
        self._function_type = None
        self._vals = None
        self._coords = None
        self._dimension = None
        self._f = None

    def set_function_type(self, function_type):
        self._function_type = function_type

    def update_boundary_data(self, vals, coords):
        self._vals = np.asarray(vals)
        self._coords = np.asarray(coords)
        self._dimension = coords.shape[1]
        assert self._dimension == 2, "Only 2D coordinates supported"

        if self.is_scalar_valued():
            assert self._vals.shape[0] == self._coords.shape[0], \
                "Number of scalar values does not match number of points"
        elif self.is_vector_valued():
            assert self._vals.shape[0] == self._coords.shape[0], \
                "Number of vectors does not match number of points"
        assert self._function_type is not None

        self._f = self.create_interpolant()



    def create_interpolant(self):
        raise NotImplementedError("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def interpolate(self, x):
        raise NotImplementedError("Derived class must implement interpolate")

    def is_scalar_valued(self):
        try:
            if self._vals.ndim == 1:
                assert (self._function_type is FunctionType.SCALAR)
                return True
            elif self._vals.ndim > 1:
                assert (self._function_type is FunctionType.VECTOR)
                return False
            else:
                raise Exception("Dimension of the function is 0 or negative!")
        except AttributeError:
            return self._function_type is FunctionType.SCALAR

    def is_vector_valued(self):
        try:
            if self._vals.ndim > 1:
                assert (self._function_type is FunctionType.VECTOR)
                return True
            elif self._vals.ndim == 1:
                assert (self._function_type is FunctionType.SCALAR)
                return False
            else:
                raise Exception("Dimension of the function is 0 or negative!")
        except AttributeError:
            return self._function_type is FunctionType.VECTOR

class RBFInterpolationExpression(CouplingExpression):
    """
    Interpolates boundary data using radial basis functions (RBF). Supports scalar and vector data.
    """
    def create_interpolant(self):
        if self._dimension != 2:
            raise NotImplementedError("Only 2D supported currently")
        
        if self.is_scalar_valued():
            # single callable
            return Rbf(self._coords[:,0], self._coords[:,1], self._vals)
        else:
            # vector: one RBF per component
            funcs = []
            for d in range(self._vals.shape[1]):
                funcs.append(Rbf(self._coords[:,0], self._coords[:,1], self._vals[:,d]))
            return funcs

    def interpolate(self, x):
        if self.is_scalar_valued():
            return self._f(x[0], x[1])  # np.array([self._f(x[0], x[1])])
        else:
            return np.array([f(x[0], x[1]) for f in self._f])
