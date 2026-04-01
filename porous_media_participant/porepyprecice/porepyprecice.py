"""
FEniCS - preCICE Adapter. API to help users couple FEniCS with other solvers using the preCICE library.
:raise ImportError: if PRECICE_ROOT is not defined
"""
import numpy as np
from .config import Config
import logging
import precice
from .adapter_core import convert_porepy_to_precice, get_vertex_coords, \
    get_coupling_boundary_edges, CouplingMode, get_coupling_triangles
from .expression_core import RBFInterpolationExpression
from .solverstate import SolverState
import porepy as pp
import copy
import os
from mpi4py import MPI

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Adapter:
    """
    This adapter class provides an interface to the preCICE coupling library for setting up a coupling case
    which has PorePy as a participant for 2D problems.
    """

    def __init__(self, adapter_config_filename="precice-adapter-config.json"):
        """
        Constructor of Adapter class adapted for PorePy.

        Parameters
        ----------
        adapter_config_filename : str
            Name of the JSON adapter configuration file.
        """
        self._config = Config(os.path.relpath(adapter_config_filename))
        
        self._comm = MPI.COMM_WORLD

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size()
        )

        # preCICE vertices corresponding to coupling points
        self._vertex_coords = None
        self._vertex_ids = None

        # interpolation expression (e.g. RBF interpolator)
        self._my_expression = RBFInterpolationExpression  # TODO remove if not needed
        # self._expression_instance = None # not present in the original code

        self._checkpoint = None

        # Store boundary condition object or boundary data container
        self._boundary_condition = None # TODO not implemented yet. This may be an alternative to CouplingExpression

        self._first_advance_done = False

        # type of coupling (UNI- or BI- directional)
        self._coupling_type = None

        # spatial dimension of PorePy problem
        self._porepy_dims = None


    def create_coupling_expression(self):
        """
        Creates a CouplingExpression object used to interpolate data received
        from preCICE onto the PorePy boundary.

        In contrast to the FEniCS adapter, this does not create a UserExpression.
        Instead, it returns a numerical interpolator that can be evaluated at
        arbitrary coordinates.

        Returns
        -------
        coupling_expression : CouplingExpression
            Instance of an interpolation strategy (e.g. RBFInterpolationExpression).
        """

        # Ensure 2D case (current interpolator limitation)
        assert self._porepy_dims == 2, "CouplingExpression currently supports only 2D"

        # Check that read function type was configured

        # Create interpolation expression
        coupling_expression = self._my_expression()

        # TODO: set_function_type

        return coupling_expression
    

    def update_coupling_expression(self, coupling_expression, data):
        """
        Update the coupling interpolator with new data received from preCICE.

        This function must be called at every coupling time step after reading
        data from preCICE. The function updates the interpolator used to map
        point-based coupling data onto the PorePy grid.

        Parameters
        ----------
        coupling_expression : CouplingExpression
            Interpolation object (e.g. RBFInterpolationExpression).

        data : dict
            Coupling data received from preCICE.
            Dictionary mapping coordinates -> values:

                data[(x, y)] = value        # scalar
                data[(x, y)] = [vx, vy]     # vector
        """

        assert self._porepy_dims == 2, "CouplingExpression currently supports only 2D"

        if len(data) == 0:
            raise ValueError("Received empty coupling data from preCICE")

        # Convert dictionary to numpy arrays
        coords = np.array(list(data.keys()), dtype=float)
        vals = np.array(list(data.values()), dtype=float)

        # Update interpolator
        coupling_expression.update_boundary_data(vals, coords)

        return

    def read_data(self, dt):
        """
        Read coupling data from preCICE.
        Depending on the configured function type, the returned data represents
        either scalar or vector values defined at coupling vertices.

        Returns
        -------
        data : dict
            Dictionary mapping vertex coordinates to values:
                data[(x, y)] = value        # scalar
                data[(x, y)] = [vx, vy]     # vector
        """

        assert (
            self._coupling_type is CouplingMode.UNI_DIRECTIONAL_READ_COUPLING
            or self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING
        ), "Adapter is not configured for reading coupling data"

        # Read values from preCICE
        read_values = self._participant.read_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_read_data_name(),
            self._vertex_ids,
            dt,
        )
        return read_values
        
        # Convert to dictionary: coordinate -> value
        read_data = {tuple(coord): value for coord, value in zip(self._vertex_coords, read_values)}

        return read_data # or maybe copy.deepcopy(read_data)
    
    def write_data(self, porepy_function):
        """
        Writes data from a PorePy field to preCICE.

        Parameters
        ----------
        porepy_function : np.array
            PorePy representation of the field that should be written to preCICE.
            Typically this will be a grid-based variable defined on boundary faces
            or nodes depending on the coupling configuration.
        """

        assert (
            self._coupling_type is CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING
            or self._coupling_type is CouplingMode.BI_DIRECTIONAL_COUPLING
        ), "Adapter is not configured for writing coupling data"

        #w_func = porepy_function.copy()
        # making sure that the PorePy function provided by the user is not directly accessed by the Adapter
        #assert (w_func != porepy_function)


        # Convert PorePy representation to preCICE vertex data
        #write_data = convert_porepy_to_precice(
        #    porepy_function,
        #    self._vertex_ids   # TODO: what vertices here? need to implement this method
        #)
        write_data = porepy_function.copy() 
        # Send data to preCICE
        self._participant.write_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_write_data_name(),
            self._vertex_ids,
            write_data
        )


    # def initialize(self, model, coupling_subdomain, coupling_type, write_function = None, fixed_boundary=None):
    def initialize(self, model, coupling_subdomain, read_function_name = None, write_function_name = None, write_function_init = None, fixed_boundary=None):
        """
        Initializes the coupling and sets up the mesh where coupling happens in preCICE.

        Parameters
        ----------
        model : a PorePy Model that implements the porous media problem

        coupling_subdomain : string containig the boundary information "nwes" in any order
            Defines the interface which is the physical coupling boundary.
        
        read_function_name : string. The name of the variable which has to be read from the pp.PorePyModel (e.g. "pressure", "displacement" etc.)
                We may modify this to make some checks
        
        write_function_name : string. The name of the pp:PorePyModel variable which has to be written.
                We may modify this to make some checks
            
        fixed_boundary : Object of class dolfin.fem.bcs.AutoSubDomain # TODO: understand if we need this
            SubDomain consisting of a fixed boundary of the mesh. For example in FSI cases usually the solid body
            is fixed at one end (fixed end of a flexible beam).

        write_init_function : required by require_initial_data

        Returns
        -------
        dt : double
            Recommended time step value from preCICE.
        """

        # Define mesh in preCICE        
        self._vertex_coords = get_vertex_coords(model, coupling_subdomain)
        _, self._porepy_dims = self._vertex_coords.shape

        self._vertex_ids = self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(), self._vertex_coords)

        if self._porepy_dims != self._participant.get_mesh_dimensions(self._config.get_coupling_mesh_name()):
            raise Exception("Dimension of preCICE setup and PorePy do not match")

        # Check what type of coupling
        if read_function_name is not None:
            assert (self._config.get_read_data_name())
        
        if write_function_name is not None:
            assert (self._config.get_write_data_name())          

        if read_function_name is not None and write_function_name is not None:
            self._coupling_type = CouplingMode.BI_DIRECTIONAL_COUPLING
            print("Participant {} is read-and-write participant".format(self._config.get_participant_name()))
        else:
            if write_function_name is not None:
                print("Participant {} is write-only participant".format(self._config.get_participant_name()))
            elif read_function_name is not None:
                print("Participant {} is read-only participant".format(self._config.get_participant_name()))
            else:
                raise ValueError("At least write_function_name or read_function_name needed")

            self._coupling_type = CouplingMode.UNI_DIRECTIONAL_WRITE_COUPLING


        if fixed_boundary: # TODO: check how it works
            self._Dirichlet_Boundary = fixed_boundary

        # TODO
        # Set mesh connectivity information in preCICE to allow nearest-projection mapping
        if self._participant.requires_mesh_connectivity_for(self._config.get_coupling_mesh_name()):
            raise NotImplementedError("requires_mesh_connectivity_for() not allowed in current implementation.")
            # # Define a mapping between coupling vertices and their IDs in preCICE
            # id_mapping = {
            #     key: value for key,
            #     value in zip(
            #         self._vertex_coords.get_ids(),
            #         self._vertex_ids)}

            # # TODO
            # edge_vertex_ids, fenics_edge_ids = get_coupling_boundary_edges(
            #     function_space, coupling_subdomain, self._owned_vertices.get_global_ids(), id_mapping) # TODO

            # # Surface coupling over 1D edges
            # self._participant.set_mesh_edges(self._config.get_coupling_mesh_name(), edge_vertex_ids)

        if self._participant.requires_initial_data():
            if write_function_name is None or write_function_init is None:
                raise Exception(
                    "preCICE requires you to write initial data. Please provide a write_function_init to initialize(...)")
            self.write_data(write_function_init)

        self._participant.initialize()


##############################################
# From here on everything is unchanged

    def store_checkpoint(self, payload, t=None, n=None):
        """
        Defines an object of class SolverState which stores the current state of the variable and the time stamp.

        Parameters
        ----------
        payload : fenics.function or a list of fenics.functions
            Current state of the physical variable(s) of interest for this participant.
        t : double (optional)
            Current simulation time.
        n : int (optional)
            Current time window (iteration) number.
        """
        if self._first_advance_done:
            assert (self.is_time_window_complete())

        logger.debug("Store checkpoint")
        self._checkpoint = SolverState(payload, t, n)

    def retrieve_checkpoint(self):
        """
        Resets the FEniCS participant state to the state of the stored checkpoint.

        Returns
        -------
        u : FEniCS Function
            Current state of the physical variable of interest for this participant.
        t : double
            Current simulation time or None if not specified in store_checkpoint
        n : int
            Current time window (iteration) number or None if not specified in store_checkpoint
        """
        assert (not self.is_time_window_complete())
        logger.debug("Restore solver state")
        return self._checkpoint.get_state()

    def advance(self, dt):
        """
        Advances coupling in preCICE.

        Parameters
        ----------
        dt : double
            Length of timestep used by the solver.

        Notes
        -----
        Refer advance() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx
        """
        self._first_advance_done = True
        self._participant.advance(dt)

    def finalize(self):
        """
        Finalizes the coupling via preCICE and the adapter. To be called at the end of the simulation.

        Notes
        -----
        Refer finalize() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx
        """
        self._participant.finalize()

    def get_participant_name(self):
        """
        Returns
        -------
        participant_name : string
            Name of the participant.
        """
        return self._config.get_participant_name()

    def is_coupling_ongoing(self):
        """
        Checks if the coupled simulation is still ongoing.

        Notes
        -----
        Refer is_coupling_ongoing() in https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if coupling is still going on and False if coupling has finished.
        """
        return self._participant.is_coupling_ongoing()

    def is_time_window_complete(self):
        """
        Tag to check if implicit iteration has converged.

        Notes
        -----
        Refer is_time_window_complete() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if implicit coupling in the time window has converged and False if not converged yet.
        """
        return self._participant.is_time_window_complete()

    def get_max_time_step_size(self):
        """
        Get the maximum time step from preCICE.

        Notes
        -----
        Refer get_max_time_step_size() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        max_dt : double
            Maximum length of timestep to be computed by solver.
        """
        return self._participant.get_max_time_step_size()

    def requires_writing_checkpoint(self):
        """
        Tag to check if checkpoint needs to be written.

        Notes
        -----
        Refer requires_writing_checkpoint() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if checkpoint needs to be written, False otherwise.
        """
        return self._participant.requires_writing_checkpoint()

    def requires_reading_checkpoint(self):
        """
        Tag to check if checkpoint needs to be read.

        Notes
        -----
        Refer requires_reading_checkpoint() in
        https://github.com/precice/python-bindings/blob/develop/cyprecice/cyprecice.pyx

        Returns
        -------
        tag : bool
            True if checkpoint needs to be written, False otherwise.
        """
        return self._participant.requires_reading_checkpoint()
    
    def update_bc_conditions(self, model, read_data):
        def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:

            domain_sides = self.domain_boundary_sides(bg)

            values = np.zeros(bg.num_cells)

            mask = np.array([False] * bg.num_cells)
            side_map = {
                "n": domain_sides.north,
                "s": domain_sides.south,
                "w": domain_sides.west,
                "e": domain_sides.east,
            }
            for c in self._bc_string:
                mask[side_map[c]] = True

            values[mask] = read_data

            return values

        model.bc_values_pressure = bc_values_pressure.__get__(model)
