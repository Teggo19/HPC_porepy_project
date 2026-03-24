import numpy as np
import precice
import logging
from config import Config
import os
from mpi4py import MPI
import copy
import porepy as pp


class Adapter:
    """
    Adapter class for coupling PorePy with PreCICE.
    """
    def __init__(self, adapter_config_filename="porepy-adapter-config.json"):
        """
        Constructor for the Adapter class.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, adapter_config_filename)

        self._config = Config(config_path)
        self._comm = MPI.COMM_WORLD

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size()
        )

        self._precice_vertex_ids = None

    def read_data(self, dt):
        read_data = self._participant.read_data(
                self._config.get_coupling_mesh_name(),
                self._config.get_read_data_name(),
                self._precice_vertex_ids,
                dt)
        
        return 0.5*read_data[1:] + 0.5*read_data[:-1]
        # Here we need to map the read data to the boundary condition for PorePy.
    

    def write_data(self, write_data):
        
        # Here we need to map the data from PorePy to the format required by PreCICE before writing it.
        write_data_processed = np.concatenate([np.zeros(1), 0.5*write_data[1:] + 0.5*write_data[:-1], np.zeros(1)])

        self._participant.write_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_write_data_name(),
            self._precice_vertex_ids,
            write_data_processed)


    def _is_parallel(self):
        return self._comm.Get_size() > 1

    def initialize(self, model):
        # Here we need to initialize the coupling mesh etc.
        domain = model.mdg.subdomains(dim=model.nd)[0]
        # Define mesh in preCICE
        self._precice_vertex_ids = self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(), domain.nodes.T[:, :2]) #(... , Nx2 array of vertex coordinates)
        
        self._precice_vertex_ids = np.unique(domain.face_nodes[:, np.where(model.domain_boundary_sides(domain).north)[0]].nonzero()[0])

        if self._participant.requires_mesh_connectivity_for(self._config.get_coupling_mesh_name()):
            # Define a mapping between coupling vertices and their IDs in preCICE
            #id_mapping = {
            #    key: value for key,
            #    value in zip(
            #        self._owned_vertices.get_global_ids(),
            #        self._precice_vertex_ids)}

            #edge_vertex_ids, fenics_edge_ids = get_coupling_boundary_edges(
            #    function_space, coupling_subdomain, self._owned_vertices.get_global_ids(), id_mapping)
            
            edge_vertex_ids = np.array([domain.face_nodes[:, i].nonzero()[0] for i in range(domain.face_nodes.shape[1])])

            # Surface coupling over 1D edges
            self._participant.set_mesh_edges(self._config.get_coupling_mesh_name(), edge_vertex_ids)

            # Code below does not work properly. Volume coupling does not integrate well with surface coupling in this state. See https://github.com/precice/fenics-adapter/issues/162.
            # # Configure mesh connectivity (triangles from edges) for 2D simulations
            # if self._fenics_dims == 2:
            #     vertices = get_coupling_triangles(function_space, coupling_subdomain, fenics_edge_ids, id_mapping)
            #     self._participant.set_mesh_triangles(self._config.get_coupling_mesh_name(), vertices)
            # else:
            #     print("Mesh connectivity information is not written for 3D cases.")

        self._participant.initialize()

    def advance(self, dt):
        return self._participant.advance(dt)

    def finalize(self):
        self._participant.finalize()

    def get_max_time_step(self):
        return self._participant.get_max_time_step_size()

    def requires_writing_checkpoint(self):
        return self._participant.requires_writing_checkpoint()

    def requires_reading_checkpoint(self):
        return self._participant.requires_reading_checkpoint()
    
    def retrieve_checkpoint(self):
        pass

    def store_checkpoint(self, ):
        pass


    def set_mesh_vertices(self, coord):


        self._precice_vertex_ids=self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(),
            coord,
        )
    
    def is_coupling_ongoing(self):
        return self._participant.is_coupling_ongoing()
    
    def update_coupling_expression(self, coupling_expression, read_data):
        # Here we need to update the coupling expression with the new read data.
        pass

