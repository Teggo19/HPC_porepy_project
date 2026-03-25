import numpy as np
import precice
import logging
from config import Config
import os
from mpi4py import MPI
import copy
import porepy as pp
from porepy_model import ProjectModel
import signal
import sys

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

        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_sigint)


    def _handle_sigint(self, signum, frame):
        print("\nCtrl+C received. Stopping adapter...")
        self._stop_requested = True



    def read_data(self, dt):
        read_data = self._participant.read_data(
                self._config.get_coupling_mesh_name(),
                self._config.get_read_data_name(),
                self._precice_vertex_ids,
                dt)
        
        print(f"Read pressure from PreCICE: {read_data}")
        
        #return 0.5*read_data[1:] + 0.5*read_data[:-1]
        
        # Here we need to map the read data to the boundary condition for PorePy.
        return read_data

    def write_data(self, write_data):
        

        # Here we need to map the data from PorePy to the format required by PreCICE before writing it.
        
        #write_data_processed = np.concatenate([np.zeros(1), 0.5*write_data[1:] + 0.5*write_data[:-1], np.zeros(1)])

        write_data_processed = write_data

        self._participant.write_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_write_data_name(),
            self._precice_vertex_ids,
            write_data_processed)
        
        print(f"Wrote advective flux to PreCICE: {write_data_processed}")


    def _is_parallel(self):
        return self._comm.Get_size() > 1

    def initialize(self):
        # Here we need to initialize the coupling mesh etc.
        tmp_model = self.make_model(north_bc_value=None)
        tmp_model.set_geometry()
        face_centers = tmp_model.export_interface_coordinates()
        # Define mesh in preCICE
        self._precice_vertex_ids = self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(), face_centers) #(... , Nx2 array of vertex coordinates)
        

        if self._participant.requires_mesh_connectivity_for(self._config.get_coupling_mesh_name()):
            # Define a mapping between coupling vertices and their IDs in preCICE
            #id_mapping = {
            #    key: value for key,
            #    value in zip(
            #        self._owned_vertices.get_global_ids(),
            #        self._precice_vertex_ids)}

            #edge_vertex_ids, fenics_edge_ids = get_coupling_boundary_edges(
            #    function_space, coupling_subdomain, self._owned_vertices.get_global_ids(), id_mapping)
            domain = tmp_model.mdg.subdomains(dim=tmp_model.nd)[0]

            north_faces = np.where(tmp_model.domain_boundary_sides(domain).north)[0]

            edge_vertex_ids = np.array([
                domain.face_nodes[:, i].nonzero()[0]
                for i in north_faces
            ])
            

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

    def make_model(self, north_bc_value: np.ndarray):
        params = {"north_bc_value": north_bc_value}
        model = ProjectModel(params)
        return model