import numpy as np
import precice
import logging
from HPC_porepy_project.porous_media_participant_yuhe.config import Config
import os
from mpi4py import MPI
import copy
import porepy as pp
from HPC_porepy_project.porous_media_participant_yuhe.porepy_model_yuhe import ProjectModel
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
        ...

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
        )

        self._precice_vertex_ids = None

        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_sigint)


    def _handle_sigint(self, signum, frame):
        print("\nCtrl+C received. Stopping adapter...")
        self._stop_requested = True

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



    def set_mesh_vertices(self, coord):
        self._precice_vertex_ids=self._participant.set_mesh_vertices(
            self._config.get_coupling_mesh_name(),
            coord,
        )

    def make_model(self, north_bc_value: np.ndarray, model_dt=1):
        params = {"north_bc_value": north_bc_value, "model_dt": model_dt}
        model = ProjectModel(params)
        return model