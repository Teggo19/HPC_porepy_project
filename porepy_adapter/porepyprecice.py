import numpy as np
import precice
import logging
from .config import Config
import os
from mpi4py import MPI
import copy


class Adapter:
    """
    Adapter class for coupling PorePy with PreCICE.
    """
    def __init__(self, adapter_config_filename="porepy-adapter-config.json"):
        """
        Constructor for the Adapter class.
        """
        self._config = Config(os.path.relpath(adapter_config_filename))
        self._comm = MPI.COMM_WORLD

        self._participant = precice.Participant(
            self._config.get_participant_name(),
            self._config.get_config_file_name(),
            self._comm.Get_rank(),
            self._comm.Get_size()
        )

    def read_data(self, dt):
        read_data = self._participant.read_data(
                self._config.get_coupling_mesh_name(),
                self._config.get_read_data_name(),
                self._precice_vertex_ids,
                dt)
        
        # Here we need to map the read data to the boundary condition for PorePy.
    

    def write_data(self, write_data):

        # Here we need to map the data from PorePy to the format required by PreCICE before writing it.

        self._participant.write_data(
            self._config.get_coupling_mesh_name(),
            self._config.get_write_data_name(),
            self._precice_vertex_ids,
            write_data)


    def _is_parallel(self):
        return self._comm.Get_size() > 1

    def initialize(self,):
        # Here we need to initialize the coupling mesh etc.

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


