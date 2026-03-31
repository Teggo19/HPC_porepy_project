import porepy as pp
import precice as pc
from porepyprecice.porepyprecice import Adapter
from porepyprecice.adapter_core import CouplingBoundaryType as cb_type
from ppm_model import *


model_params = { "permeability": 1e-6,
           "porosity": 0.4,
           "n_cells": 16,
           "sidelength": 1,
           "grid_type": "simplex", #"cartesian", "tensor_grid"
           "coupling_boundary": "n", # "nsew", random order is allowed
           "coupling_value": ... # TODO (possible write_data if needed)
        }

problem = PorousMediaProblem(model_params)
problem.model.prepare_simulation()
# precice initialization
precice = Adapter("porepy-adapter-config.json")

# precice.initialize(model, model_params["coupling_boundary"], "rw")
precice.initialize(problem.model, model_params["coupling_boundary"], read_function_name = "pressure", write_function_name = "velocity")

coupling_expression = precice.create_coupling_expression()

# TODO: in FEniCS DirichletBCs are build using create_coupling_expression.
# Since eveything is dynamic, the Dirichlet b.c. is automatically updated
# once the coupling expression is updated. Therefore, we:
# - either ensure that the bc_value used inside the Model class are dynamic
# - or find, inside the loop, a way to update the Model class object

# dummy dt to let precice start
dt = precice.get_max_time_step()


while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint()

    # read data
    read_data = precice.read_data(dt)

    precice.update_coupling_expression(coupling_expression, read_data)
        
    # Compute solution
    print("Solving Darcy...")
    pp.run_stationary_model(problem.model)
    print("...done.")

    # # compute the flux
    # advective_flux = problem.model.advective_flux_north_boundary()

    # write data
    precice.write_data(None)

    # advance
    precice.advance(dt)

    if precice.requires_reading_checkpoint():
        precice.retrieve_checkpoint()

# model.export_pressure_field()

precice.finalize()    