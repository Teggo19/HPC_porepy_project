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

# Keep a persistent solver for the coupling loop.
# Calling pp.run_stationary_model() each iteration would call
# prepare_simulation() again and rebuild internal grid-related state.
solver = (
    pp.NewtonSolver({})
    if problem.model._is_nonlinear_problem()
    else pp.LinearSolver({})
)

# precice initialization
precice = Adapter("porepy-adapter-config.json")

precice.initialize(problem.model, model_params["coupling_boundary"], read_function_name = "pressure", write_function_name = "velocity")


# TODO: in FEniCS DirichletBCs are build using create_coupling_expression.
# Since eveything is dynamic, the Dirichlet b.c. is automatically updated
# once the coupling expression is updated. Therefore, we:
# - either ensure that the bc_value used inside the Model class are dynamic
# - or find, inside the loop, a way to update the Model class object
#coupling_expression = precice.create_coupling_expression()


# dummy dt to let precice start
dt = precice.get_max_time_step_size()


while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(problem.model.pressure(problem.model.mdg.subdomains()).value(problem.model.equation_system))
    # read data
    read_data = precice.read_data(dt)

    #precice.update_coupling_expression(coupling_expression, read_data)
    precice.update_bc_conditions(problem.model, read_data)
        
    # Compute solution
    print("Solving Darcy...")
    converged = solver.solve(problem.model)
    if not converged:
        raise RuntimeError("PorePy solver did not converge in coupling iteration")
    #pp.run_stationary_model(problem.model, {})
    print("...done.")

    # # compute the flux
    # advective_flux = problem.model.advective_flux_north_boundary()
    darcy_flux = problem.model.compute_boundary_flux()

    # write data
    precice.write_data(darcy_flux)

    # advance
    precice.advance(dt)

    if precice.requires_reading_checkpoint():
        precice.retrieve_checkpoint()

# model.export_pressure_field()

precice.finalize() 

problem.model.export_darcy_and_pressure("darcy_and_pressure")