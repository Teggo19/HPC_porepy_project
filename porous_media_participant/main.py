import porepy as pp
import matplotlib.pyplot as plt

from porepyprecice.porepyprecice import Adapter
from ppm_model import *


# definition of the PorePy model
model_params = { "permeability": 1e-6,
           "porosity": 0.4,
           "density": 1e3,
           "viscosity": 1e-6,
           "n_cells": 32,
           "sidelength": 1,
           "grid_type": "cartesian", #"cartesian", "tensor_grid"
           "coupling_boundary": "n", # "nsew", random order is allowed
        }

show_covergence = False

problem = PorousMediaProblem(model_params)
problem.model.prepare_simulation()

# Calling pp.run_stationary_model() each iteration would call
# prepare_simulation() again and rebuild internal grid-related state.
solver = (
    pp.NewtonSolver({})
    if problem.model._is_nonlinear_problem()
    else pp.LinearSolver({})
)

# precice initialization
precice = Adapter("porepy-adapter-config.json")

precice.initialize(
    problem.model,
    model_params["coupling_boundary"],
    read_function_name = "pressure",
    write_function_name = "velocity"
)

# variables to check the convergence of the iterative coupling
window_iteration = 0
pressure_norm_list = []
flux_norm_list = []
prev_pressure = None
prev_flux = None

while precice.is_coupling_ongoing():

    window_iteration += 1
    dt = precice.get_max_time_step_size()
    print(f"[PorousMedia] Coupling iteration {window_iteration}, dt={dt}")

    if precice.requires_writing_checkpoint():
        print("[PorousMedia] Writing checkpoint at start of coupling window")
        precice.store_checkpoint(problem.get_pressure())
    
    # read data
    read_data = precice.read_data(dt)

    # update the coupling conditions in the porepy model with the read data
    precice.update_bcs(problem.model, read_data)
        
    # compute solution
    print("Solving Darcy...")
    converged = solver.solve(problem.model)
    if not converged:
        raise RuntimeError("PorePy solver did not converge in coupling iteration")
    print("...done.")

    # compute the flux (postprocess)
    darcy_flux = problem.model.compute_boundary_flux()

    # write data
    precice.write_data(darcy_flux)

    # advance
    precice.advance(dt)

    # get the pressure
    pressure = read_data
    flux = darcy_flux
    if window_iteration > 1:
        pressure_diff = np.linalg.norm(pressure - prev_pressure)
        flux_diff = np.linalg.norm(flux - prev_flux)
        pressure_norm_list.append(pressure_diff)
        flux_norm_list.append(flux_diff)
        print(f"[PorousMedia] Pressure norm difference to previous iteration: {pressure_diff, flux_diff}")
    prev_pressure = pressure
    prev_flux = flux

    if precice.requires_reading_checkpoint():
        print("[PorousMedia] Window not converged, continuing with next implicit iteration")
        precice.retrieve_checkpoint()
    else:
        print(f"[PorousMedia] Coupling window converged in {window_iteration} iteration(s)")
        window_iteration = 0

precice.finalize() 


problem.model.export_flux_and_pressure()

if show_covergence:
    # make a log plot of the pressure norm differences
    plt.figure()
    plt.plot(pressure_norm_list, label="Interface pressure change")
    plt.plot(flux_norm_list, label="Interface flux change")
    plt.yscale("log")
    plt.xlabel("Coupling iteration")
    plt.ylabel("Norm difference to previous iteration")
    plt.title("Convergence of pressure in porous media participant")
    plt.legend()
    plt.grid()
    plt.show()