import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

def get_darcy_flux(model, ):
    domain = model.mdg.subdomains()[0]
    
    face_fluxes = model.darcy_flux(model.mdg.subdomains()).value(model.equation_system)

    cell_fluxes = np.zeros((domain.num_cells, 2), dtype=float)

    for cell in range(domain.num_cells):
        for face in domain.cell_faces[:, cell].nonzero()[0]:
            normal = domain.face_normals[:2, face]
            flux = face_fluxes[face]
            cell_fluxes[cell] += 0.5 * flux * normal

    return cell_fluxes

def export_darcy_flux(model, file_name):
    darcy_flux = get_darcy_flux(model)
    exporter = pp.Exporter(model.mdg, file_name=file_name, folder_name="darcy_flux_test")
    exporter.write_vtu([(model.mdg.subdomains()[0], "darcy_flux", darcy_flux.T)])


def plot_darcy_flux(model):
    darcy_flux = get_darcy_flux(model)
    plt.figure(figsize=(10, 8))
    # Make a quiver plot of the Darcy flux
    domain = model.mdg.subdomains()[0]
    plt.quiver(domain.cell_centers[0], domain.cell_centers[1], darcy_flux[:, 0], darcy_flux[:, 1])
    plt.title("Darcy Flux")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

