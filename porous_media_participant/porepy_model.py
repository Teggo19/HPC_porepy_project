import numpy as np
import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from typing import Optional

class PreciceCoupling:


    def __init__(self, params: Optional[dict] = None):

        solid_constants = pp.SolidConstants(permeability=1e-6, porosity=0.4)

        material_constants = {"solid": solid_constants}


        default_params={
            "meshing_arguments": {"cell_size_x": 0.0625,
                                        "cell_size_y": 0.0625},
            "material_constants": material_constants,
        }

        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}

        super().__init__(params)



        value = params.get("north_bc_value")

        self.north_bc_value = None if value is None else np.asarray(value, dtype=float)



    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the north boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)

        north_mask = domain_sides.north
        num_north = np.sum(north_mask)

        if self.north_bc_value is None:
            raise ValueError("north_bc_value is not set")


        # Ensure correct length
        assert len(self.north_bc_value) == num_north, (
            f"north_bc_value length {len(self.north_bc_value)} "
            f"does not match number of north boundary cells {num_north}"
        )
        values[north_mask] = self.north_bc_value

        return values        
    
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:

        return self.bc_type_darcy_flux(sd)
        
    def advective_flux_north_boundary(self) -> np.ndarray:
        """Return normalized advective flux on the north boundary as a NumPy array.

        The returned array has the same style as read_data(): a 1D np.ndarray
        containing one scalar value per north boundary interface entry.
        """

        subdomains = self.mdg.subdomains(dim=self.nd)
        flux_values = []

        for sd in subdomains:
            darcy_flux_time_dependent = self.darcy_flux([sd]).value(self.equation_system)

            domain_sides = self.domain_boundary_sides(sd)
            north_faces = domain_sides.north

            sd_flux = darcy_flux_time_dependent[north_faces] / sd.face_areas[north_faces]
            flux_values.append(sd_flux)

        if len(flux_values) == 0:
            return np.array([], dtype=float)

        return np.concatenate(flux_values).astype(float)

    def export_interface_coordinates(self):
        # Here we need to export the coordinates of the coupling boundary to PreCICE.
        subdomain = self.mdg.subdomains(dim=self.nd)[0]
        coupling_boundary = self.domain_boundary_sides(subdomain).north
        coordinates = subdomain.face_centers[:2, coupling_boundary].T

        return coordinates
    
class ProjectModel(
    PreciceCoupling,
    SinglePhaseFlow,
):
    ...