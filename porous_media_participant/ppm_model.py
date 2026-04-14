import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from typing import Optional

class ModifiedGeometry:
    """Adding modified geometry to the default model."""
    
    def __init__(self, model_params, *args, **kwargs):

        # read parameters
        self._n_cells = model_params.get("n_cells", 16)
        self._sidelength = model_params.get("sidelength", 1)
        self._cell_size = self._sidelength / self._n_cells

        super().__init__(model_params, *args, **kwargs)


    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength self._sidelength."""
        size = self.units.convert_units(self._sidelength, "m")
        self._domain = nd_cube_domain(2, size) # only dim = 2 is allowed for our implementation
    
    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        Cartesian grid is the default grid type, and we override this method to possibly assign simplex.

        """
        return self.params.get("grid_type", "cartesian")
    
    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        cell_size = self.units.convert_units(self._cell_size, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args
    

class ModifiedBCs(BoundaryConditionsSinglePhaseFlow):
    """Adding modified boundary conditions to the default model."""

    def __init__(self, model_params, *args, **kwargs):

        # store parameters
        self._bc_string = model_params.get("coupling_boundary","n").lower()
        valid = set("nswe")
        if not set(self._bc_string).issubset(valid):
            raise ValueError("coupling_boundary must contain only 'n', 's', 'w', 'e'")
        
        super().__init__(model_params, *args, **kwargs)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the north boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)

        side_map = {
            "n": domain_sides.north,
            "s": domain_sides.south,
            "w": domain_sides.west,
            "e": domain_sides.east,
        }

        # assert len(self._bc_string != 0), "No boundary specified"

        coupling_sides = side_map[self._bc_string[0]]
        for c in self._bc_string[1:]:
            coupling_sides.extend(side_map[c])

        bc = pp.BoundaryCondition(sd, coupling_sides, "dir")

        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Override to set non-homogeneous Neumann boundary conditions."""
        
        values = np.zeros(bg.num_cells)

        # # See section on scaling for explanation of the conversion.
        # # To set non-zero Neumann conditions (example)
        # domain_sides = self.domain_boundary_sides(bg)
        # values[domain_sides.west] = self.units.convert_units(5, "Pa")
        # values[domain_sides.east] = self.units.convert_units(2, "Pa")
        return values


class PostProcessFlux:
    """Adding post-process functionalities to the default model."""

    def compute_boundary_flux(self) -> np.ndarray:
        """Return normalized advective flux on the north boundary as a NumPy array.

        The returned array has the same style as read_data(): a 1D np.ndarray
        containing one scalar value per north boundary interface entry.
        """

        subdomains = self.mdg.subdomains(dim = self.nd)

        flux_values = []

        for sd in subdomains:
            domain_sides = self.domain_boundary_sides(sd)
            side_map = {
                "n": domain_sides.north,
                "s": domain_sides.south,
                "w": domain_sides.west,
                "e": domain_sides.east,
            }
            darcy_flux_time_dependent = self.darcy_flux([sd]).value(self.equation_system)

            coupling_sides = side_map[self._bc_string[0]]
            for c in self._bc_string[1:]:
                coupling_sides.extend(side_map[c])

            sd_flux = darcy_flux_time_dependent[coupling_sides] / sd.face_areas[coupling_sides]

            flux_values.append(sd_flux)

        if len(flux_values) == 0:
            return np.array([], dtype=float)

        return np.concatenate(flux_values).astype(float)
    
    def interpolate_darcy_flux(self):
        """Interpolates the Darcy flux at cell centers."""

        domain = self.mdg.subdomains()[0]
        
        face_fluxes = self.darcy_flux(self.mdg.subdomains()).value(self.equation_system)

        cell_fluxes = np.zeros((domain.num_cells, 2), dtype=float)

        for cell in range(domain.num_cells):
            for face in domain.cell_faces[:, cell].nonzero()[0]:
                normal = domain.face_normals[:2, face]
                flux = face_fluxes[face]
                cell_fluxes[cell] += 0.5 * flux * normal

        return cell_fluxes

    def export_darcy_flux(self, folder = "../output/"):
        """Export the Darcy flux at cell centers."""

        darcy_flux = self.interpolate_darcy_flux()
        exporter = pp.Exporter(self.mdg, "PorousMedia_flux", folder)
        exporter.write_vtu([(self.mdg.subdomains()[0], "darcy_flux", darcy_flux.T)])

    def export_flux_and_pressure(self, folder = "../output/"):
        darcy_flux = self.interpolate_darcy_flux()

        exporter = pp.Exporter(self.mdg, "PorousMedia_flux", folder)
        exporter.write_vtu([(self.mdg.subdomains()[0], "PorousMedia_flux", darcy_flux.T)])

        exporter = pp.Exporter(self.mdg, "PorousMedia_pressure", folder)
        exporter.write_vtu(self.pressure_variable)



class SinglePhaseFlowGeometryBCs(
    ModifiedGeometry,
    ModifiedBCs,
    SinglePhaseFlow,
    PostProcessFlux):
    """The PorePy model constructed using multiple inheritance"""
    ...

class PorousMediaProblem:
    def __init__(self, model_params: Optional[dict] = None):
        
        defaults = {
            "permeability": 1e-6,
            "porosity": 0.4,
            "density": 1e3,
            "viscosity": 1e-6,
        }

        p = {**defaults, **model_params}

        # Build material constants
        solid_constants = pp.SolidConstants(
            permeability = p["permeability"],
            porosity = p["porosity"],
        )

        fluid_constants = pp.FluidComponent(
            viscosity = p["viscosity"],
            density = p["density"]
        )

        material_constants = {
            "fluid": fluid_constants,
            "solid": solid_constants
        }

        model_params["material_constants"] = material_constants

        self.model = SinglePhaseFlowGeometryBCs(model_params)

    def get_pressure(self) -> np.ndarray:
        """Return the pressure solution as a NumPy array."""
        return self.model.pressure(self.model.mdg.subdomains()).value(self.model.equation_system)
    
    ... # other methods