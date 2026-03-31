import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
from typing import Optional

class ModifiedGeometry:
    
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
    """Adding both geometry and modified boundary conditions to the default model."""
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
            # coupling_sides += side_map[c]
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

    
class SinglePhaseFlowGeometryBCs(
    ModifiedGeometry,
    ModifiedBCs,
    SinglePhaseFlow):
    ...

class PorousMediaProblem:
    def __init__(self, model_params: Optional[dict] = None):
        
        defaults = {
            "permeability": 1e-6,
            "porosity": 0.4,
            "density": 1e3,
            "viscosity": 1e-3,
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
    
    ... # other methods