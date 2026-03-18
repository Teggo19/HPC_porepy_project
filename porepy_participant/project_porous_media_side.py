import numpy as np
import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow, BoundaryConditionsSinglePhaseFlow


class ModifiedBC:
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the north boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        # See section on scaling for explanation of the conversion.
        xmax = self.domain.bounding_box["xmax"]
        north_values = xmax-bg.cell_centers[0]
        values[domain_sides.north] = north_values[domain_sides.north]
        return values
    
    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:

        return self.bc_type_darcy_flux(sd)

class BoundaryAdvectiveFluxExport:
    def data_to_export(self):
        data=super().data_to_export()
        subdomains=self.mdg.subdomains(dim=self.nd)
        for sd in subdomains:
            darcy_flux_time_dependent=self.darcy_flux([sd]).value(self.equation_system)
            #taking the cells close to the north boundary
            domain_sides=self.domain_boundary_sides(sd)
            north_cells=domain_sides.north
            num_north_cells=np.sum(north_cells)
            values=np.zeros(sd.num_cells)
            values[-num_north_cells:]=darcy_flux_time_dependent[north_cells]/sd.face_areas[north_cells]
            data.append(

                (sd,"advective_flux_north_boundary",values)
            )
        return data


class ProjectModel(
    BoundaryAdvectiveFluxExport,
    ModifiedBC, SinglePhaseFlow
):
    ...

if __name__ == "__main__":
    solid_constants = pp.SolidConstants(permeability=1e-6, porosity=0.4)

    material_constants = {"solid": solid_constants}


    model_params={
        "meshing_arguments": {"cell_size_x": 0.025,
                                    "cell_size_y": 0.025},
        "material_constants": material_constants,
    }
    model=ProjectModel(model_params)
    pp.run_time_dependent_model(model)
    pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
#    pp.plot_grid(model.mdg, "advective_flux_north_boundary", figsize=(10, 8), plot_2d=True)