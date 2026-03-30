import numpy as np
import porepy as pp
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from typing import Optional

class PreciceCoupling:
    def __init__(self, params: Optional[dict] = None):

        if params is None:
            params = {}

        # physical parameters
        permeability = params.get("permeability", 1e-6)
        porosity = params.get("porosity", 0.4)

        solid_constants = pp.SolidConstants(
            permeability = permeability,
            porosity = porosity,
        )

        material_constants = {"solid": solid_constants}

        # geometry parameters
        n_cells = params.get("n_cells", [16, 16])
        phys_dim = params.get("phys_dim", [1, 1])

        cell_size_x = phys_dim[0] / n_cells[0]
        cell_size_y = phys_dim[1] / n_cells[1]

        meshing_arguments = {
            "cell_size_x": cell_size_x,
            "cell_size_y": cell_size_y,
        }

        # time manager (dummy)
        self.model_dt = params.get("model_dt", 1)

        time_manager = pp.TimeManager(
            schedule=[0, self.model_dt],
            dt_init=self.model_dt,
            constant_dt=True,
        )

        default_params = {
            "meshing_arguments": meshing_arguments,
            "material_constants": material_constants,
            "time_manager": time_manager,
        }

        params = {**default_params, **params}

        super().__init__(params)

        # optional BC value
        value = params.get("north_bc_value", None)
        self.north_bc_value = None if value is None else np.asarray(value, dtype=float)


    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the north boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom"""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)

        north_mask = domain_sides.north
        num_north = np.sum(north_mask)

        # if self.north_bc_value is None:
        #     raise ValueError("north_bc_value is not set")

        # Ensure correct length
        assert len(self.north_bc_value) == num_north, (
            f"north_bc_value length {len(self.north_bc_value)} "
            f"does not match number of north boundary cells {num_north}"
        )

        if self.north_bc_value is None:
            values[north_mask] = self.units.convert_units(0, "Pa")
        else:
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

    def export_flux_write_nodes_and_edges(self):
        sd = self.mdg.subdomains(dim=self.nd)[0]
        north_faces = np.where(self.domain_boundary_sides(sd).north)[0]

        # PorePy node indices that belong to north boundary faces
        north_node_ids = np.unique(sd.face_nodes[:, north_faces].nonzero()[0])

        coords = np.asarray(sd.nodes[:2, north_node_ids].T, dtype=float)

        # Map PorePy node index -> local write-mesh vertex number
        local_index = {node_id: i for i, node_id in enumerate(north_node_ids)}

        # Each north boundary face is an edge between two nodes
        edges = []
        for f in north_faces:
            face_nodes = sd.face_nodes[:, f].nonzero()[0]
            if len(face_nodes) != 2:
                raise ValueError(f"Expected 2 nodes on boundary face {f}, got {len(face_nodes)}")
            n0, n1 = face_nodes
            edges.append([local_index[n0], local_index[n1]])

        edges = np.asarray(edges, dtype=int)
        return north_node_ids, coords, edges


    def get_normalized_flux_on_write_nodes(self) -> np.ndarray:
        sd = self.mdg.subdomains(dim=self.nd)[0]
        north_faces = np.where(self.domain_boundary_sides(sd).north)[0]

        # Face-centered normalized flux, one value per north face
        face_flux = self.advective_flux_north_boundary()

        # North-boundary nodes in left-to-right order
        north_node_ids, coords, edges = self.export_flux_write_nodes_and_edges()
        x = coords[:, 0]
        order = np.argsort(x)

        coords = coords[order]
        # reorder node ids accordingly
        ordered_nodes = north_node_ids[order]

        # Build node -> adjacent north-face values
        adjacency = {nid: [] for nid in ordered_nodes}
        local_face_nodes = [sd.face_nodes[:, f].nonzero()[0] for f in north_faces]

        for q, face_nodes in zip(face_flux, local_face_nodes):
            for nid in face_nodes:
                if nid in adjacency:
                    adjacency[nid].append(q)

        nodal_flux = np.zeros(len(ordered_nodes), dtype=float)
        for i, nid in enumerate(ordered_nodes):
            vals = adjacency[nid]
            if len(vals) == 0:
                raise ValueError(f"No adjacent north-face flux found for node {nid}")
            nodal_flux[i] = np.mean(vals)

        return nodal_flux

    def export_pressure_field(self):
        step=self.model_dt
        print(f"Exporting pressure field at time {self.model_dt:.2f} with step {step}")
        file_name=f"pressure_field_{int(step):03d}"
        exporter_fixed_name=pp.Exporter(self.mdg,folder_name="output",file_name="pressure_field")

        exporter=pp.Exporter(self.mdg,folder_name="output",file_name=file_name)
        if self.model_dt==1:
            exporter0=pp.Exporter(self.mdg,folder_name="output",file_name="pressure_field_000")
            exporter_fixed_name.write_vtu([self.pressure_variable],time_step=0)
        exporter_fixed_name.write_vtu([self.pressure_variable],time_step=self.model_dt)


class DarcyNorthBCs(
    PreciceCoupling,
    SinglePhaseFlow,
):
    ...