from dolfin import *
from fenics import *
from fenicsprecice import Adapter

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()
if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()
if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


class LeftBoundary(SubDomain):
    """Determines if the point is at the left boundary with tolerance of 1E-14.
    :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
    """

    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], x_left, tol):
            return True
        else:
            return False

class RightBoundary(SubDomain):
    """Determines if the point is at the right boundaryy with tolerance of
    1E-14.

    :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
    """

    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[0], x_right, tol):
            return True
        else:
            return False

class TopBoundary(SubDomain):
    """Determines if the point is at the top boundary with tolerance of 1E-14.
    :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
    """

    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[1], y_top, tol):
            return True
        else:
            return False

class BottomBoundary(SubDomain):
    """Determines if the point is at the bottom boundary with tolerance of
    1E-14.

    :func inside(): returns True if point belongs to the boundary, otherwise
                    returns False
    """

    def inside(self, x, on_boundary):
        tol = 1E-14
        if on_boundary and near(x[1], y_bottom, tol):
            return True
        else:
            return False

class NoFluxBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and (near(x[0], x_left, tol) 
                                or near(x[0], x_right, tol) 
                                or near(x[1], y_bottom, tol))

########################################
# MESH AND FUNCTION SPACES
y_top = 1
y_bottom = 0
x_left = 0
x_right = 1

mesh_d = RectangleMesh(Point(x_left, y_bottom), Point(x_right, y_top), 16,16)
n = FacetNormal(mesh_d)
nvec = as_vector(n)
tvec = as_vector([-n[1], n[0]])


P = FunctionSpace(mesh_d, "CG", 1)          # pressure: solution and read

Vd = VectorFunctionSpace(mesh_d,"CG",1)     # flow: post-process
Vd_n = Vd.sub(1)                            # flow: write normal component --> write_object

# DATA
mu = Constant(1)
rho = Constant(1000)
f = Constant((0.0, 0.0))
p0 = Constant(0.0)
K  = Constant(1e-6)


#########################
# BOUNDARY CONDITIONS
boundary_left = LeftBoundary()
boundary_right = RightBoundary()
boundary_bottom = BottomBoundary()
coupling_boundary = TopBoundary()
noflux_boundary = NoFluxBoundary()

# PRECISE INITIALIZATION
precice = Adapter(adapter_config_filename="fenics-adapter-config.json")
precice.initialize(coupling_boundary, read_function_space=P, write_object=Vd_n)
# Create a FEniCS Expression to define and control the coupling boundary values
coupling_expression = precice.create_coupling_expression()

# dummy dt to let precice start
dt = precice.get_max_time_step_size()
T = 1.0


# bc_noflux = DirichletBC(P, p0, noflux_boundary)
bc_coupling = DirichletBC(P, coupling_expression, coupling_boundary)
bcs_dir = [bc_coupling]

# Define variational problem
p = TrialFunction(P)
q = TestFunction(P)

a = (K/mu)*inner(grad(p),grad(q))*dx
L = Constant(0)*q*dx

###################################
ph = Function(P)
# TODO: maybe need to initialize to 0 constant, but I think it is the default

# mark mesh w.r.t ranks
ranks = File("../output/ranks%s.pvd.pvd" % precice.get_participant_name())
mesh_rank = MeshFunction("size_t", mesh_d, mesh_d.topology().dim())
mesh_rank.set_all(MPI.rank(MPI.comm_world))
mesh_rank.rename("myRank", "")
ranks << mesh_rank

# Create output file
ufile = File("../output/%s_velocity.pvd" % precice.get_participant_name())
pfile = File("../output/%s_pressure.pvd" % precice.get_participant_name())


while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(ph)

    # read data
    read_data = precice.read_data(dt)
    
    # Update the coupling expression with the new read data
    precice.update_coupling_expression(coupling_expression, read_data)

    # Compute solution
    print("Solving Darcy...")
    solve(a==L,ph,bcs_dir)
    print("...done.")

    # Darcy velocity
    uh = project(-(K/mu)*grad(ph),Vd)   # https://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu
    uh_n = uh.sub(1)

    # write data
    precice.write_data(uh_n)

    precice.advance(dt)

    if precice.requires_reading_checkpoint():  # roll back to checkpoint
        ph_cp, _, _ = precice.retrieve_checkpoint()
        ph.assign(ph_cp)
    else:  
        pass

uh.rename("Darcy Flow", "")
ph.rename("Pressure PM", "")
ufile.write(uh)
pfile.write(ph)