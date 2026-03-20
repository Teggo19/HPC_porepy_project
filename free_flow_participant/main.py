
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


# DEFINITIONS OF FORMS
def nonlinear_form(uh, ph, v, q, f, neu_f, beta, tvec):
    F = (inner(grad(uh)*uh,v)*dx
        + mu*inner(grad(uh),grad(v))*dx
        - div(v)*ph*dx
        - q*div(uh)*dx
        - inner(f,v)*dx
        + inner(neu_f, v)*ds(1)
        + beta*inner(dot(uh, tvec)*tvec, dot(v, tvec)*tvec)*ds(2))
    return F

def initialization_problem_forms(u,v,p,q,mu,rho,f, traction):
    # Stokes initialization problem
    a_stokes = mu*inner(grad(u), grad(v))*dx \
        - div(v)*p*dx \
        + q*div(u)*dx
    L_stokes = rho*inner(f, v)*dx \
        + inner(traction, v)*ds(1)
    return a_stokes, L_stokes

########################################
# MESH AND FUNCTION SPACES
y_top = 0
y_bottom = y_top - .25
x_left = 0
x_right = x_left + 1

mesh_f = RectangleMesh(Point(x_left, y_bottom), Point(x_right, y_top), 16,16)
n = FacetNormal(mesh_f)
tvec = as_vector([-n[1], n[0]])

P2 = VectorElement("Lagrange", mesh_f.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh_f.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh_f, TH)     # V x Q

V_n = W.sub(0).sub(1)       # read normal component of velocity --> read_function_space
Q = W.sub(1)                # write total stress --> write_object

# DATA
mu = Constant(1)
rho = Constant(1000)
f = Constant((0.0, 0.0))
u_top = Constant((0.0, 0.0))    # noslip
p_left = Expression("1e-9", degree=2)
traction = Expression(("0", "0.0"), degree=1, t=0)
K  = Constant(1e-6)
alpha_BJS = 1
beta = alpha_BJS*mu/sqrt(K)

neu_f = Expression(("1e-1", "0.0"), degree=2)   # Neumann b.c. for NS

#########################
# BOUNDARY CONDITIONS
boundary_left = LeftBoundary()
boundary_top = TopBoundary()
coupling_boundary = BottomBoundary()

# PRECISE INITIALIZATION
precice = Adapter(adapter_config_filename="fenics-adapter-config.json")
precice.initialize(coupling_boundary, read_function_space=V_n, write_object=Q)
# Create a FEniCS Expression to define and control the coupling boundary values
coupling_expression = precice.create_coupling_expression()

# Appropriate dt
fenics_dt = Constant(0.05)
dt = Constant(0)
precice_dt = precice.get_max_time_step_size()
dt.assign(min([fenics_dt, precice_dt]))
T = 1.0

bc_top = DirichletBC(W.sub(0), u_top, boundary_top)
bc_bottom = DirichletBC(W.sub(0), coupling_expression, coupling_boundary)
bcs_dir = [bc_top, bc_bottom]

boundaries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1)
boundaries.set_all(0)
AutoSubDomain(boundary_left).mark(boundaries, 1)
ds = Measure("ds", domain=mesh_f, subdomain_data=boundaries)    # to impose Neumann BCs

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

########################################
# INITIALIZATION (STOKES PROBLEM)
wh_init = Function(W)
solver = KrylovSolver("gmres", "ilu")
a_init, L_init = initialization_problem_forms(u,v,p,q,mu,rho,f,traction)
A_init, b_init = assemble_system(a_init, L_init, bcs_dir)
solver.set_operator(A_init)
print("Solving linear system for initialization...")
solver.solve(wh_init.vector(), b_init)
print("Initialization done.")
uh_init, ph_init = wh_init.split(deepcopy=True)
uh_init.rename("Velocity", "")   # this name will be used in Paraview
ph_init.rename("Pressure", "")  # this name will be used in Paraview

ufile = File("result/velocity.pvd")
pfile = File("result/pressure.pvd")
ufile.write(uh_init)
pfile.write(ph_init)

###################################
wh = Function(W)
(uh, ph) = split(wh)
uh.rename("velocity_fluid")
ph.rename("pressure_fluid")

# initial guess
wh.assign(wh_init)

# mark mesh w.r.t ranks
ranks = File("output/ranks%s.pvd.pvd" % precice.get_participant_name())
mesh_rank = MeshFunction("size_t", mesh_f, mesh_f.topology().dim())
mesh_rank.set_all(MPI.rank(MPI.comm_world))
mesh_rank.rename("myRank", "")
ranks << mesh_rank
# Create output file
file_out = File("output/%s_velocity.pvd" % precice.get_participant_name())
file_out << (uh, t)
file_out = File("output/%s_pressure.pvd" % precice.get_participant_name())
file_out << (ph, t)

print("output vtk for time = {}".format(float(t)))

# info(LinearVariationalSolver.default_parameters(), True)
# solver.parameters["monitor_convergence"] = False
while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(wh)

    # precice_dt = precice.get_max_time_step_size()
    # dt.assign(min([fenics_dt, precice_dt]))

    # read data
    read_data = precice.read_data(dt)
    
    # Update the coupling expression with the new read data
    precice.update_coupling_expression(coupling_expression, read_data)

    # Compute solution
    F = nonlinear_form(uh, ph, v, q, f, neu_f, beta, tvec)
    print("Solving NS...")
    solve(F==0,wh,bcs=bcs_dir,
          solver_parameters={"nonlinear_solver":"newton"})
    print("...done.")

    # compute total pressure
    n_const = Constant((0.0,1.0))
    sigma_f = outer(uh, uh) - mu*(grad(uh)+grad(uh).T) + ph*Identity(2)
    tension_normal = Function(Q)
    tension_normal = project(dot(n_const, dot(sigma_f, n_const)), Q)
    tension_normal.set_allow_extrapolation(True)

    # write data
    precice.write_data(tension_normal)

    precice.advance(dt(0))

    if precice.requires_reading_checkpoint():  # roll back to checkpoint
        wh_cp = precice.retrieve_checkpoint()
        wh.assign(wh_cp)
    else:  
        pass

uh.rename("Velocity", "")
ph.rename("Pressure", "")
ufile.write(uh)
pfile.write(ph)