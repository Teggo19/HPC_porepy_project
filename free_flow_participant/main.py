
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


# definition of forms
def nonlinear_form(uh, ph, v, q, neu_f, mu, rho, beta, tvec):
    F = (inner(grad(uh)*uh,v)*dx \
        + mu/rho*inner(grad(uh),grad(v))*dx \
        - 1/rho*div(v)*ph*dx \
        - 1/rho*q*div(uh)*dx \
        + 1/rho*inner(rho/mu*neu_f, v)*ds(1) \
        + 1/rho*beta*inner(dot(uh, tvec)*tvec, dot(v, tvec)*tvec)*ds(2))
        # - inner(f,v)*dx
    return F

def initialization_problem_forms(u,v,p,q,mu,rho, traction):
    # Stokes initialization problem
    a_stokes = 1/rho*mu*inner(grad(u), grad(v))*dx \
        + 1/rho*div(v)*p*dx \
        - 1/rho*q*div(u)*dx
    L_stokes = 1/rho*inner(traction, v)*ds(1)
    return a_stokes, L_stokes

# mesh and function spaces
y_top = 2
y_bottom = 1
x_left = 0
x_right = 1

mesh_f = RectangleMesh(Point(x_left, y_bottom), Point(x_right, y_top), 16,16)
n = FacetNormal(mesh_f)
tvec = as_vector([-n[1], n[0]])

P2 = VectorElement("Lagrange", mesh_f.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh_f.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh_f, TH)       # V x Q

V_n = FunctionSpace(mesh_f, "CG", 1)
Q = W.sub(1).collapse()             # write total stress --> write_object
Qd = FunctionSpace(mesh_f, "DG", 0) # or "CG", depending on needs

# data
mu = Constant(1e-3)
rho = Constant(1e3)
f = Constant((0.0, 0.0))
u_noslip = Constant((0.0, 0.0))    # noslip
traction = Expression(("0", "0.0"), degree=1, t=0)
K  = Constant(1e-6)
alpha_BJS = 1
beta = alpha_BJS*mu/sqrt(K)

neu_f = Expression(("-1e-9", "0.0"), degree=2)   # Neumann b.c. for NS

# boundary conditions
boundary_left = LeftBoundary()
boundary_top = TopBoundary()
coupling_boundary = BottomBoundary()

# preCICE initiailization
precice = Adapter(adapter_config_filename="fenics-adapter-config.json")
precice.initialize(coupling_boundary, read_function_space=V_n, write_object=Q)

# create a FEniCS Expression to define and control the coupling boundary values
coupling_expression = precice.create_coupling_expression()

# assign a dummy dt (needed by preCICE)
dt = precice.get_max_time_step_size()
T = 1.0

bc_top = DirichletBC(W.sub(0), u_noslip, boundary_top)
bc_bottom = DirichletBC(W.sub(0).sub(1), coupling_expression, coupling_boundary)
bc_bottom_stokes = DirichletBC(W.sub(0), u_noslip, coupling_boundary)
bcs_dir = [bc_top, bc_bottom]
bcs_stokes = [bc_top, bc_bottom_stokes]

def interface(x,on_boundary):
    return near(x[1],y_bottom) and on_boundary

def left_f(x,on_boundary):
    return near(x[0],x_left) and on_boundary
boundaries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1)
boundaries.set_all(0)
AutoSubDomain(left_f).mark(boundaries, 1)
AutoSubDomain(interface).mark(boundaries, 2)
ds = Measure("ds", domain=mesh_f, subdomain_data=boundaries)    # to impose Neumann BCs

# define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# initialization with Stokes
wh_init = Function(W)
solver = KrylovSolver("gmres", "ilu")
a_init, L_init = initialization_problem_forms(u,v,p,q,mu,rho,traction)
A_init, b_init = assemble_system(a_init, L_init, bcs_stokes)
solver.set_operator(A_init)
print("Solving linear system for initialization...")
solver.solve(wh_init.vector(), b_init)
print("Initialization done.")
uh_init, ph_init = wh_init.split(deepcopy=True)

wh = Function(W)
(uh, ph) = split(wh)

# initial guess
wh.assign(wh_init)


# mark mesh w.r.t ranks
ranks = File("../output/ranks%s.pvd" % precice.get_participant_name())
mesh_rank = MeshFunction("size_t", mesh_f, mesh_f.topology().dim())
mesh_rank.set_all(MPI.rank(MPI.comm_world))
mesh_rank.rename("myRank", "")
ranks << mesh_rank

# create output file
ufile = File("../output/%s_velocity.pvd" % precice.get_participant_name())
pfile = File("../output/%s_pressure.pvd" % precice.get_participant_name())


while precice.is_coupling_ongoing():

    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(wh)

    # read data
    read_data = precice.read_data(dt)
    
    # Update the coupling expression with the new read data
    precice.update_coupling_expression(coupling_expression, read_data)

    # Compute solution
    F = nonlinear_form(uh, ph, v, q, neu_f, mu, rho, beta, tvec)

    print("Solving NS...")
    solve(F==0,wh,bcs=bcs_dir,
          solver_parameters={"nonlinear_solver":"newton"})
    print("...done.")

    tt = TestFunction(Q)
    traction_normal = Function(Q)
    sigma_f = outer(uh, uh) - mu*(grad(uh)+grad(uh).T) + ph*Identity(2)
    t_form = dot(dot(sigma_f, n), n) * tt * ds(2)
    assemble(t_form, tensor=traction_normal.vector())

    # write data
    precice.write_data(traction_normal)

    precice.advance(dt)

    if precice.requires_reading_checkpoint():  # roll back to checkpoint
        wh_cp, _, _ = precice.retrieve_checkpoint()
        wh.assign(wh_cp)
    else:  
        pass


uh, ph = wh.split(deepcopy=True)
uh.rename("Velocity NS", "")
ph.rename("Pressure NS", "")
ufile.write(uh)
pfile.write(ph)