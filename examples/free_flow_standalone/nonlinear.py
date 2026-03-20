from dolfin import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False

# Define variational forms (steady Navier-Stokes)
def non_linear_forms(uh,v,ph,q,nu,f,traction):
     F = (inner(grad(uh)*uh, v)*dx  # convection term (nonlinear)
     + nu*inner(grad(uh), grad(v))*dx
     - div(v)*ph*dx
     - q*div(uh)*dx
     - inner(f, v)*dx + inner(traction, v)*ds(1))
     return F

# Create mesh
mesh = RectangleMesh(Point(0, 1), Point(1, 2), 16, 16)

# Define function spaces (P2-P1)
# Build function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Parameters
nu = 1e-6
f = Constant((0, 0))

# Boundary conditions
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

def right(x, on_boundary):
    return near(x[0], 1) and on_boundary

def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary

def top(x, on_boundary):
    return near(x[1], 1) and on_boundary

# Neumann boundary conditions
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
AutoSubDomain(left).mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

bc_top  = DirichletBC(W.sub(0), (0, 0), top)
bc_bottom = DirichletBC(W.sub(0), (0, 0), bottom) # this has to take the value from the porous media result
inflow  = DirichletBC(W.sub(1), 1e-9, left)  # steady inflow pressure
outflow = DirichletBC(W.sub(1), 0, right)

traction = Expression(("1e-9", "0.0"), degree=2)

bc = [bc_top, bc_bottom]#, inflow, outflow]
wh = Function(W)  # combined function for velocity and pressure
uh, ph = split(wh)
v, q = TestFunctions(W)

F = non_linear_forms(uh,v,ph,q,nu,f,traction)

params = {"nonlinear_solver": "newton"}
# Solve nonlinear system
solve(F == 0, wh, bcs=bc,
      solver_parameters=params)
uh, ph = wh.split()

# Save results
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
ufile << uh
pfile << ph

# params = {"nonlinear_solver": "newton",
#           "newton_solver": {
#             "absolute_tolerance": 1e-8,
#             "relative_tolerance": 1e-6,
#             "maximum_iterations": 25,
#             "linear_solver": "bicgstab",
#             "preconditioner": "ilu",
#             "report": True,
#             "error_on_nonconvergence": True,
#             "relaxation_parameter": 1.0}
#         }

# params = {"nonlinear_solver": "newton",
#           "newton_solver": {
#             "linear_solver": "bicgstab",
#             "preconditioner": "amg",
#             "report": True,
#             "error_on_nonconvergence": True,}
#          }