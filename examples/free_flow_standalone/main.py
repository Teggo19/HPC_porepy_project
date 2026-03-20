
from dolfin import *
from fenics import *

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

# DEFINITIONS OF FORMS
def initialization_problem_forms(u,v,p,q,mu,rho,f, traction):
    # Stokes initialization problem
    a_stokes = mu*inner(grad(u), grad(v))*dx \
        - div(v)*p*dx \
        + q*div(u)*dx
    L_stokes = rho*inner(f, v)*dx \
        + inner(traction, v)*ds(1)
    return a_stokes, L_stokes

def nonlinear_iteration_forms(u,v,p,q,mu,rho,f,traction, u_old):
    # Fixed-point iteration
    a_fp = mu * inner(grad(u), grad(v)) * dx  \
        + rho*dot(dot(grad(u), u_old), v) * dx  \
        - div(v) * p * dx  \
        + q * div(u) * dx
    L_fp = rho*dot(f, v) * dx \
        + inner(traction, v)*ds(1)
    # Newton iteration
    a_Newton = mu * inner(grad(u), grad(v)) * dx \
        + rho*dot(dot(grad(u), u_old), v) * dx \
        + rho*dot(dot(grad(u_old), u), v) * dx \
        - div(v) * p * dx \
        + q * div(u) * dx
    L_Newton = rho*dot(f, v) * dx \
        + rho*dot(dot(grad(u_old), u_old), v) * dx \
        + inner(traction, v)*ds(1)
    
    return a_fp, L_fp
    return a_Newton, L_Newton
def preconditioner_forms(u,v,p,q,mu,rho):
    a_prec = mu * inner(grad(u), grad(v)) * dx + p*q*dx
    return a_prec

########################################
# MESH
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 16,16)

# FUNCTION SPACES
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# DATA
mu = Constant(1)
rho = Constant(1000)
f = Constant((0.0, 0.0))
noslip = Constant((0.0, 0.0))
u_in = Expression(("4*x[1]*(1-x[1])", "0.0"), degree=2)
p_in = Expression("1e-9", degree=2)
traction = Expression(("0", "0.0"), degree=1, t=0)

# BOUNDARY CONDITIONS
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary
def right(x, on_boundary):
    return near(x[0], 1) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], 1) and on_boundary
# Dirichlet boundary conditions
bc_top = DirichletBC(W.sub(0), noslip, top)
bc_bottom = DirichletBC(W.sub(0), noslip, bottom)
bc_left = DirichletBC(W.sub(1), p_in, left)
bc_right = DirichletBC(W.sub(1), 0, right)
# bc_in = DirichletBC(W.sub(0), u_in, left)
bcs = [bc_top, bc_bottom, bc_left, bc_right]

# Neumann boundary conditions
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
AutoSubDomain(left).mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)


# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

a_prec = preconditioner_forms(u,v,p,q,mu,rho)

########################################
# INITIALIZATION (STOKES PROBLEM)
wh_init = Function(W)
solver = KrylovSolver("gmres", "ilu")

# list_lu_solver_methods()
# list_krylov_solver_preconditioners()
# list_krylov_solver_methods()
# solver.parameters["absolute_tolerance"] = 1e-5
# solver.parameters["relative_tolerance"] = 1e-4
# solver.parameters["maximum_iterations"] = 3000
# solver.parameters["monitor_convergence"] = True

a_init, L_init = initialization_problem_forms(u,v,p,q,mu,rho,f,traction)
A_init, b_init = assemble_system(a_init, L_init, bcs)
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

#################################
# NONLINEAR ITERATION (FIXED-POINT OR NEWTON)
u_old = Function(W.sub(0).collapse())   # function to hold the velocity from the previous iteration, for use in the nonlinear iteration forms
u_old.assign(uh_init)    # copy the dof's of uh_init (defined over W) into u_old (defined over V)
                    # Do not use 'u_old = uh_init', which would yield a "shallow copy", by which
                    # u_old would contain a sort of pointer to uh_init: in such case, any update of uh_init
                    # would immediately reflect on u_old, whilst we want to keep the two functions
                    # separate from one another, and update them only when actually intended.
p_old = Function(W.sub(1).collapse())   # function to hold the pressure from the previous iteration, for use in the nonlinear iteration forms
p_old.assign(ph_init)

maxit = 100
it = 0
tol = 1e-3
err = tol+1     # >tol in order to enter the loop at the beginning

wh = Function(W)
# info(LinearVariationalSolver.default_parameters(), True)
# solver.parameters["monitor_convergence"] = False
while it <= maxit and err > tol:
    print("-- iteration ", it)
    a, L = nonlinear_iteration_forms(u, v, p, q, mu, rho, f, traction, u_old)
    A, b = assemble_system(a, L, bcs)
    P, _ = assemble_system(a_prec, L, bcs)
    
    solver.set_operators(A, P)
    print("Solving linear system...")
    solver.solve(wh.vector(), b)
    # solve(a == L, wh, bcs=bcs,
    #           solver_parameters={"linear_solver": "lu", "preconditioner": "ilu"},
    #           form_compiler_parameters={"optimize": True})
    
    uh, ph = wh.split(deepcopy=True)
    it += 1

    err = (errornorm(uh, u_old, 'H1') / norm(u_old, 'H1') +
           errornorm(ph, p_old, 'L2') / norm(p_old, 'L2'))

    print("Iteration = ", it, " Error = ", err)

    u_old.assign(uh) # update the old solution
    # u_old = uh #NO
    p_old.assign(ph) # update the old solution

if it <= maxit:
    print('Nonlinear solver converged in', it, 'iterations.')
    uh.rename("Velocity", "")
    ph.rename("Pressure", "")
    ufile.write(uh)
    pfile.write(ph)
else:
    print('Nonlinear solver di NOT converge!\nRelative error =', err, 'after', it, 'iterations.')
