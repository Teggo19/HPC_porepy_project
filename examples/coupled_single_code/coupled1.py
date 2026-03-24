from dolfin import *

parameters["std_out_all_processes"] = False

# MESHES
mesh_d = RectangleMesh(Point(0,0), Point(1,1), 16,16)
mesh_f = RectangleMesh(Point(0,1), Point(1,2), 16,16)

# BOUNDARY INFORMATION (for imposing bcs)
n = FacetNormal(mesh_f)
tvec = as_vector([-n[1], n[0]])
def interface(x,on_boundary):
    return near(x[1],1) and on_boundary

def left_f(x,on_boundary):
    return near(x[0],0) and on_boundary

boundaries = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1)
boundaries.set_all(0)
AutoSubDomain(left_f).mark(boundaries, 1) # pressure driven
AutoSubDomain(interface).mark(boundaries, 2) # BJS
ds = Measure("ds", domain=mesh_f, subdomain_data=boundaries)

# FUNCTION SPACES
P = FunctionSpace(mesh_d, "CG", 1)          # Darcy

P2 = VectorElement("Lagrange", mesh_f.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh_f.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh_f, TH)               # NS

Vf = VectorFunctionSpace(mesh_f,"CG",2)     # For transfer
Vd = VectorFunctionSpace(mesh_d,"CG",1)
Pf = FunctionSpace(mesh_f, "CG", 1)

# PARAMETERS
max_iter = 20
tol = 1e-6

K  = Constant(1e-6)
mu = Constant(1e-3)
rho = Constant(1e3)
alpha_BJS = 1
beta = alpha_BJS*mu/sqrt(K)
f = Constant((0,0))

neu_f = Expression(("1e-1", "0.0"), degree=2)   # Neumann b.c. for NS


# ITERATIVE LOOP
p_interface = Function(P)
p_interface.assign(Constant(1000.0))    # initial guess

p_old = Function(P)     # for error computation
p_old.assign(Constant(0.))

for k in range(max_iter):
    print("ITERATION ",k)

########################################
# 1.DARCY
    p_d = TrialFunction(P)
    q_d = TestFunction(P)

    a = (K/mu)*inner(grad(p_d),grad(q_d))*dx
    L = Constant(0)*q_d*dx

    bc_darcy = DirichletBC(P,p_interface,interface)

    ph_d = Function(P)
    print("Solving Darcy...")
    solve(a==L,ph_d,bc_darcy)
    print("...done.")

    # Darcy velocity
    Puh_d = project(-(K/mu)*grad(ph_d),Vd)
    Puh_d.set_allow_extrapolation(True)

####################################
# 2. TRANSFER VELOCITY: Darcy --> NS
    print("Transfering velocity")
    Vfn = FunctionSpace(mesh_f, "CG", 1)
    u_d_n = project(-Puh_d[1], Vfn) # TODO: modify this to account for n
    u_d_n.set_allow_extrapolation(True)

    print("...done.")

###################################
# 3. NAVIER-STOKES
    wh = Function(W)
    uh_f,ph_f = split(wh)

    v_f,q_f = TestFunctions(W)

    F = (inner(grad(uh_f)*uh_f,v_f)*dx
         + mu*inner(grad(uh_f),grad(v_f))*dx
         - div(v_f)*ph_f*dx
         - q_f*div(uh_f)*dx
         - inner(f,v_f)*dx
         + inner(neu_f, v_f)*ds(1)
         + beta*inner(dot(uh_f, tvec)*tvec, dot(v_f, tvec)*tvec)*ds(2))

    bcu_top = DirichletBC(W.sub(0),Constant((0,0)), "near(x[1],2) && on_boundary")
    bcu_interface_n = DirichletBC(W.sub(0).sub(1), u_d_n, interface)
    # bcu_freeslip_t = DirichletBC(W.sub(0).sub(0), u_d_t, interface_f)
    
    print("Solving NS...")
    solve(F==0,wh,bcs=[bcu_interface_n, bcu_top],
          solver_parameters={"nonlinear_solver":"newton"})
    print("...done.")
    uh_f,ph_f = wh.split()

###################################
# 4. TRANSFER FORCE: NS --> Darcy
    print("Transfering force...")
    
    n_const = Constant((0.0,1.0))
    sigma_f = outer(uh_f, uh_f) - mu*(grad(uh_f)+grad(uh_f).T) + ph_f*Identity(2)
    tension_normal = Function(P)
    tension_normal = project(dot(n_const, dot(sigma_f, n_const)), Vfn)
    tension_normal.set_allow_extrapolation(True)

    p_interface = project(tension_normal, P) # TODO: modify this to account for n
    p_interface.set_allow_extrapolation(True)


    print("...done.")

    # convergence check
    
    error = errornorm(p_interface, p_old, "L2")
    print("error =", error)

    p_old.assign(p_interface)

    if error < tol:
        print("Coupling converged")
        break


# save results
File("results/darcy_pressure.pvd") << ph_d
File("results/darcy_flow.pvd") << Puh_d
File("results/ns_velocity.pvd") << uh_f
File("results/ns_pressure.pvd") << ph_f