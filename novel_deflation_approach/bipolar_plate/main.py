import sys

sys.path.insert(0, "..")

import topology_optimization_problem as tpo
import deflation_functional as df
import cashocs
from dolfin import *
import ufl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

file_objective = open('./results/objective.txt', 'w')
file_it_total = open('./results/iterations_total.txt', 'w')
file_it_deflation = open('./results/iterations_deflation.txt', 'w')

colors = np.array([[0, 107, 164], [255, 128, 14]]) / 255.0
cmap = LinearSegmentedColormap.from_list("tab10_colorblind", colors, N=256)

n = 75
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(n, diagonal="crossed")
config = cashocs.load_config('config.ini')

au = 2.5 / (0.0025 ** 2)
al = 2.5 / (100 ** 2)
volume = [0.5, 0.7]
max_it = 500
dt = 1e-3
ud_scalar = Constant(0.01)

DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)
CG2 = FunctionSpace(mesh, 'CG', 2)
alpha = Function(DG0)
beta = Function(DG0)
psi = Function(CG1)
psi.vector()[:] = -1.0

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2 * P1)
up = Function(W)
u, p = split(up)
vq = Function(W)
v, q = split(vq)
u_smooth = Function(W.sub(0).collapse())
v_smooth = Function(W.sub(0).collapse())

F = inner(grad(u), grad(v)) * dx + alpha * inner(u, v) * dx + div(v) * p * dx + \
    q * div(u) * dx
F_smooth = Constant(1. / dt) * inner((u_smooth - u), v_smooth) * dx + \
           inner(grad(u_smooth), grad(v_smooth)) * dx
F_total = [F, F_smooth]

inflow = Expression(("(x[1] >= 0.35 && x[1] <= 0.65) ? -(400./9.)*(x[1]-0.35)*(x[1]-0.65) : 0.0", "0.0"), degree=2)
noslip = Constant((0.0, 0.0))
right = CompiledSubDomain(
    "on_boundary && near(x[0], 1., tol) && (x[1]>=0.65 || x[1]<=0.35)", tol=DOLFIN_EPS
)
right.mark(boundaries, 5)
bc0 = cashocs.create_dirichlet_bcs(W.sub(0), noslip, boundaries, [3, 4])
bc1 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 1)
bc2 = cashocs.create_dirichlet_bcs(W.sub(0), noslip, boundaries, 5)
bcs = bc0 + bc1 + bc2
bcs_smooth = []
bcs_total = [bcs, bcs_smooth]

scale = Constant(100000000)
J = cashocs.IntegralFunctional(
    scale * ufl.Min(Constant(0.0), inner(u_smooth, u_smooth) - ud_scalar) *
    ufl.Min(Constant(0.0), inner(u_smooth, u_smooth) - ud_scalar) * dx
)
gradient = -(au - al) * dot(u, v)


def save_functions_def(top, it):
    file_phi = HDF5File(
        mesh.mpi_comm(), './results/xdmf/phi_{d}_def.h5'.format(d=it), 'w'
    )
    file_phi.write(top.phi, 'phi_{d}_def'.format(d=it))
    file_phi.close()
    file_characteristic = HDF5File(
        mesh.mpi_comm(), './results/xdmf/beta_{d}_def.h5'.format(d=it), 'w'
    )
    file_characteristic.write(beta, 'beta_{d}_def'.format(d=it))
    file_characteristic.close()


def save_functions(top, it, It, It_re, vol_const):
    file_phi = HDF5File(
        mesh.mpi_comm(), './results/xdmf/phi_{d}.h5'.format(d=it), 'w'
    )
    file_phi.write(top.phi, 'phi_{d}'.format(d=it))
    file_phi.close()
    file_characteristic = HDF5File(
        mesh.mpi_comm(), './results/xdmf/beta_{d}.h5'.format(d=it), 'w'
    )
    file_characteristic.write(beta, 'beta_{d}'.format(d=it))
    file_characteristic.close()

    file_it_deflation.write('%d\n' % It)
    file_it_total.write('%d\n' % It_re)
    file_objective.write("%.2f\n" % vol_const)


def constraint_fulfillment(top):
    norm_u = project(inner(top.states[1], top.states[1]), CG2)
    function_target = project(conditional(gt(norm_u, ud_scalar), 1.0, 0.0), DG0)
    vol_fulfilled = assemble(function_target * dx)
    vol_per = 100 * vol_fulfilled / assemble(1 * dx)
    return vol_per


for num_it in range(0, 201):
    J_new = [J]
    gradient_new = [gradient]
    beta_new = [Function(DG0)] * num_it

    for i in range(0, num_it):
        file_beta = HDF5File(
            mesh.mpi_comm(), './results/xdmf/beta_{d}_def.h5'.format(d=i), 'r'
        )
        beta_temp = Function(DG0)
        file_beta.read(beta_temp, 'beta_{d}_def'.format(d=i))
        file_beta.close()

        beta_new[i] = Function(DG0)
        beta_new[i].vector()[:] = project(beta_temp, DG0).vector()[:]

        J_beta = df.DeflationFunctional(
            0.25 ** 2, inner(beta - beta_new[i], beta - beta_new[i]) * dx, 500000.0
        )
        J_new.append(J_beta)
        gradient_beta = (1 - 2 * beta_new[i])
        gradient_new.append(gradient_beta)

    deflation = tpo.TopologyOptimizationProblem(
        F_total, bcs_total, J_new, gradient_new, [up, u_smooth], [vq, v_smooth], psi,
        [alpha, beta], [[al, au], [1, 0]], config, volume_constraint=volume,
        max_it=max_it
    )
    deflation.solve()
    save_functions_def(deflation, num_it)

    test = tpo.TopologyOptimizationProblem(
        F_total, bcs_total, J, gradient, [up, u_smooth], [vq, v_smooth], deflation.phi,
        [alpha, beta], [[al, au], [1, 0]], config, volume_constraint=volume,
        max_it=max_it
    )

    test.solve()

    if num_it == 0:
        test.It = 0

    vol_constraint = constraint_fulfillment(test)
    save_functions(test, num_it, deflation.It, test.It, vol_constraint)

    psi.vector()[:] = -1.0

file_it_deflation.close()
file_it_total.close()
file_objective.close()
