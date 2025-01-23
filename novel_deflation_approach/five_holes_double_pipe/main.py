import sys

sys.path.insert(0, "..")

import topology_optimization_problem as tpo
import deflation_functional as df
import cashocs
from dolfin import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = np.array([[0, 107, 164], [255, 128, 14]]) / 255.0
cmap = LinearSegmentedColormap.from_list("tab10_colorblind", colors, N=256)

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    "./mesh/mesh_deflation.xdmf"
)
config = cashocs.load_config('config.ini')

file_objective = open('./results/objective.txt', 'w')
file_it_total = open('./results/iterations_total.txt', 'w')
file_it_deflation = open('./results/iterations_deflation.txt', 'w')

au = 2.5 / (0.01 ** 2)
al = 2.5 / (100 ** 2)
volume = [0.5, 0.5]
max_it = 250
pen = 1000000.
gamma = 0.7

DG0 = FunctionSpace(mesh, "DG", 0)
CG1 = FunctionSpace(mesh, "CG", 1)
alpha = Function(DG0)
beta = Function(DG0)
psi = Function(CG1)
psi.vector()[:] = -1.0

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P2*P1)
up = Function(W)
u, p = split(up)
vq = Function(W)
v, q = split(vq)
F = inner(grad(u), grad(v)) * dx + alpha * inner(u, v) * dx + div(v) * p * dx + \
    q * div(u) * dx

flow1 = Expression(('-144*(x[1]-1.0/6)*(x[1]-2.0/6)', '0.0'), degree=2)
flow2 = Expression(('-144*(x[1]-4.0/6)*(x[1]-5.0/6)', '0.0'), degree=2)
bc_no_slip = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0, 0)), boundaries, [1])
bc_inflow1 = cashocs.create_dirichlet_bcs(W.sub(0), flow2, boundaries, [2])
bc_inflow2 = cashocs.create_dirichlet_bcs(W.sub(0), flow1, boundaries, [3])
bc_outflow1 = cashocs.create_dirichlet_bcs(W.sub(0), flow2, boundaries, [4])
bc_outflow2 = cashocs.create_dirichlet_bcs(W.sub(0), flow1, boundaries, [5])
bcs = bc_no_slip + bc_inflow1 + bc_inflow2 + bc_outflow1 + bc_outflow2

J = cashocs.IntegralFunctional(inner(grad(u), grad(u)) * dx + alpha * inner(u, u) * dx)
gradient = -(au - al) * (u[0] * (u[0] + v[0]) + u[1] * (u[1] + v[1]))


def save_functions_def(top, it):
    plot(top.phi, vmin=-1e-7, vmax=1e-7, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig("./results/phi/phi_{d}_def.png".format(d=it), bbox_inches='tight')
    plt.close()

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


def save_functions(top, it, It, It_re, val_obj):
    plot(top.phi, vmin=-1e-7, vmax=1e-7, cmap=cmap)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./results/phi/phi_{d}.png".format(d=it), bbox_inches='tight')
    plt.close()

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
    file_objective.write("%.2f\n" % val_obj)


for num_it in range(0, 101):
    J_new = [J]
    gradient_new = [gradient]
    beta_new = [Function(DG0)] * num_it

    for i in range(0, num_it):
        file_beta = HDF5File(
            mesh.mpi_comm(), './results/xdmf/beta_{d}_def.h5'.format(d=i), 'r'
        )
        beta_new[i] = Function(DG0)
        file_beta.read(beta_new[i], 'beta_{d}_def'.format(d=i))
        file_beta.close()

        J_beta = df.DeflationFunctional(
            gamma ** 2, inner(beta - beta_new[i], beta - beta_new[i]) * dx, pen
        )
        J_new.append(J_beta)
        gradient_beta = (1 - 2 * beta_new[i])
        gradient_new.append(gradient_beta)

    deflation = tpo.TopologyOptimizationProblem(
        F, bcs, J_new, gradient_new, up, vq, psi, [alpha, beta], [[al, au], [1, 0]],
        config, volume_constraint=volume, max_it=max_it
    )
    deflation.solve()
    save_functions_def(deflation, num_it)

    test = tpo.TopologyOptimizationProblem(
        F, bcs, J, gradient, up, vq, deflation.phi, [alpha, beta], [[al, au], [1, 0]],
        config, volume_constraint=volume, max_it=max_it
    )
    test.solve()

    val = test.reduced_cost_functional.evaluate()
    if num_it == 0:
        test.It = 0

    save_functions(test, num_it, deflation.It, test.It, val)

    psi.vector()[:] = -1.0

file_it_deflation.close()
file_it_total.close()
file_objective.close()
