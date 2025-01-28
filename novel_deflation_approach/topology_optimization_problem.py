from __future__ import annotations

import copy
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import cashocs
import fenics
import numpy as np
import ufl
import ufl.algorithms

from interpolations import InterpolateLevelSetToElems

from cashocs import _forms
from cashocs import _pde_problems
from cashocs import _utils
from cashocs._optimization import optimization_problem
from cashocs._optimization.optimal_control import optimal_control_problem

if TYPE_CHECKING:
    from cashocs import io
    from cashocs import types


class TopologyOptimizationProblem(optimization_problem.OptimizationProblem):

    def __new__(
            cls,
            state_forms: Union[List[ufl.Form], ufl.Form],
            bcs_list: Union[
                List[List[fenics.DirichletBC]],
                List[fenics.DirichletBC],
                fenics.DirichletBC,
            ],
            cost_functional_form: Union[
                List[types.CostFunctional], types.CostFunctional, List[ufl.Form],
                ufl.Form
            ],
            gradient: Union[List[ufl.Form], ufl.Form],
            states: Union[List[fenics.Function], fenics.Function],
            adjoints: Union[List[fenics.Function], fenics.Function],
            phi: fenics.Function,
            jumping_functions: Union[List[fenics.Function], fenics.Function],
            jumping_coefficients: Union[List[List], List],
            config: Optional[io.Config] = None,
            volume_constraint: Optional[List[float]] = None,
            initial_guess: Optional[List[fenics.Function]] = None,
            reinitialization: Optional[bool] = False,
            maxIt_reinit: Optional[int] = 500,
            reinit_per: Optional[int] = 5,
            max_it: Optional[int] = 25,
            ksp_options: Optional[
                Union[types.KspOptions, List[List[Union[str, int, float]]]]
            ] = None,
            adjoint_ksp_options: Optional[
                Union[types.KspOptions, List[List[Union[str, int, float]]]]
            ] = None,
            scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
            min_max_terms: Optional[Union[List[Dict], Dict]] = None,
            desired_weights: Optional[List[float]] = None,
    ) -> TopologyOptimizationProblem:

        use_scaling = bool(desired_weights is not None)

        if use_scaling:
            unscaled_problem = super().__new__(cls)
            unscaled_problem.__init__(
                state_forms,
                bcs_list,
                cost_functional_form,
                states,
                adjoints,
                config=config,
                initial_guess=initial_guess,
                ksp_options=ksp_options,
                adjoint_ksp_options=adjoint_ksp_options,
                scalar_tracking_forms=scalar_tracking_forms,
                min_max_terms=min_max_terms,
                desired_weights=desired_weights,
            )
            unscaled_problem._scale_cost_functional()

        return super().__new__(cls)

    def __init__(
            self,
            state_forms: Union[List[ufl.Form], ufl.Form],
            bcs_list: Union[
                List[List[fenics.DirichletBC]],
                List[fenics.DirichletBC],
                fenics.DirichletBC,
            ],
            cost_functional_form: Union[
                List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.
                Form
            ],
            gradient: Union[List[ufl.Form], ufl.Form],
            states: Union[List[fenics.Function], fenics.Function],
            adjoints: Union[List[fenics.Function], fenics.Function],
            phi: fenics.Function,
            jumping_functions: Union[List[fenics.Function], fenics.Function],
            jumping_coefficients: Union[List[List], List],
            config: Optional[io.Config] = None,
            volume_constraint: Optional[List[float]] = None,
            initial_guess: Optional[List[fenics.Function]] = None,
            max_it: Optional[int] = 25,
            ksp_options: Optional[
                Union[types.KspOptions, List[List[Union[str, int, float]]]]
            ] = None,
            adjoint_ksp_options: Optional[
                Union[types.KspOptions, List[List[Union[str, int, float]]]]
            ] = None,
            scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
            min_max_terms: Optional[Union[List[Dict], Dict]] = None,
            desired_weights: Optional[List[float]] = None,
    ) -> None:

        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            config,
            initial_guess,
            ksp_options,
            adjoint_ksp_options,
            scalar_tracking_forms,
            min_max_terms,
            desired_weights,
        )

        self.db.parameter_db.problem_type = "topology"
        self.mesh_parametrization = None

        self.phi = phi
        self.V_level = self.phi.function_space()
        self.mesh = self.phi.function_space().mesh()
        self.mesh_cells = self.mesh.cells()
        self.EPS = self.mesh.hmax() * 1e-6
        self.dx = fenics.Measure("dx", self.mesh)

        self.gradient = _utils.enlist(gradient)
        self.top_gradient = fenics.Function(self.V_level)

        self.jumping_functions = _utils.enlist(jumping_functions)
        if len(self.jumping_functions) == 1:
            self.jumping_coefficients = [jumping_coefficients]
        else:
            self.jumping_coefficients = jumping_coefficients

        self.DG0 = fenics.FunctionSpace(self.mesh, 'DG', 0)
        self.Omega = fenics.Function(self.DG0)
        self.dg0_function = fenics.Function(self.DG0)

        self.update_level_set()

        ocp_config = copy.deepcopy(self.config)
        self._base_ocp = optimal_control_problem.OptimalControlProblem(
            self.state_forms,
            self.bcs_list,
            self.cost_functional_list,
            self.states,
            self.phi,
            self.adjoints,
            config=ocp_config,
            initial_guess=initial_guess,
            ksp_options=ksp_options,
            adjoint_ksp_options=adjoint_ksp_options,
            desired_weights=desired_weights,
        )
        self._base_ocp.db.parameter_db.problem_type = "topology"
        self.db.function_db.control_spaces = (
            self._base_ocp.db.function_db.control_spaces
        )
        self.db.function_db.controls = self._base_ocp.db.function_db.controls
        self.form_handler: _forms.ControlFormHandler = self._base_ocp.form_handler
        self.state_problem: _pde_problems.StateProblem = self._base_ocp.state_problem
        self.adjoint_problem: _pde_problems.AdjointProblem = (
            self._base_ocp.adjoint_problem
        )
        self.reduced_cost_functional = self._base_ocp.reduced_cost_functional

        self.volume_constraint: bool = False
        self.equality_constraint: bool = False
        self.inequality_constraint: bool = False
        self.constraint_values: Optional[Union[List[float], float]] = volume_constraint
        if self.constraint_values != None:
            self.volume_constraint = True
            if type(self.constraint_values) == float:
                self.equality_constraint = True
            elif type(self.constraint_values) == list:
                self.inequality_constraint = True

        self.theta = 1.
        self.kappa = 1.
        self.eps_theta = 1.5 * np.pi / 180.
        self.stop = False
        self.tol_bisection = 1e-4
        self.bisection_param = 0.
        self.max_it = max_it
        self.It = 0

    def _erase_pde_memory(self) -> None:
        super()._erase_pde_memory()

    def scalar_product(
            self, a: fenics.Function, b: fenics.Function
    ) -> float:
        return fenics.assemble(a * b * self.dx)

    def norm(
            self, a: fenics.Function
    ) -> float:
        return np.sqrt(self.scalar_product(a, a))

    def normalize(self, a: fenics.Function):
        norm_a = self.norm(a)
        a.vector()[:] /= norm_a

    def interpolate_by_volume(
            self, cell_function: fenics.Function, node_function: fenics.Function
    ) -> None:
        function_space = node_function.function_space()
        test = fenics.TestFunction(function_space)
        
        arr = fenics.assemble(cell_function * test * self.dx)
        vol = fenics.assemble(test * self.dx)
        node_function.vector()[:] = arr[:] / vol[:]

    def compute_gradient(self):
        self.update_level_set()
        self.state_problem.has_solution = False
        self.adjoint_problem.has_solution = False
        self.adjoint_problem.solve()
        grad_loc = 0.
        for i in range(0, len(self.cost_functional_list)):
            if hasattr(self.cost_functional_list[i], 'topological_derivative'):
                grad_loc += (
                    self.cost_functional_list[i].topological_derivative(
                        self.gradient[i])
                )
            else:
                grad_loc += self.gradient[i]

        self.dg0_function.vector()[:] = fenics.project(grad_loc, self.DG0).vector()[:]
        self.interpolate_by_volume(self.dg0_function, self.top_gradient)

    def update_level_set(self):
        vertex_values_phi = self.phi.compute_vertex_values()

        for i in range(0, len(self.jumping_functions)):
            self.jumping_functions[i].vector()[:] = InterpolateLevelSetToElems(
                vertex_values_phi, self.mesh_cells, self.jumping_coefficients[i][0],
                self.jumping_coefficients[i][1], self.EPS
            )

        self.Omega.vector()[:] = InterpolateLevelSetToElems(
            vertex_values_phi, self.mesh_cells, 1, 0, self.EPS
        )

    def compute_theta(self):
        theta_arg = (
            self.scalar_product(self.phi, self.top_gradient) /
            (self.norm(self.phi) * self.norm(self.top_gradient))
        )
        if theta_arg > 1.0:
            theta_arg = 1.0
        elif theta_arg < -1.0:
            theta_arg = -1.0
        self.theta = np.arccos(theta_arg)

    def compute_state(self):
        self.update_level_set()
        self.state_problem.has_solution = False
        self.state_problem.solve()

    def update_shape(self, phi_old: fenics.Function):
        self.phi.vector()[:] = (
                1. / np.sin(self.theta) *
                (np.sin((1. - self.kappa) * self.theta) * phi_old.vector()[:]
                 + np.sin(self.kappa * self.theta) / self.norm(self.top_gradient)
                 * self.top_gradient.vector()[:])
        )
        self.normalize(self.phi)

    def output(self, It, J, vol, ls):
        print('Iteration Number %s' % It)
        print('Function value %.10f' % J)
        print('Volume of fluid %.4f' % vol)
        print('Number of performed line search iterations: %s' % ls)
        print('Step width line search: %s' % self.kappa)
        print('Value of Angle (optimality condition): %.4f' % self.theta)
        print('Parameter to move level-set: %.4f' % self.bisection_param)

    def bisection(self, target):
        omega = fenics.assemble(self.Omega * self.dx)
        max_phi = max(self.phi.vector().max(), -self.phi.vector().min())
        if abs(omega - target) < self.tol_bisection:
            return
        if omega < target:
            lower = -max_phi
            upper = 0.
        else:
            lower = 0.
            upper = max_phi

        self.bisection_recursive(target, lower, upper, 0., omega)

    def bisection_recursive(self, target, lower, upper, c_old, omega):
        if abs(omega - target) < self.tol_bisection:
            self.bisection_param = c_old
            self.normalize(self.phi)
            return

        self.phi.vector()[:] = self.phi.vector()[:] - c_old
        c = (lower + upper) / 2.
        self.phi.vector()[:] = self.phi.vector()[:] + c
        self.update_level_set()
        omega = fenics.assemble(self.Omega * self.dx)
        if omega > target:
            self.bisection_recursive(target, c, upper, c, omega)
        elif omega < target:
            self.bisection_recursive(target, lower, c, c, omega)

    def constraints(self):
        if self.volume_constraint:
            if self.equality_constraint:
                self.bisection(self.constraint_values)
            elif self.inequality_constraint:
                self.update_level_set()
                omega = fenics.assemble(self.Omega * self.dx)
                if omega > self.constraint_values[1]:
                    self.bisection(self.constraint_values[1])
                elif omega < self.constraint_values[0]:
                    self.bisection(self.constraint_values[0])

    def check_optimality_condition(self):
        if self.theta < self.eps_theta:
            self.stop = True
            print('Angle optimality condition is reached\n')
        if self.kappa < 1e-8:
            self.stop = True
            print('Step width too small\n')

    def solve(self) -> None:
        self.It = 0
        self.stop = False
        J = np.zeros(self.max_it)
        ls = 0
        ls_max = 5

        phi_old = fenics.Function(self.V_level)
        phi_old.vector()[:] = self.phi.vector()[:]

        while self.It < self.max_it and self.stop is False:
            self.update_level_set()
            self.compute_state()

            vol_Omega = fenics.assemble(self.Omega * self.dx)
            J[self.It] = self.reduced_cost_functional.evaluate()

            if self.It > 0 and J[self.It] >= J[self.It - 1] and ls < ls_max:
                ls += 1
                self.kappa = self.kappa / 2.
                self.update_shape(phi_old)
                self.constraints()

            else:
                self.compute_gradient()
                if self.norm(self.top_gradient) == 0:
                    print('Norm of topological gradient is zero')
                    return

                phi_old.vector()[:] = self.phi.vector()[:]
                self.compute_theta()

                self.output(self.It, J[self.It], vol_Omega, ls)

                if self.It > 0:
                    self.kappa = min(1, self.kappa * 1.5)
                ls = 0
                self.It += 1
                self.bisection_param = 0.

                self.check_optimality_condition()

                self.update_shape(phi_old)
                self.constraints()

        self.update_level_set()
        self.compute_state()

    def gradient_test(self) -> float:
        raise NotImplementedError(
            "Gradient test is not implemented for topology optimization."
        )
