import fenics
import ufl
from typing import Union
import cashocs

class DeflationFunctional(cashocs._optimization.cost_functional.Functional):
    def __init__(self,
                 gamma: Union[float, int],
                 distance: ufl.Form,
                 weight: Union[float, int] = 1.0):

        super().__init__()
        self.distance = distance
        self.gamma = fenics.Constant(gamma)

        mesh = self.distance.integrals()[0].ufl_domain().ufl_cargo()
        R = fenics.FunctionSpace(mesh, "R", 0)
        self.distance_value = fenics.Function(R)
        self.weight = fenics.Function(R)
        self.weight.vector().vec().set(weight)
        self.weight.vector().apply("")

    def coefficients(self):
        return self.distance.coefficients()

    def derivative(self, argument, direction):

        deriv = fenics.derivative(
            fenics.conditional(
                fenics.lt(self.distance_value, self.gamma),
                self.weight * fenics.exp(self.gamma / (self.distance_value - self.gamma))
                * (-self.gamma / ((self.distance_value - self.gamma) ** 2)),
                0.
            ) * self.distance,
            argument,
            direction,
        )

        return deriv

    def evaluate(self):
        dist_value = fenics.assemble(self.distance)
        self.distance_value.vector().vec().set(dist_value)
        self.distance_value.vector().apply("")

        if dist_value < self.gamma.values()[0]:
            val = (
                    self.weight.vector().vec().sum()
                        * fenics.exp(self.gamma.values().sum() /
                                     (self.distance_value.vector().sum() - self.gamma.values().sum()))
            )
        else:
            val = 0.

        return val

    def scale(self, scaling_factor):
        self.weight.vector().vec().set(scaling_factor)
        self.weight.vector().apply("")

    def update(self):
        dist_value = fenics.assemble(self.distance)
        self.distance_value.vector().vec().set(dist_value)
        self.distance_value.vector().apply("")

    def topological_derivative(self, form_grad: ufl.Form) -> ufl.Form:
        top_gradient = fenics.conditional(
            fenics.lt(self.distance_value, self.gamma),
            -self.weight * self.gamma / ((self.distance_value - self.gamma) ** 2) * fenics.exp(
                self.gamma / (self.distance_value - self.gamma)) * form_grad,
            0. * form_grad
        )

        return top_gradient