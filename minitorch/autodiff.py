from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_lst = list(vals)

    vals_lst[arg] += epsilon
    forward = f(*vals_lst)

    vals_lst[arg] -= 2 * epsilon
    backward = f(*vals_lst)

    derivative = (forward - backward) / (2 * epsilon)
    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for the node."""
        ...

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Check if the value is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the gradients.

        Args:
        ----
            d_output (Any): The derivative of the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    traversal = []

    def dfs(variable: Variable) -> None:
        if variable.is_constant() or variable.unique_id in visited:
            return
        visited.add(variable.unique_id)
        for parent in variable.parents:
            dfs(parent)
        traversal.append(variable)

    dfs(variable)
    return reversed(traversal)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable (Variable): The variable to backpropagate from.
        deriv (Any): The derivative to propagate.

    """
    topo_order = list(topological_sort(variable))
    gradient_map = {variable.unique_id: deriv}

    for var in topo_order:
        current_deriv = gradient_map.get(var.unique_id, 0)
        if var.is_leaf():
            var.accumulate_derivative(current_deriv)
        else:
            for parent_var, parent_deriv in var.chain_rule(current_deriv):
                if parent_var.unique_id in gradient_map:
                    gradient_map[parent_var.unique_id] += parent_deriv
                else:
                    gradient_map[parent_var.unique_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors."""
        return self.saved_values
