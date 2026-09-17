Quick Start
===========

Let's get started with DIDPPy by modeling and solving a simple knapsack problem.

Dynamic Programming for Knapsack
--------------------------------

In the knapsack problem, we are given the set of items :math:`N = \{ 0, ..., n-1 \}` with weights :math:`w_i` and profits :math:`p_i` for :math:`i \in N` and a knapsack with capacity :math:`c`.
We want to maximize the total profit of the items in the knapsack.

Consider selecting the items one by one from :math:`0` to :math:`n - 1`.
If we pack item :math:`0`, the remaining problem is to select items to pack from :math:`\{ 1, ..., n - 1 \}` into a knapsack with capacity :math:`c - w_0`.
Otherwise, we use a knapsack with capacity :math:`c` for the remaining items.
Let :math:`V(r, i)` be the maximum profit of selecting items to pack from :math:`\{ i, ..., n - 1 \}` into a knapsack with capacity :math:`r`.
We compute :math:`V(c, 0)` using the following recursive equation, which is the dynamic programming (DP) formulation.

.. math::
    V(r, i) = \begin{cases}
        \max\{ p_i + V(r - w_i, i + 1), V(r, i + 1) \} & \text{if } i < n \land r \geq w_i \\
        V(r, i + 1) & \text{if } i < n \land r < w_i \\
        0 & \text{otherwise.}
    \end{cases}

Modeling in DIDPPy
------------------

Now, let's model the above DP formulation in DIDPPy.
Suppose that :math:`n = 4`, :math:`w_0 = 10`, :math:`w_1 = 20`, :math:`w_2 = 30`, :math:`w_3 = 40`, :math:`p_0 = 5`, :math:`p_1 = 25`, :math:`p_2 = 35`, :math:`p_3 = 50`, and :math:`c = 50`.
We add a dummy item :math:`4` with :math:`w_4 = 0` and :math:`p_4 = 0` to terminate the recursion.

.. code-block:: python

    import math

    import didppy as dp

    n = 4
    weights = [10, 20, 30, 40]
    profits = [5, 25, 35, 50]
    capacity = 50

    model = dp.Model(maximize=True, float_cost=False)

    item = model.add_object_type(number=n + 1)
    r = model.add_int_resource_var(target=capacity, less_is_better=False)
    i = model.add_element_var(object_type=item, target=0)

    w = model.add_int_table(weights + [0])
    p = model.add_int_table(profits + [0])

    pack = dp.Transition(
        name="pack",
        cost=p[i] + dp.IntExpr.state_cost(),
        effects=[(r, r - w[i]), (i, i + 1)],
        preconditions=[i < n, r >= w[i]],
    )
    model.add_transition(pack)

    ignore = dp.Transition(
        name="ignore",
        cost=dp.IntExpr.state_cost(),
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    model.add_transition(ignore)

    model.add_base_case([i == n])

    remaining_items = model.add_set_table(
        [list(range(i, n)) for i in range(n + 1)], object_type=item
    )
    model.add_dual_bound(math.floor(dp.fractional_knapsack(remaining_items[i], r, p, w)))

We will explain the details in the :doc:`tutorial <tutorial>`, but here is a summary:

* State variables :code:`r` and :code:`i`, corresponding to :math:`r` and :math:`i`, are defined with the *target* values :code:`c` and :code:`0`, which states that we want to compute :math:`V(c, 0)`.
* Given :math:`i`, a state with larger :math:`r` is better, so we define :code:`r` using :meth:`~didppy.Model.add_int_resource_var` and set :code:`less_is_better=False`.
* Recursive equations are defined by transitions, which change the state variables and the cost.
* The cost of the subproblem on the right-hand side of the recursive equations is represented by :meth:`~didppy.IntExpr.state_cost`.
* The condition to terminate the recursion is defined by the base case :code:`i == n`.
* The dual bound is defined by the fractional knapsack problem (:func:`~didppy.fractional_knapsack`), which is a relaxation of the knapsack problem.

Solving the Model
-----------------

Once you have the model, you can use solvers provided by DIDPPy to solve the model.
You do not need to implement the DP algorithm yourself.
Let's use :class:`~didppy.CABS` to solve the model.

.. code-block:: python

    solver = dp.CABS(model)
    solution = solver.search()

    for i, t in enumerate(solution.transitions):
        if t.name == "pack":
            print("pack {}".format(i))

    print("profit: {}".format(solution.cost))

The solvers are listed in the :ref:`API reference <api-reference:Solvers>`, and their restrictions are described in the individual pages.
Also, we provide a :doc:`guideline to select a solver </solver-selection>`.
