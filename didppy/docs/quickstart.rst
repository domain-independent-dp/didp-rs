Quick Start
===========

Let's get started with DIDPPy by modeling and solving a simple knapsack problem.

Dynamic Programming for Knapsack
--------------------------------

In the knapsack problem, we are given the set of items :math:`N = \{ 0, ..., n-1 \}` with weights :math:`w_i` and profits :math:`p_i` for :math:`i \in N` and a knapsack with capacity :math:`c`.
We want to maximize the total profit of the items  in the knapsack.

Think about considering the items one by one.
If we pack item :math:`0`, the remaining problem is to pack the remaining items :math:`\{ 1, ..., n - 1 \}` into a knapsack with capacity :math:`c - w_0`.
Otherwise, we pack the remaining items into a knapsack with capacity :math:`c`.
Let :math:`V(r, i)` be the maximum profit of packing items :math:`\{ i, ..., n - 1 \}` into a knapsack with capacity :math:`r`.
We compute :math:`V(c, 0)` using the following recursive equation, which is the dynamic programming (DP) model.

.. math::
    V(r, i) = \begin{cases}
        \max\{ p_i + V(r - w_i, i + 1), V(r, i + 1) \} & \text{if } i < n \land r \geq w_i \\
        V(r, i + 1) & \text{if } i < n \land r < w_i \\
        0 & \text{otherwise.}
    \end{cases}

Modeling in DIDPPy
------------------

Now, let's code the above DP model in DIDPPy.
Suppose that :math:`n = 4`, :math:`w_0 = 10`, :math:`w_1 = 20`, :math:`w_2 = 30`, :math:`w_3 = 40`, :math:`v_0 = 5`, :math:`v_1 = 25`, :math:`v_2 = 35`, :math:`v_3 = 50`, and :math:`c = 50`.

.. code-block:: python

    import didppy as dp

    n = 4
    weights = [10, 20, 30, 40]
    profits = [5, 25, 35, 50]
    c = 50

    model = dp.Model(maximize=True, float_cost=False)

    item = model.add_object_type(number=n)
    r = model.add_int_var(target=c)
    i = model.add_element_var(object_type=item, target=0)

    w = model.add_int_table(weights)
    p = model.add_int_table(profits)

    state_cost = dp.IntExpr.state_cost()

    pack = dp.Transition(
        name="pack",
        cost=p[i] + state_cost,
        effects=[(r, r - w[i]), (i, i + 1)],
        preconditions=[i < n, r >= w[i]],
    )
    model.add_transition(pack)

    ignore = dp.Transition(
        name="ignore",
        cost=state_cost,
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    model.add_transition(ignore)

    model.add_base_case([i == n])

We will explain the details in the :doc:`tutorial <tutorial>`, but here is a summary:

* State variables :code:`r` and :code:`i` corresponding to :math:`r` and :math:`i` are defined with the *target* values :code:`c` and :code:`0`, which states that we want to compute :math:`V(c, 0)`.
* Recursive equations are defined by transitions, which change the state variables and the cost.
* The cost of the subproblem in the right-hand side of the recursive equations is represented by `state_cost`.
* The condition to terminate the recursion is defined by the base case `i == n`.

Solving the Model
-----------------

Once you have the model, you can use solvers provided by DIDPPy to solve the model.
You do not need to implement the DP algorithm yourself.
Let's use :class:`~didppy.ForwardRecursion` to solve the model.

.. code-block:: python

    solver = dp.ForwardRecursion(model)
    solution = solver.search()

    for i, t in enumerate(solution.transitions):
        if t.name == "pack":
            print("pack {}".format(i))

    print("profit: {}".format(solution.cost))

This solver is the most generic, i.e., it can handle almost any model you can formulate in DIDPPy.
However, if your DP model has a particular structure, you can use more efficient solvers.
For example. you can use :class:`~didppy.CABS` for this model.

.. code-block:: python

    solver = dp.CABS(model)
    solution = solver.search()

    for i, t in enumerate(solution.transitions):
        if t.name == "pack":
            print("pack {}".format(i))

    print("profit: {}".format(solution.cost))

The solvers are listed in the :ref:`API reference <reference:Solvers>`, and their restrictions are described in the individual pages.
Also, we provide a :doc:`guideline to select a solver </solver-selection>`.
