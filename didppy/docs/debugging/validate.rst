Validating a Solution
=====================

One way to debug a model is to use a small problem instance and see why an expected solution is not found.
DIDPPy provides :meth:`didppy.Model.validate_forward` to validate a solution.
Let's use the model for the knapsack problem from the :doc:`quickstart </quickstart>` to illustrate this.

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

    solution = [ignore, pack, pack, ignore]
    cost = 60

    result = model.validate_forward(solution, cost)

    if result:
        print("Solution is valid.")
    else:
        print("Solution is invalid.")

:meth:`~didppy.Model.validate_forward` takes a sequence of transitions and a cost as input.
It returns :code:`True` if the transitions change the target state to a base case, and its cost is equal to the given cost.

If the solution is invalid, it returns :code:`False` and displays the reason for the failure.
For example, suppose that we comment out the base case.
Then, the problem is unsolvable as there is no way to reach a base case.

.. code-block:: python

    # Comment out the base case
    # model.add_base_case([i == n])

    solution = [ignore, pack, pack, ignore]
    cost = 60

    result = model.validate_forward(solution, cost)

Then, it will display the following message:

.. code-block:: bash

    The last state is not a base state.

If we make it impossible to satisfy the preconditions of the :code:`ignore` transition,

.. code-block:: python

    ignore = dp.Transition(
        name="ignore",
        cost=dp.IntExpr.state_cost(),
        effects=[(i, i + 1)],
        preconditions=[i < n, i > n],
    )
    model.add_transition(ignore)

it will display the following message:

.. code-block:: bash

    The 0 th transition ignore is not applicable.

It also checks if the cost of the solution is correct.

.. code-block:: python

    solution = [ignore, pack, pack, ignore]
    cost = 50

    result = model.validate_forward(solution, cost)

.. code-block:: bash

    The cost 50 does not match the actual cost 60. This is possibly due to the cost is continuous.