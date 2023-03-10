Common Mistakes
===============

Here are some common mistakes that users make.

Forgetting Adding a Transition
------------------------------

When you define a transition in the model, do not forget to add it to the model using :meth:`~didppy.Model.add_transition`.

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    var = model.add_int_var(target=0)

    transition = dp.Transition(
        name='increment',
        cost=1 + dp.IntExpr.state_cost(),
        effects=[(var, var + 1)],
    )
    # Do not forget this!
    model.add_transition(transition)

Using Built-in :func:`max` for Expressions
------------------------------------------

When you want to take the maximum of two :class:`~didppy.IntExpr`, you cannot use built-in :func:`max` in Python.
Instead, you need to use :class:`~didppy.IntExpr.max`.

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    var = model.add_int_var(target=0)

    # This is wrong!
    model.add_dual_bound(max(var, 1))

    # This is correct.
    model.add_dual_bound(dp.max(var, 1))

The same thing applies to :func:`min`.

Using If Statements with :class:`~didppy.Condition`
---------------------------------------------------

Values of expressions are not determined immediately: it is evaluated given a state inside the solver.
Therefore, you cannot use if statements with :class:`~didppy.Condition` because the result of the if statement is determined immediately.
Instead, you need to use :meth:`~didppy.Condition.if_then_else`.

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    var = model.add_int_var(target=0)

    # This is wrong!
    if var >= 1:
        model.add_dual_bound(1)
    else:
        model.add_dual_bound(0)

    # This is correct.
    model.add_dual_bound((var >= 1).if_then_else(1, 0))

If you want to make a transition available only when a condition is satisfied, you should define it as a precondition.

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    var = model.add_int_var(target=0)

    # This is wrong!
    if var <= 2:
        transition = dp.Transition(
            name='increment',
            cost=1 + dp.IntExpr.state_cost(),
            effects=[(var, var + 1)],
        )
        model.add_transition(transition)

    # This is correct.
    transition = dp.Transition(
        name='increment',
        cost=1 + dp.IntExpr.state_cost(),
        effects=[(var, var + 1)],
        preconditions=[var <= 2],
    )
    model.add_transition(transition)

Using Boolean Operators in :class:`~didppy.Condition`
-----------------------------------------------------

When you want to take the negation, disjunction, and conjunction of :class:`~didppy.Condition`, you cannot use built-in boolean operators (:code:`not`, :code:`or`, and :code:`and`) in Python.
Instead, you need to use bitwise operators (:code:`~`, :code:`|`, and :code:`&`).

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    var = model.add_int_var(target=0)

    # This is wrong!
    model.add_base_case([(var >= 0 or not var >= 3) and var <= 2])

    # This is correct.
    model.add_base_case([(var >= 0 or ~(var >= 3)) & (var <= 2)])

Using a Table as a Nested List
------------------------------

A table in the model can be created from a nested list, but it is not a nested list.
Use :class:`tuple` as indices instead of nested indices.

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    obj = model.add_object_type(number=2)
    var = model.add_element_var(object_type=obj, target=0)

    table = model.add_int_table([[1, 2], [3, 4]])

    # This is wrong!
    model.add_base_case([table[var][0] == 2])

    # This is correct.
    model.add_base_case([table[var, 0] == 2])

Using an Inappropriate Solver
-----------------------------

The solvers provided by DIDPPy are not always applicable to all models.
If a solver produces a wrong solution, it is likely that the model is not supported by the solver.
Please refer to the :doc:`solver selection guide </solver-selection>` and the :ref:`API reference <api-reference:Solvers>` to check which solver supports which types of models.
