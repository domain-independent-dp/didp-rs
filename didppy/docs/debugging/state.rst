Debugging using States
======================

In DIDPPy, the values of expressions used in transitions, base cases, state constraints, and dual bounds depend on the values of state variables, which are manipulated inside the solver.
However, we sometimes want to know the values of the expressions given a specific state.
DIDPPy allows us to do this kind of debugging using :class:`~didppy.State`.
You can even develop your own solver in Python using the interfaces introduced here.

Getting and Changing a State
----------------------------

We can get the target state of the model by :attr:`didppy.Model.target_state`.

.. code-block:: python

    >>> import didpppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> state = model.target_state
    >>> state[var]
    0

We can get and change the value of a state variable using :code:`state[var]`.

.. code-block:: python

    >>> state[var] = 1
    >>> state[var]
    1

Note that the value of the target state does not change with the above code because :code:`state` is the copy of the target state.
If we want to change the target state, we can use :meth:`~didppy.Model.set_target` or do like the following.

.. code-block:: python

    >>> model.target_state = state

However, :code:`model.target_state[var] = 1` does not work.

In the case of :class:`~didppy.SetVar`, we need to use :class:`~didppy.SetConst` to change its value.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> obj = model.add_object_type(number=4)
    >>> var = model.add_set_var(object_type=obj, target=[0, 1])
    >>> state = model.target_state
    >>> state[var]
    {0, 1}
    >>> const = model.create_set_const(object_type=obj, value=[1, 2])
    >>> state[var] = const
    >>> state[var]
    {1, 2}

Evaluating an Expression
------------------------

We can evaluate an expression given a state with :meth:`~didppy.IntExpr.eval`.
Expressions other than :class:`~didppy.IntExpr` can be evaluated in the same way.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> state = model.target_state
    >>> (var + 1).eval(state, model)
    1
    >>> (var > 0).eval(state, model)
    False

Applying a Transition to a State
--------------------------------

For a transition, we can check if it is applicable, the values of state variables after application, and the cost given a state.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> state = model.target_state
    >>> transition = dp.Transition(
    ...     name="increment",
    ...     cost=2 + dp.IntExpr.state_cost(),
    ...     preconditions=[var <= 2],
    ...     effects=[(var, var + 1)],
    ... )
    >>> transition.is_applicable(state, model)
    True
    >>> next_state = transition.apply(state, model)
    >>> next_state[var]
    1
    >>> transition.eval_cost(1, state, model)
    3

We can also get components of transitions by using :attr:`~didppy.Transition.cost`, :attr:`~didppy.Transition.preconditions`, and :code:`transition[var]` for an effect.

.. code-block:: python

    >>> cost = transition.cost
    >>> cost.eval_cost(1, state, model)
    3
    >>> preconditions = transition.preconditions
    >>> preconditions[0].eval(state, model)
    True
    >>> effect = transition[var]
    >>> effect.eval(state, model)
    1

Note that the order of the preconditions might be changed due to internal implementation.

We can update :attr:`~didppy.Transition.cost` and the effects.

.. code-block:: python

    >>> transition.cost = 1 + dp.IntExpr.state_cost()
    >>> transition.eval_cost(1, state, model)
    2
    >>> transition[var] = var + 2
    >>> next_state = transition.apply(state, model)
    >>> next_state[var]
    2

Checking Base Cases
-------------------

We can check if a state satisfies the base cases with :meth:`~didppy.Model.is_base`.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> model.add_base_case([var >= 2])
    >>> state = model.target_state
    >>> model.is_base(state)
    False

We can get the base cases with :attr:`~didppy.Model.base_cases`.

.. code-block:: python

    >>> base_cases = model.base_cases
    >>> base_cases[0][0].eval(state, model)
    False


Checking State Constraints
--------------------------

We can check if a state satisfies the state constraints with :meth:`~didppy.Model.check_state_constr`.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> model.add_state_constr(var >= 0)
    >>> state = model.target_state
    >>> model.check_state_constr(state)
    True

We can get the state constraints with :attr:`~didppy.Model.state_constrs`.

.. code-block:: python

    >>> constraints = model.state_constrs
    >>> constraints[0].eval(state, model)

Evaluating Dual Bound
---------------------

We can evaluate the value of the dual bound for a state with :meth:`~didppy.Model.eval_dual_bound`.

.. code-block:: python

    >>> import didppy as dp
    >>> model = dp.Model()
    >>> var = model.add_int_var(target=0)
    >>> model.add_dual_bound(var)
    >>> state = model.target_state
    >>> model.eval_dual_bound(state)
    0

We can get the dual bounds with :attr:`~didppy.Model.dual_bounds`.

.. code-block:: python

    >>> dual_bounds = model.dual_bounds
    >>> dual_bounds[0].eval(state, model)
    0