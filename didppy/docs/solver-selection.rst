Solver Selection
================

DIDPPy provides a number of :ref:`solvers <api-reference:Solvers>`.
This document provides a guideline to select an appropriate solver for your DP model.

In general, we recommend using :class:`~didppy.CABS` if possible.
It has the following advantages:

* Anytime: it usually finds a feasible solution quickly and improves the solution quality over time.
* Complete: it is guaranteed to find an optimal solution or prove the infeasibility of the model given sufficient time.
* Memory efficient: it consumes less memory compared to other solvers.

However, it has the following disadvantages:

* It may take a :ref:`longer time to prove the optimality <solver-selection:Time to Prove Optimality>` compared to other solvers.
* A :ref:`configuration <solver-selection:Layer-by-Layer Search>` is needed to handle certain types of DP models as it searches layer by layer.
* :ref:`Cost expressions <solver-selection:Restriction on Cost Expressions>` must be in the form of addition, product, maximum, or minimum.

Time to Prove Optimality
------------------------

:class:`~didppy.CABS` is sometimes slow to prove the optimality.
This does not mean that :class:`~didppy.CABS` is slow to find an optimal solution; it just takes time to prove the optimality of the found solution.
If you want to prove the optimality as fast as possible, :class:`~didppy.CAASDy` might be a choice.
One disadvantage of :class:`~didppy.CAASDy` is that it is not an anytime solver: it does not find any solution until it proves the optimality.
If you want to use anytime solvers, consider :class:`~didppy.ACPS` and :class:`~didppy.APPS`.
However, these alternatives consume more memory than :class:`didppy.CABS`, so if the memory limit is a concern, they may not be a good choice.
The experimental comparison of :class:`~didppy.CAASDy` and the anytime solvers is provided in :cite:t:`DIDPAnytime`.

Layer-by-Layer Search
---------------------

DP solvers typically search the state space: they generate states that are reachable from the target state using transitions.
They store the states encountered in memory and check if it has been encountered before each time a state is generated.
In this way, DP solvers save computational time by avoiding evaluating the same state multiple times at the expense of the computational space.

:class:`~didppy.CABS` searches layer by layer:
in the :math:`i` th iteration, it searches states that are reachable from the target state using :math:`i` transitions.
By default, :class:`~didppy.CABS` only stores the states in the current layer in memory.
However, in some problems, a state can belong to multiple layers, i.e., the state can be reached from the target state with different numbers of transitions.
It is also possible that a state space contains cycles: a state can be reached from itself with a finite number of transitions.
In such a case, we may want to store states not only in the current layer but also in the previous layers.
We can do that by using :code:`keep_all_layers=True` when creating a solver.

.. code-block:: python

    solver = dp.CABS(model, keep_all_layers=True)

This is also the case for :class:`~didppy.BreadthFirstSearch` and :class:`~didppy.ExpressionBeamSearch`.

Restriction on Cost Expressions
-------------------------------

To use :class:`~didppy.CABS`, the cost expressions (:code:`cost` in :class:`~didppy.Transition`) of all transitions must be in either of the following forms:

* :code:`w + dp.IntExpr.state_cost()`
* :code:`w * dp.IntExpr.state_cost()`
* :code:`dp.max(w, dp.IntExpr.state_cost())`
* :code:`dp.min(w, dp.IntExpr.state_cost())`

where :code:`w` is an :class:`~didppy.IntExpr` independent of :meth:`~didppy.IntExpr.state_cost`.
For float cost, we can use :class:`~didppy.FloatExpr` instead of :class:`~didppy.IntExpr`.
By default, :class:`~didppy.CABS` assumes that :code:`cost` is the additive form.
For other types of :code:`cost`, we need to tell the solver by using the argument :code:`f_operator`, which takes either of :attr:`didppy.FOperator.Plus`, :attr:`didppy.FOperator.Product`, :attr:`didppy.FOperator.Max`, or :attr:`didppy.FOperator.Min` (:attr:`~didppy.FOperator.Plus` is the default).
An example is provided in as an :doc:`advanced tutorial <advanced-tutorials/general-cost>`.

This restriction is shared by the following path-finding (or heuristic search) based solvers:

* :class:`~didppy.CABS`
* :class:`~didppy.CAASDy`
* :class:`~didppy.ACPS`
* :class:`~didppy.APPS`
* :class:`~didppy.DFBB`
* :class:`~didppy.DBDFS`
* :class:`~didppy.BreadthFirstSearch`
* :class:`~didppy.WeightedAstar`
* :class:`~didppy.ExpressionBeamSearch`

Currently, only :class:`~didppy.ForwardRecursion` supports arbitrary cost expressions.
However, it does not support cyclic state spaces.
