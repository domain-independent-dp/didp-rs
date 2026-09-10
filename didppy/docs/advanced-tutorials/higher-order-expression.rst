Higher-Order Expressions
========================

In this tutorial, we explore *higher-order expressions* in DIDPPy: expressions that take other expressions as arguments.
DIDPPy provides the following higher-order expressions, all defined on :class:`~didppy.SetExpr`, :class:`~didppy.SetVar`, :class:`~didppy.SetResourceVar`, and :class:`~didppy.SetConst`:

* :meth:`~didppy.SetExpr.filter`: takes a local variable and a condition, and returns the subset of elements for which the condition holds.
* :meth:`~didppy.SetExpr.any` and :meth:`~didppy.SetExpr.all`: take a local variable and a condition, and test whether the condition holds for at least one or every element, respectively.
* :meth:`~didppy.SetExpr.sum`, :meth:`~didppy.SetExpr.product`, :meth:`~didppy.SetExpr.max`, and :meth:`~didppy.SetExpr.min`: each takes a local variable and an expression, and reduce the value of the expression over the elements of the set using the corresponding operator.

Filtering and reduction compose directly; for example, :code:`items.filter(x, predicate).sum(x, value)` sums :code:`value` over the elements satisfying :code:`predicate`.

In all of these, a fresh *local variable*, created by :meth:`~didppy.Model.add_local_var`, is bound to each element of the set while the condition or the expression being reduced is evaluated.
This is what makes them higher-order: the condition (for :meth:`~didppy.SetExpr.filter`, :meth:`~didppy.SetExpr.any`, and :meth:`~didppy.SetExpr.all`) or the value being reduced (for the others) is itself an expression parameterized by a variable representing "the current element," instead of a fixed expression.

In this tutorial, we demonstrate the usage of :meth:`~didppy.SetExpr.filter` using the orienteering problem with time windows (OPTW) as an example.
We assume that you are already familiar with the concepts covered in the :doc:`main tutorial <../tutorial>`, such as object types, state variables, tables, transitions, resource variables, and dual bounds.

OPTW
----

In OPTW, we are given a set of locations :math:`N = \{ 0, ..., n + 1 \}`.
The vehicle starts from the depot :math:`0`, visits each customer :math:`i \in \{ 1, ..., n \}` at most once, and must reach the goal :math:`n + 1` by the deadline :math:`b_{n+1}`.
By visiting customer :math:`i`, profit :math:`p_i` is obtained.
The travel time from :math:`i` to :math:`j` is :math:`c_{ij}`.
Each customer :math:`i` must be visited within time window :math:`[a_i, b_i]`, and the vehicle must wait until :math:`a_i` if it arrives at :math:`i` before :math:`a_i`.
The objective is to maximize the total profit.

Unlike the traveling salesperson problem with time windows (TSPTW) in the :doc:`main tutorial <../tutorial>`, the vehicle does not need to visit every customer: it may skip customers to reach the goal in time, and the total profit, not the total travel time, is what we optimize.

DP Formulation
--------------

Let :math:`R \subseteq \{ 1, ..., n \}` be the set of *reachable* customers, i.e., customers not yet visited that can still be visited without missing their own deadline; let :math:`i \in N` be the current location; and let :math:`t` be the current time.
Let :math:`V(R, i, t)` be the maximum additional profit that can still be collected from state :math:`(R, i, t)`.

When customer :math:`j \in R` is visited next, the current location becomes :math:`j`, and the current time becomes :math:`t_j = \max \{ t + c_{ij}, a_j \}`, the arrival time at :math:`j`, after waiting until :math:`a_j` if the vehicle arrives early.
A customer :math:`k \in R \setminus \{ j \}` remains reachable after this visit only if it can still be reached before its own deadline, i.e., :math:`t_j + c_{jk} \leq b_k` (assuming the triangle inequality for the travel time).
The vehicle may also choose to finish by heading directly to the goal, which is possible only if it can arrive by the deadline, i.e., :math:`t_{n+1} = t + c_{i,n+1} \leq b_{n+1}`; finishing collects no further profit.
Overall, we get the following DP formulation:

.. math::
    \text{compute } & V(N \setminus \{ 0, n + 1 \}, 0, 0) \\
    & V(R, i, t) = \begin{cases}
         \max_{j \in R \cup \{ n + 1 \}, t + c_{ij} \leq b_j} p_{j} + V(R_j, j, t_j) & \text{if } i \neq n + 1 \\
         0 & \text{if } i = n+1,
    \end{cases} 

where :math:`R_j = \{ k \in R \setminus \{ j \} \mid t_j + c_{jk} \leq b_k \}` for :math:`j \in N \setminus \{ 0, n + 1 \}` and :math:`R_{n+1} = \emptyset`.
We treat the goal as an additional customer with :math:`p_{n+1} = 0` and :math:`a_{n+1} = 0`, so that finishing collects no further profit and requires no waiting -- this is what lets the single case above also cover heading directly to the goal.
If no :math:`j \in R \cup \{ n + 1 \}` satisfies :math:`t + c_{ij} \leq b_j`, the maximum is over an empty set, which we take to be :math:`-\infty`: by the triangle inequality, once the goal is unreachable directly it can never become reachable through a detour, so such a state has no valid continuation.
The second case reflects that, once the vehicle has reached the goal, no more profit can be collected.

When two states :math:`(R, i, t)` and :math:`(R', i, t')` share the same current location :math:`i` and satisfy :math:`t \leq t'` and :math:`R' \subseteq R`, the vehicle in state :math:`(R, i, t)` has at least as much time left and at least as many customers still reachable, so it can reproduce any solution available from :math:`(R', i, t')`.
Therefore,

.. math::
    V(R, i, t) \geq V(R', i, t') \text{ if } t \leq t' \land R' \subseteq R.

This is exactly the dominance relation that resource variables are meant to exploit: preferring a smaller :math:`t` and a larger :math:`R` at the same time.

The total remaining profit can be overestimated by solving a fractional knapsack problem: each item :math:`j \in R` is a customer, its value is the profit :math:`p_j`, and its weight is the minimum possible travel time to reach :math:`j`, :math:`\min_{k \in N \setminus \{ j, n+1 \}} c_{kj}` (precomputed as :math:`\mathsf{min\_to}_j`), and the knapsack has capacity :math:`b_{n+1} - t - \mathsf{min\_to}_{n+1}`, the time budget left after also reserving enough time to travel from wherever the next customer is back to the goal.

.. math::
    V(R, i, t) \leq \left\lfloor \mathsf{fractional\_knapsack}\left(R,\ b_{n+1} - t - \mathsf{min\_to}_{n+1},\ (p_j)_{j \in R},\ (\mathsf{min\_to}_j)_{j \in R} \right) \right\rfloor.

Similarly, we can use the minimum travel time to *leave* each customer, :math:`\mathsf{min\_from}_j = \min_{k \in N \setminus \{ j, n+1 \}} c_{jk}`, as the weight, and reserve the time to leave the current location instead:

.. math::
    V(R, i, t) \leq \left\lfloor \mathsf{fractional\_knapsack}\left(R,\ b_{n+1} - t - \mathsf{min\_from}_i,\ (p_j)_{j \in R},\ (\mathsf{min\_from}_j)_{j \in R} \right) \right\rfloor.

Modeling in DIDPPy
------------------

We assume that the data is precomputed and given.
As in the main tutorial, we index locations from :math:`0` to :math:`n + 1`, with the depot at :math:`0` and the goal at :math:`n + 1`.

.. code-block:: python

    import math

    import didppy as dp

    # Number of customers
    n = 3
    # Profit
    p = [0, 2, 3, 4, 0]
    # Ready time
    a = [0, 5, 0, 8, 0]
    # Due time
    b = [100, 12, 10, 12, 17]
    # Travel time
    c = [
        [0, 3, 4, 5, 0],
        [3, 0, 5, 4, 3],
        [4, 5, 0, 3, 4],
        [5, 4, 3, 0, 5],
        [0, 3, 4, 5, 0],
    ]

    model = dp.Model(maximize=True)

    customer = model.add_object_type(number=n + 2)

    # R
    reachable = model.add_set_resource_var(
        object_type=customer, target=list(range(1, n + 1)), less_is_better=False
    )
    # i
    location = model.add_element_var(object_type=customer, target=0)
    # t
    time = model.add_int_resource_var(target=0, less_is_better=True)

    travel_time = model.add_int_table(c)
    due_time = model.add_int_table(b)

:code:`p`, :code:`a`, :code:`b`, and :code:`c` correspond directly to :math:`p`, :math:`a`, :math:`b`, and :math:`c` in the DP formulation, with entries :code:`0` and :code:`n + 1` for the depot and the goal.
Because a larger :code:`reachable` and a smaller :code:`time` are both preferable, matching the dominance relation derived above, we define :code:`reachable` as a :class:`~didppy.SetResourceVar` with :code:`less_is_better=False`, using :meth:`~didppy.Model.add_set_resource_var`.
:class:`~didppy.SetResourceVar` behaves like :class:`~didppy.SetVar` but additionally participates in this kind of dominance pruning.
We register :code:`c` and :code:`b` as tables since they are indexed by state variables below; :code:`a` is only ever indexed by :code:`j`, a plain Python :class:`int` at each unrolled iteration of the loop, so we can use it directly without registering it as a table.

Next, we define the :code:`visit` transitions.

.. code-block:: python

    k = model.add_local_var()

    for j in range(1, n + 1):
        time_next = dp.max(time + travel_time[location, j], a[j])
        visit = dp.Transition(
            name="visit {}".format(j),
            cost=p[j] + dp.IntExpr.state_cost(),
            effects=[
                (
                    reachable,
                    reachable.remove(j).filter(
                        k, time_next + travel_time[j, k] <= due_time[k]
                    ),
                ),
                (location, j),
                (time, time_next),
            ],
            preconditions=[
                reachable.contains(j),
                time + travel_time[location, j] <= due_time[j],
            ],
        )
        model.add_transition(visit)

    empty_set = model.create_set_const(object_type=customer, value=[])
    finish = dp.Transition(
        name="finish",
        cost=dp.IntExpr.state_cost(),
        effects=[(reachable, empty_set), (location, n + 1), (time, time + travel_time[location, n + 1])],
        preconditions=[time + travel_time[location, n + 1] <= due_time[n + 1]],
    )
    model.add_transition(finish)

    model.add_base_case([location == n + 1])

We create a single local variable :code:`k` with :meth:`~didppy.Model.add_local_var` and reuse it across all :code:`visit` transitions: each :meth:`~didppy.SetExpr.filter` call is evaluated independently, so nothing is shared between the iterations, and DIDPPy does not require a fresh local variable per call.
The effect on :code:`reachable` computes :math:`R_j` from the DP formulation: we first remove :code:`j` from :code:`reachable`, and then call :meth:`~didppy.SetExpr.filter` with :code:`k` and the condition :code:`time_next + travel_time[j, k] <= due_time[k]`.
This filters the set down to the elements :code:`k` for which the condition holds, with :code:`k` bound to each candidate element in turn -- exactly the local-variable binding that makes :meth:`~didppy.SetExpr.filter` higher-order.
This computation could not be expressed with the set operators (:code:`|`, :code:`&`, :code:`-`, :code:`^`) alone, since whether each remaining customer :code:`k` stays reachable depends on an expression evaluated *for that customer*.

The :code:`finish` transition sets :code:`reachable` to :code:`empty_set`, matching :math:`R_{n+1} = \emptyset` in the DP formulation.

Finally, we define the two fractional-knapsack dual bounds.

.. code-block:: python

    profit = model.add_int_table(p)

    min_to = model.add_int_table(
        [min(c[i][j] for i in range(n + 1) if i != j) for j in range(n + 2)]
    )
    model.add_dual_bound(
        math.floor(
            dp.fractional_knapsack(
                reachable, due_time[n + 1] - time - min_to[n + 1], profit, min_to
            )
        )
    )

    min_from = model.add_int_table(
        [min(c[j][i] for i in range(n + 1) if i != j) for j in range(n + 2)]
    )
    model.add_dual_bound(
        math.floor(
            dp.fractional_knapsack(
                reachable, due_time[n + 1] - time - min_from[location], profit, min_from
            )
        )
    )

:code:`min_to` and :code:`min_from` are precomputed exactly as :math:`\mathsf{min\_to}` and :math:`\mathsf{min\_from}` in the DP formulation: for each location, the minimum travel time to/from any other location, excluding the goal, since the goal can never be an intermediate stop.
:func:`~didppy.fractional_knapsack` takes the set of items, the capacity, and two tables (or lists of expressions) of values and weights, and returns a :class:`~didppy.FloatExpr` bounding the optimal value of the fractional knapsack problem.
Since our model uses an integer cost, we convert the resulting :class:`~didppy.FloatExpr` to an :class:`~didppy.IntExpr` with :func:`math.floor`, which DIDPPy expressions support via :meth:`~didppy.FloatExpr.__floor__`.

Solving the Model
-----------------

The cost of every transition has the form :code:`w + dp.IntExpr.state_cost()`, so we can use :class:`~didppy.CABS` with its default :code:`f_operator`.

.. code-block:: python

    solver = dp.CABS(model)
    solution = solver.search()

    print("Transitions to apply:")

    for t in solution.transitions:
        print(t.name)

    print("Profit: {}".format(solution.cost))
