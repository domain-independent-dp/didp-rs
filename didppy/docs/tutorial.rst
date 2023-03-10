Tutorial
========

Let's learn DIDPPy using the traveling salesperson problem with time windows (TSPTW) as an example.

TSPTW
-----

In TSPTW, we are given a set of locations :math:`N = \{ 0, ..., n-1 \}`.
The salesperson starts from the depot :math:`0`, visit each customer :math:`i \in \{ 1, ..., n-1 \}` exactly once, and returns to the depot.
The traveling time from :math:`i` to :math:`j` is :math:`c_{ij}`.
Each customer :math:`i` must be visited within time window :math:`[a_i, b_i]`, and the salesperson must wait until :math:`a_i` if arriving at :math:`i` before :math:`a_i`.
The objective is to minimize the total travel time (not including the waiting time).

The following image is an example of TSPTW with four locations.

.. image:: _static/images/TSPTW.png

DP Formulation for TSPTW
------------------------

In DP, using recursive equations, we decompose the problem into subproblems and describe the optimal value of the original problem using the values of the subproblems.
Each problem is defined as a *state*, which is a tuple of variables that describe the problem.
The value function :math:`V` maps a state to the optimal value of the problem.

In TSPTW, think about visiting customers one by one.
If the salesperson visits :math:`j` from the depot, the problem is reduced to visiting customers :math:`N \setminus \{ 0, j \}` from location :math:`j` at time :math:`\max \{ c_{0j}, a_j \}`.
Therefore, we can define a subproblem using the following three variables:

* :math:`U \subseteq N` : the set of unvisited customers.
* :math:`i \in N` : the current location.
* :math:`t` : the current time.

In genral, when customer :math:`j \in U` is visited from location :math:`i` at time :math:`t`, the problem is reduced to visiting customers :math:`U \setminus \{ j \}` from location :math:`j` at time :math:`\max \{ t + c_{ij}, a_j \}`.
To visit customer :math:`j`, the salesperson must arrive before :math:`b_j`, i.e., :math:`t + c_{ij} \leq b_j`.
When all customers are visited, the salesperson must return to the depot from location :math:`i`.
Overall, we get the following DP formulation:

.. math::
    \text{compute } & V(N \setminus \{ 0 \}, 0, 0) \\ 
    & V(U, i, t) = \begin{cases}
         \min\limits_{j \in U: t + c_{ij} \leq b_j} c_{ij} + V(U \setminus \{ j \}, j, \max \{ t + c_{ij}, a_j \})  & \text{if } U \neq \emptyset \\
         c_{i0} + V(U, 0, t + c_{i0}) & \text{if } U = \emptyset \land i \neq 0 \\
         0 & \text{if } U = \emptyset \land i = 0.
    \end{cases}

In the first line, if there is no :math:`j \in U` with :math:`t + c_{ij} \leq b_j` while :math:`U \neq \emptyset`, we assume that :math:`V(U, i, t) = \infty` because the subproblem does not have a solution.

We call the state :math:`(N \setminus \{ 0 \}, 0, 0)`, which corresponds to the original problem, the *target state*.

This DP formulation is based on :cite:t:`Dumas1995`.

Modeling in DIDPPy
------------------

Now, let's model the above DP formulation in DIDPPy.
Assume that the data is given.
First, start with importing DIDPPy and creating the model.

.. code-block:: python

    import didppy as dp


    # Number of locations
    n = 4
    # Ready time
    a = [0, 5, 0, 8]
    # Due time
    b = [100, 16, 10, 14]
    # Travel time
    c = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
   
    model = dp.Model(maximize=False, float_cost=False)

Because the objective is to minimize the total travel time, we set :code:`maximize=False`.
We assume that the travel time is an integer, so we set :code:`float_cost=False`.
Actually, :code:`maximize=False` and :code:`float_cost=False` are the default values, so we can omit them.

Object Types
~~~~~~~~~~~~

First, we define an *object type*, which represents the type of objects that are used in the model.
In TSPTW, customers are objects with the same object type.

.. code-block:: python

   customer = model.add_object_type(number=n)

When defining an object type, we need to specify the number of objects.
If the number of objects is :math:`n`, the objects are indexed from :math:`0` to :math:`n-1` (**not** :math:`1` **to** :math:`n`) in DIDPPy. 
Object types are sometimes required to define a state, as explained later.

State Variables
~~~~~~~~~~~~~~~

A state of a problem is defined by *state variables*.
There are four types of state variables:

* :class:`~didppy.SetVar` : a set of the indices of objects associated with an object type.
* :class:`~didppy.ElementVar` : the index of an object associated with an object type.
* :class:`~didppy.IntVar` : an integer.
* :class:`~didppy.FloatVar` : a continuous value.

In TSPTW, :math:`U` is a :class:`~didppy.SetVar`, :math:`i` is an :class:`~didppy.ElementVar`, and :math:`t` is an :class:`~didppy.IntVar`.

.. code-block:: python

   # U
   unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
   # i
   location = model.add_element_var(object_type=customer, target=0)
   # t
   time = model.add_int_var(target=0)

While :math:`i` is an integer, we define it as an :class:`~didppy.ElementVar`  as it represents an element in the set :math:`N`.
There are some practical differences between :class:`~didppy.ElementVar` and :class:`~didppy.IntVar`:

* :class:`~didppy.ElementVar` is nonnegative.
* :class:`~didppy.ElementVar` can be used to describe changes and conditions on :class:`~didppy.SetVar`.
* :class:`~didppy.ElementVar` can be used to access a value of a table (explained later).

While we use the integer cost and an integer variable for :math:`t`, we can use the float cost and a float variable for :code:`t` by using :meth:`~didppy.Model.add_float_var` if we want to use continuous travel time.

The value of :class:`~didppy.SetVar` is a set of elements in :math:`N`.
Because the object type of :code:`unvisited` is customer, which has :code:`n` objects, :code:`unvisited` can contain :code:`0` to :code:`n - 1` (**not** :code:`1` **to** :code:`n`).

State variables are defined with their *target values*, values in the target state.
The objective of the DP model is to compute the value of the target state, i.e., :math:`U = N \setminus \{ 0 \}`, :math:`i = 0`, and :math:`t = 0`.
The target value of an :class:`~didppy.SetVar` can be a :class:`list` or a :class:`set` in Python.
In addition, we can initialize it using :class:`~didppy.SetConst`, which is created by :meth:`~didppy.Model.create_set_const`.

Tables of Constants
~~~~~~~~~~~~~~~~~~~

In TSPTW, :math:`a_i`, :math:`b_i`, and :math:`c_{ij}` are constants depending on customers.
In DIDPPy, such constants are defined as *tables*.

.. code-block:: python

   ready_time = model.add_int_table(a)
   due_time = model.add_int_table(b)
   travel_time = model.add_int_table(c)

By passing a nested list of :class:`int` to :meth:`~didppy.Model.add_int_table`, we can create up to a three-dimensional int table.
For tables more than three-dimensional, we can pass a :class:`dict` in Python with specifying the default value used when an index is not included in the :class:`dict`.
See :meth:`~didppy.Model.add_int_table` for more details.

We can add different types of tables using the following functions:

* :meth:`~didppy.Model.add_set_table`
* :meth:`~didppy.Model.add_element_table`
* :meth:`~didppy.Model.add_int_table`
* :meth:`~didppy.Model.add_float_table`

In the case of :meth:`~didppy.Model.add_set_table`, we can pass a :class:`list` (or a :class:`dict`) of :class:`list` or :class:`set` in Python with specifying the object type.
See :meth:`~didppy.Model.add_set_table` and an :doc:`advanced tutorial <advanced-tutorials/talent-scheduling>` for more details.

The benefit of defining a table is that we can access its value using state variables as indices, as explained later.

Transitions
~~~~~~~~~~~

The recursive equation of the DP model is defined by *transitions*.
A transition transforms the state on the left-hand side into the state on the right-hand side.

In TSPTW, we have the following recursive equation:

.. math::
    V(U, i, t ) = \min\limits_{j \in U: t + c_{ij} \leq b_j} c_{ij} + V(U \setminus \{ j \}, j, \max \{ t + c_{ij}, a_j \})  \text{ if } U \neq \emptyset.

In DIDPPy, it is represented by a set of transitions.

.. code-block:: python

    for j in range(1, n):
        visit = dp.Transition(
            name="visit {}".format(j),
            cost=travel_time[location, j] + dp.IntExpr.state_cost(),
            preconditions=[
                unvisited.contains(j),
                time + travel_time[location, j] <= due_time[j]
            ],
            effects=[
                (unvisited, unvisited.remove(j)),
                (location, j),
                (time, dp.max(time + travel_time[location, j], ready_time[j]))
            ],
        )
        model.add_transition(visit)

The *cost expression* :code:`cost` defines how the value of the left-hand side state, :math:`V(U, i, t)`, is computed based on the value of the right-hand side state, :math:`V(U \setminus \{ j \}, j, \max\{ t + c_{ij}, a_j \})`, represented by :meth:`didppy.IntExpr.state_cost`.
In the case of the continuous cost, we can use :meth:`didppy.FloatExpr.state_cost`.

We can use the values of state variables in the **left-hand side state** in :code:`cost`, :code:`preconditions`, and :code:`effects`.
For example, :code:`location` corresponds to :math:`i` in :math:`V(U, i, t)`, so :code:`travel_time[location, j]` corresponds to :math:`c_{ij}`.
Because :code:`location` is a state variable, :code:`travel_time[location, j]` is not just an :class:`int` but an *expression* (:class:`~didppy.IntExpr`), whose value is determined given a state inside the solver.
Therefore, we cannot use :code:`c[location][j]` and need to register :code:`c` to the model as :code:`travel_time`.
Also, :code:`travel_time[location, j]` must be used instead of :code:`travel_time[location][j]`.
For :code:`ready_time` and :code:`due_time`, we can actually use :code:`a` and :code:`b` instead because they are not indexed by state variables.

*Preconditions* :code:`preconditions` make sure that the transition is considered only when :math:`j \in U` (:code:`unvisited.contains(j)`) and :math:`t + c_{ij} \leq b_j` (:code:`time + travel_time[location, j] <= due_time[j]`).
The value of the left-hand side state is computed by taking the minimum (maximum for maximization) of :code:`cost` over all transitions whose preconditions are satisfied by the state.
:code:`preconditions` are defined by a :class:`list` of :class:`~didppy.Condition`.

*Effects* :code:`effects` describe how the right-hand side state is computed based on the left-hand side state.
Effects are described by a :class:`list` of :class:`tuple` of a state variable and its updated value described by an expression.

* :math:`U \setminus \{ j \}` : :code:`unvisited.remove(j)` (:class:`~didppy.SetExpr`).
* :math:`j` : :code:`j` (automatically converted from :class:`int` to :class:`~didppy.ElementExpr`).
* :math:`\max\{ t + c_{ij}, a_j \}` : :code:`dp.max(time + travel_time[location, j], ready_time[j])` (:class:`~didppy.IntExpr`).

:class:`~didppy.SetVar`, :class:`~didppy.SetExpr` and :class:`~didppy.SetConst` have a similar interface as :class:`set` in Python, e.g., they have methods :meth:`~didppy.SetVar.contains`, :meth:`~didppy.SetVar.add`, :meth:`~didppy.SetVar.remove` which take an :class:`int`, :class:`~didppy.ElementVar`, or :class:`~didppy.ElementExpr` as an argument.

We use :func:`didppy.max` instead of built-in :func:`max` to take the maximum of two :class:`~didppy.IntExpr`.
As in this example, some built-in functions are replaced by :ref:`functions in DIDPPy <api-reference:Functions>` to support expressions.
However, we can apply built-in :func:`sum`, :func:`abs`, and :func:`pow` to :class:`~didppy.IntExpr`.

The equation

.. math::
    V(U, i, t) = c_{i0} + V(U, 0, t + c_{i0}) \text{ if } U = \emptyset \land i \neq 0

is defined by another transition in a similar way.

.. code-block:: python

    return_to_depot = dp.Transition(
        name="return",
        cost=travel_time[location, 0] + dp.IntExpr.state_cost(),
        effects=[
            (location, 0),
            (time, time + travel_time[location, 0]),
        ],
        preconditions=[unvisited.is_empty(), location != 0]
    )
    model.add_transition(return_to_depot)

The effect on :code:`unvisited` is not defined because it is not changed.

Once a transition is created, it is registered to a model by :meth:`~didppy.Model.add_transition`.
We can define a *forced transition*, by using :code:`forced=True` in this function while it is not used in TSPTW.
A forced transition is useful to represent dominance relations between transitions in the DP model.
See an :doc:`advanced tutorial <advanced-tutorials/talent-scheduling>` for more details.

Base Cases
~~~~~~~~~~

A *base cases* is a set of conditions to terminate the recursion.
In our DP model,

.. math::
    V(U, i, t) = 0 \text{ if } U = \emptyset \land i = 0

is a base case.
In DIDPPy, a base case is defined by a :class:`list` of :class:`~didppy.Condition`.

.. code-block:: python

    model.add_base_case([unvisited.is_empty(), location == 0])

When all conditions in a base case are satisfied, the value of the state is 0, and no further transitions are applied.
We can define multiple base cases (not multiple conditions in the same base case) by using :meth:`~didppy.Model.add_base_case` multiple times.
In that case, the value of a state is 0 if any of the base cases is satisfied.

If we want to define conditions with which a state has a non-zero constant value, we need to introduce a dummy transition to the base case, which increases the cost by the constant.
Indeed, the transition :code:`return` can be viewed as such a dummy transition for an equation :math:`V(U, i, t) = c_{i0} \text{ if } U = \emptyset`. 

Solving the Model
-----------------

Now, we have defined a DP model.
Let's use the :class:`~didppy.CABS` solver to solve this model.

.. code-block:: python

    solver = dp.CABS(model, time_limit=10)
    solution = solver.search()

    print("Transitions to apply:")

    for t in solution.transitions:
        print(t.name)

    print("Cost: {}".format(solution.cost))


:meth:`~didppy.CABS.search` returns a :class:`~didppy.Solution`, from which we can extract the transitions to reach a base case from the target state and the cost of the solution.
:class:`~didppy.CABS` is an anytime solver, which returns the best solution found within the time limit.
Instead of :meth:`~didppy.CABS.search`, we can use :meth:`~didppy.CABS.search_next`, which returns the next solution found.
:class:`~didppy.CABS` is complete, which means that it returns an optimal solution given enough time.
If we use :code:`time_limit=None`, it continues to search until an optimal solution is found.
Whether the returned solution is optimal or not can be checked by :attr:`didppy.Solution.is_optimal`.

While :class:`~didppy.CABS` is usually the most efficient solver, it has some restrictions:
it solves the DP model as a path-finding problem in a graph, so it is only applicable to particular types of DP models.
Concretely, :code:`cost` in all transitions must have either of the following structure:

* :code:`w + dp.IntExpr.state_cost()`
* :code:`w * dp.IntExpr.state_cost()`
* :code:`dp.max(w, dp.IntExpr.state_cost())`
* :code:`dp.min(w, dp.IntExpr.state_cost())`

where :code:`w` is an :class:`~didppy.IntExpr` independent of :meth:`~didppy.IntExpr.state_cost`.
For float cost, we can use :class:`~didppy.FloatExpr` instead of :class:`~didppy.IntExpr`.
By default, :class:`~didppy.CABS` assumes that :code:`cost` is the additive form.
For other types of :code:`cost`, we need to tell the solver by using the argument :code:`f_operator`, which takes either of :attr:`didppy.FOperator.Plus`, :attr:`didppy.FOperator.Product`, :attr:`didppy.FOperator.Max`, or :attr:`didppy.FOperator.Min` (:attr:`~didppy.FOperator.Plus` is the default).
An example is provided in an :doc:`advanced tutorial <advanced-tutorials/mosp>`.

If your problem does not fit into the above structure, you can use :class:`~didppy.ForwardRecursion`, which is the most generic but might be an inefficient solver.
For further details, see :doc:`the guide for the solver selection <solver-selection>` as well as :ref:`the API reference <api-reference:Solvers>`.

Improving the DP Formulation 
----------------------------

So far, we defined the DP formulation for TSPTW, model it in DIDPPy, and solved the model using a solver.
However, the formulation above is **not efficient**.
Actually, we can improve the formulation by incorporating more information.
Such information is unnecessary to define a problem but potentially helps a solver.
We introduce three enhancements to the DP formulation.

Dominance Between States
~~~~~~~~~~~~~~~~~~~~~~~~

Consider two states :math:`(U, i, t)` and :math:`(U, i, t')` with :math:`t \leq t'`, which share the set of unvisited customers and the current location.
In TSPTW, smaller :math:`t` is always better, so :math:`(U, i, t)` leads to a better solution than :math:`(U, i, t')`.
Therefore, we can introduce the following inequality:

.. math::
    V(U, i, t) \leq V(U, i, t') \text{ if } t \leq t'.

With this information, a solver may not need to consider :math:`(U, i, t')` if it has already considered :math:`(U, i, t)`.

Looking Ahead the Deadlines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In TSPTW, all customers must be visited before their deadlines.
In a state :math:`(U, i, t)`, if the salesperson cannot visit customer :math:`j \in U` before :math:`b_j`, the subproblem defined by this state does not have a solution.
The earliest possible time to visit :math:`j` is :math:`t + c_{ij}` (we assume the triangle inequality, :math:`c_{ik} + c_{kj} \geq c_{ij}`).
Therefore, if :math:`t + c_{ij} > b_j`, we can conclude that :math:`(U, i, t)` does not have a solution.
This inference is formulated as the following equation:

.. math::
    V(U, i, t) = \infty \text{ if } \exists j \in U, t + c_{ij} > b_j.

A solver can prune a state if it satisfies the above condition.

Lower Bounds on the Value Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In model-based approaches such as mixed-integer programming (MIP), modeling the bounds on the objective function is commonly used to improve the efficiency of a solver.
In the case of DIDP, we consider bounding the value function :math:`V` for a state :math:`(U, i, t)`.

The lowest possible travel time to visit customer :math:`j` is :math:`\min_{k \in N \setminus \{ j \}} c_{kj}`.
Because we need to visit all customers in :math:`U`, the total travel time is at least

.. math::
    \sum_{j \in U} \min_{k \in N \setminus \{ j \}} c_{kj}.

Furthermore, if the current location :math:`i` is not the depot, we need to visit the depot.
Therefore,

.. math::
    V(U, i, t) \geq \sum_{j \in (U \cup \{ 0 \}) \setminus \{ i \} } \min_{k \in N \setminus \{ j \}} c_{kj}.

Similarly, we need to depart from each customer in :math:`U` and the current location :math:`i` if :math:`i` is not the depot.
Therefore,

.. math::
    V(U, i, t) \geq \sum_{j \in (U \cup \{ i \}) \setminus \{ 0 \} } \min_{k \in N \setminus \{ j \}} c_{jk}.

Full Formulation
~~~~~~~~~~~~~~~~

Overall, our model is now as follows:

.. math::
    \text{compute } & V(N \setminus \{ 0 \}, 0, 0) \\ 
    & V(U, i, t) = \begin{cases}
         \infty & \text{if } \exists j \in U, t + c_{ij} > b_j \\
         \min\limits_{j \in U} c_{ij} + V(U \setminus \{ j \}, j, \max \{ t + c_{ij}, a_j \})  & \text{else if } U \neq \emptyset \\
         c_{i0} + V(U, 0, t + c_{i0}) & \text{else if } U = \emptyset \land i \neq 0 \\
         0 & \text{else if } U = \emptyset \land i = 0.
    \end{cases} \\
    & V(U, i, t) \leq V(U, i, t') \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ~ \text{ if } t \leq t' \\
    & V(U, i, t) \geq \sum_{j \in (U \cup \{ 0 \}) \setminus \{ i \} } \min_{k \in N \setminus \{ j \}} c_{kj} \\
    & V(U, i, t) \geq \sum_{j \in (U \cup \{ i \}) \setminus \{ 0 \} } \min_{k \in N \setminus \{ j \}} c_{jk}.

Note that in the second line, :math:`t + c_{ij} \leq b_j` for :math:`j \in U` is ensured by the first line.

Improved Model in DIDPPy 
------------------------

Now, let's model the improved formulation in DIDPPy.

Resource Variables
~~~~~~~~~~~~~~~~~~

The dominance of states, :math:`V(U, i, t) \leq V(U, i, t') \text{ if } t \leq t'`, can be modeled by *resource variables*.

.. code-block:: python

   # U
   unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
   # i
   location = model.add_element_var(object_type=customer, target=0)
   # t (resource variable)
   time = model.add_int_resource_var(target=0, less_is_better=True)

Now, :code:`time` is an :class:`~didppy.IntResourceVar` created by :meth:`~didppy.Model.add_int_resource_var` instead of :meth:`~didppy.Model.add_int_var`, with the preference :code:`less_is_better=True`.
This means that if the other state variables have the same values, a state having a smaller value of :code:`time` is better.
If :code:`less_is_better=False`, a state having a larger value is better.

There are three types of resource variables in DIDPPy:

* :class:`~didppy.IntResourceVar`
* :class:`~didppy.FloatResourceVar`
* :class:`~didppy.ElementResourceVar`

State Constraints
~~~~~~~~~~~~~~~~~

*State constraints* are constraints that must be satisfied by all states.
In other words, a state that does not satisfy the state constraints can be immediately pruned.

In our DP formulation, we have the following equation, which defines the condition when a state does not have a solution:

.. math::
    V(U, i, t) = \infty \text{ if } \exists j \in U, t + c_{ij} > b_j.

We can model this equation in the DP model by using the negation of the condition, :math:`\forall j \in U, t + c_{ij} \leq b_j`, as state constraints:

.. code-block:: python

    for j in range(1, n):
        model.add_state_constr(
            ~unvisited.contains(j) | (time + travel_time[location, j] <= due_time[j])
        )

For each customer :code:`j`, we define a disjunctive condition :math:`j \notin U \lor t + c_{ij} \leq b_j`.
:code:`~` is the negation operator of :class:`~didppy.Condition`, and :code:`|` is the disjunction operator.
We can also use :code:`&` for the conjunction.
We cannot use :code:`not`, :code:`or`, and :code:`and` in Python because they are only applicable to :class:`bool` in Python.

State constraints are different from preconditions of transitions.
State constraints are evaluated each time a state is generated while preconditions are evaluated only when a transition is taken.

Dual Bounds
~~~~~~~~~~~

In DIDP, lower bounds for minimization and upper bounds for maximization are called *dual bounds*.
In our DP formulation, the following inequalities define the dual bounds:

.. math::
    & V(U, i, t) \geq \sum_{j \in (U \cup \{ 0 \}) \setminus \{ i \} } \min_{k \in N \setminus \{ j \}} c_{kj} \\
    & V(U, i, t) \geq \sum_{j \in (U \cup \{ i \}) \setminus \{ 0 \} } \min_{k \in N \setminus \{ j \}} c_{jk}.

These bounds are modeled as follows:

.. code-block:: python

    min_to = model.add_int_table(
        [min(c[k][j] for k in range(n) if k != j) for j in range(n)]
    )

    model.add_dual_bound(min_to[unvisited] + (location != 0).if_then_else(min_to[0], 0))

    min_from = model.add_int_table(
        [min(c[j][k] for k in range(n) if k != j) for j in range(n)]
    )

    model.add_dual_bound(
        min_from[unvisited] + (location != 0).if_then_else(min_from[location], 0)
    )

We first register :math:`\min\limits_{k \in N \setminus \{ j \}} c_{kj}` to the model as a table :code:`min_to`.
:code:`min_to[unvisited]` represents :math:`\sum\limits_{j \in U} \min\limits_{k \in N \setminus \{ j \}} c_{kj}`,  i.e., the sum of values in :code:`min_to` for customers in :code:`unvisited`.
Similarly, :code:`min_to.product(unvisited)` :code:`min_to.max(unvisited)`, and :code:`min_to.min(unvisited)` can be used to take the product, maximum, and minimum.
We can do the same for tables with more than one dimension.
For example, if :code:`table` is a two-dimensional table, :code:`table[unvisited, unvisited]` takes the sum over all pairs of customers in :code:`unvisited`, and :code:`table[unvisited, location]` takes the sum of :code:`table[i, location]` where :code:`i` iterates through customers in :code:`unvisited`.

When the current location is not the depot, i.e., :code:`location != 0`, :math:`\min\limits_{k \in N \setminus \{ 0 \}} c_{k0}` (:code:`min_to[0]`) is added to the dual bound, which is done by :meth:`~didppy.Condition.if_then_else`.

We repeat a similar procedure for the other dual bound.

**Defining a dual bound in DIDP is extremely important**: a dual bound may significantly boost the performance of solvers.
We strongly recommend defining a dual bound even if it is trivial, such as :math:`V(U, i, t) \geq 0`.

Full Code
~~~~~~~~~

Here is the full code for the DP model:

.. code-block:: python

    import didppy as dp


    # Number of locations
    n = 4
    # Ready time
    a = [0, 5, 0, 8]
    # Due time
    b = [100, 16, 10, 14]
    # Travel time
    c = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]

    model = dp.Model(maximize=False, float_cost=False)

    customer = model.add_object_type(number=n)

    # U
    unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
    # i
    location = model.add_element_var(object_type=customer, target=0)
    # t (resource variable)
    time = model.add_int_resource_var(target=0, less_is_better=True)

    ready_time = model.add_int_table(a)
    due_time = model.add_int_table(b)
    travel_time = model.add_int_table(c)

    for j in range(1, n):
        visit = dp.Transition(
            name="visit {}".format(j),
            cost=travel_time[location, j] + dp.IntExpr.state_cost(),
            preconditions=[unvisited.contains(j)],
            effects=[
                (unvisited, unvisited.remove(j)),
                (location, j),
                (time, dp.max(time + travel_time[location, j], ready_time[j])),
            ],
        )
        model.add_transition(visit)

    return_to_depot = dp.Transition(
        name="return",
        cost=travel_time[location, 0] + dp.IntExpr.state_cost(),
        effects=[
            (location, 0),
            (time, time + travel_time[location, 0]),
        ],
        preconditions=[unvisited.is_empty(), location != 0],
    )
    model.add_transition(return_to_depot)

    model.add_base_case([unvisited.is_empty(), location == 0])

    for j in range(1, n):
        model.add_state_constr(
            ~unvisited.contains(j) | (time + travel_time[location, j] <= due_time[j])
        )

    min_to = model.add_int_table(
        [min(c[k][j] for k in range(n) if k != j) for j in range(n)]
    )

    model.add_dual_bound(min_to[unvisited] + (location != 0).if_then_else(min_to[0], 0))

    min_from = model.add_int_table(
        [min(c[j][k] for k in range(n) if k != j) for j in range(n)]
    )

    model.add_dual_bound(
        min_from[unvisited] + (location != 0).if_then_else(min_from[location], 0)
    )

    solver = dp.CABS(model)
    solution = solver.search()

    print("Transitions to apply:")

    for t in solution.transitions:
        print(t.name)

    print("Cost: {}".format(solution.cost))

Next Steps
----------

Congratulations! You have finished the tutorial.

We covered fundamental concepts of DIDP modeling and advanced techniques to improve the performance of the model.

* Several features that did not appear in the DP model for TSPTW are covered in the :doc:`advanced tutorials <advanced-tutorials>`.
* `More examples <https://github.com/domain-independent-dp/didp-rs/tree/main/didppy/examples>`_ are provided in our repository as Jupyter notebooks.
* :doc:`The API reference <api-reference>` describes each class and function in detail.
* If your model does not work as expected, :doc:`the debugging guide <debugging>` might help you.
* If you want to know the algorithms used in the solvers, we recommend reading :cite:t:`DIDPAnytime`.
* Our papers on which DIDPPy is based are listed on :doc:`this page <papers>`.