Transition Dominance
====================

In this tutorial, we introduce the concept of transition dominance and how to model it in DIDPPy.

Graph-Clear
-----------

In a graph-clear problem, we are given a floormap represented by an undirected graph :math:`(N, E)`, where :math:`N = \{ 0, ..., n - 1 \}` is the set of nodes, and :math:`E \subseteq N \times N` is the set of edges.
Each node corresponds to a room, and edges are corridors connecting two rooms.
We want to clear intruders in the floor using robots.
At each time step, we can clear a node :math:`i` using :math:`a_i` robots to sweep the room and use :math:`b_{ij}` robots to block each incident edge :math:`\{ i, j \}`.
At the beginning, all nodes are contaminated, i.e., potentially include intruders.
Even if a node is swept, if there exists a non-blocked path from a contaminated node to that node, it become contaminated again in the next time step.
Therefore, we may want to block edges that are not directly connected to the currently swept node.
We want to find a schedule over time steps to clear all nodes while minimizing the maximum number of robots used at a time.

DP Formulation
--------------

It is proved that there exists an optimal schedule where an already swept node never becomes contaminated again :cite:p:`Morin2018`.
We just need to clear a node one by one while blocking all edges connected to already swept nodes.
Let :math:`C \subseteq N` be the set of already swept nodes, and assume that :math:`b_{ij} = 0` if :math:`\{ i, j \} \notin E`.
To clear node :math:`c \in \overline{C} = N \setminus C`, we need to use :math:`a_c` robots to sweep :math:`c`, :math:`\sum_{i \in N} b_{ci}` robots to block the edges incident to :math:`c`, and :math:`\sum_{i \in C} \sum_{j \in \overline{C} \setminus \{ c \}} b_{ij}` robots to block the edges connected to already swept nodes.
Therefore,

.. math::

    \text{compute } & V(\emptyset) \\
    & V(C) = \begin{cases}
    \min\limits_{c \in \overline{C}} \max\left\{ a_c + \sum\limits_{i \in N} b_{ci} + \sum\limits_{i \in C} \sum\limits_{j \in \overline{C} \setminus \{ c \}} b_{ij}, V(C \cup \{ c \}) \right\} & \text{if } C \neq N \\
    0 & \text{if } C = N
    \end{cases} \\
    & V(C) \geq 0.

Modeling in DIDPPy
------------------

Let's model the above DP formulation in DIDPPy.

.. code-block:: python

    import didppy as dp

    # Number of nodes
    n = 4
    # Node weights
    a = [1, 2, 2, 3]
    # Edge weights
    b = [
        [0, 2, 3, 0],
        [2, 0, 0, 1],
        [3, 0, 0, 2],
        [0, 1, 2, 0],
    ]

    model = dp.Model()

    node = model.add_object_type(number=n)

    # C
    clean = model.add_set_var(object_type=node, target=[])

    all_nodes = model.create_set_const(object_type=node, value=list(range(n)))
    model.add_base_case([clean == all_nodes])

    edge_weight = model.add_int_table(b)
    # Node weight plus the sum of the edge weights
    node_edge_weight = model.add_int_table([a[i] + sum(b[i]) for i in range(n)])
    # State function to cache the complement set
    contaminated = model.add_set_state_fun(clean.complement())
    transition_ids = []

    for c in range(n):
        sweep = dp.Transition(
            name="sweep {}".format(c),
            cost=dp.max(
                dp.IntExpr.state_cost(),
                node_edge_weight[c] + edge_weight[clean, contaminated.remove(c)]
            ),
            effects=[(clean, clean.add(c))],
            preconditions=[~clean.contains(c)],
        )
        current_id = model.add_transition(sweep)
        transition_ids.append(current_id)

    model.add_dual_bound(0)

The state variable :code:`clean` represents the set of swept nodes (i.e., :math:`C`).
We introduce a constant :code:`all_nodes` to represent the set of all nodes (i.e., :math:`N`).

We define an int table :code:`edge_weight` representing the edge weights :math:`b_{ij}`.
We also define an int table :code:`node_edge_weight` representing the sum of the node weight and the edge weights incident to that node, i.e., :math:`a_c + \sum_{i \in N} b_{ci}`.

We use a state function :code:`contaminated` to represent the set of contaminated nodes (i.e., :math:`\overline{C}`) by calling :meth:`~didppy.Model.add_set_state_fun`.
A state function is a function of a state defined by an expression.
It is useful to represent information implied by state variables.
A solver can cache the value of a state function to avoid redundant computation if it is used in multiple places.

For each node :math:`c`, we define a transition to sweep that node :code:`sweep`.
Here, we store the id of each transition returned by :meth:`~didppy.Model.add_transition` in a list :code:`transition_ids` to refer to the transition later for transition dominance.

Defining Transition Dominance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a node :math:`i` and :math:`j`, when the following condition holds, it is known that sweeping $i$ now is at least as good as sweeping :math:`j` now :cite:p:`TransitionDominance`.

.. math::

  & a_i + \sum_{k \in \overline{C}} b_{ik} \leq a_{j} + \sum_{k \in \overline{C}} b_{jk} \\  
  & \sum_{k \in \overline{C}} b_{ik} \leq \sum_{k \in C} b_{ik}.

This knowledge can be modeled as a transition dominance in DIDP.

.. code-block:: python

    clean_edge_weights = [
        model.add_int_state_fun(edge_weight[i, clean]) for i in range(n)
    ]
    contaminated_edge_weights = [
        model.add_int_state_fun(edge_weight[i, contaminated]) for i in range(n)
    ]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            condition1 = a[i] + contaminated_edge_weights[i] <= a[j] + contaminated_edge_weights[j]
            condition2 = contaminated_edge_weights[i] <= clean_edge_weights[i]

            model.add_transition_dominance(
                transition_ids[i],
                transition_ids[j],
                conditions=[condition1, condition2]
            )

We use two state functions :code:`clean_edge_weights` and :code:`contaminated_edge_weights` to represent the edge weights of the nodes in the clean set and contaminated set, respectively.
They are computed multiple times in the transition dominance conditions, so we define them as state functions to cache the values.

With :meth:`~didppy.Model.add_transition_dominance`, we can define a transition dominance between two transitions.
The first argument is the ID of the transition that potentially dominates the transition specified by the second argument.
:code:`conditions` is a list of conditions that must hold for the dominance to be valid.
By default, :code:`conditions` is :code:`None`, which means the dominance is unconditional.
When the first transition is applicable and the conditions hold, the second transition can be ignored by a solver as long as the first transition is considered.
It is possible that two transitions are dominated by each other, or multiple transitions forms a cyclic dependency, in which case the solver can choose one.

Note that :doc:`forced transitions <forced-transitions>` can be viewed as a special case of transition dominance, where it dominates all the other transitions.
Thus, we cannot define transition dominance for forced transitions.