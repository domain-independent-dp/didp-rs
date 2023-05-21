Solvers for General Costs
=========================

In this tutorial, we will see how to configure solvers for more general costs than :code:`w + dp.IntExpr.state_cost()` using the minimization of open stacks problem (MOSP) as an example.

MOSP
----

In MOSP, we are given a set of customers :math:`C = \{ 0, ..., n-1 \}` and a set of products :math:`P = \{ 0, ..., m-1 \}`.
Each customer :math:`c \in C` requests a subset of products :math:`P_c \subseteq P`.
We want to decide the order to produce the products.
Once we produced a product :math:`p \in P_c`, we need to open a stack for customer :math:`c` to store the product.
When we produced all products in :math:`P_c`, we can close the stack for :math:`c`.
We want to minimize the maximum number of open stacks at a time.

DP Formulation
--------------

The DP formulation is based on :cite:t:`Chu2009`.
The approach is to find the order of customers to close stacks instead of the order of products to produce.
Once the order of customers is determined, for each customer, products requested by the customer that are not produced yet are consecutively produced in an arbitrary order.
:cite:t:`Chu2009` proved that this approach finds an optimal solution.

When we close the stack for customer :math:`c`, we need to produce all products in :math:`P_c`.
If another customer :math:`c'` requests a product in :math:`P_c` and its stack is not opened yet, we need to open the stack for :math:`c'`.
In a sense, we can say that :math:`c'` is a neighbor of :math:`c`.
Let :math:`N_c \subseteq C` be the set of neighbors including :math:`c`, i.e.,

.. math::
    N_c = \{ c' \in C \mid P_{c'} \cap P_c \neq \emptyset \}.

Let :math:`O` be the set of customers whose stacks have been opened.
When we are producing the products requested by :math:`c`, we need to open new stacks for customers :math:`N_c \setminus O`.
Let :math:`R` be the set of customers whose stacks are not closed yet.
Because the set of customers whose stacks have been opened and not closed is :math:`O \cap R`, the number of open stacks when producing the products for :math:`c` is

.. math::
    |(O \cap R) \cup (N_c \setminus O)|

When we close the stack for :math:`c`, the set of customers whose stacks are not closed becomes :math:`R \setminus \{ c \}`, and the set of customers whose stacks have been opened becomes :math:`O \cup N_c`.
Let :math:`V(R, O)` be the minimum of the maximum number of open stacks at a time to close the stacks for customers in :math:`R` when the stacks for customers in :math:`O` have been opened.
Then, the DP formulation is

.. math::
    \text{compute } & V(C, \emptyset) \\
    & V(R, O) = \begin{cases}
        \min\limits_{c \in R} \max\left\{ |(O \cap R) \cup (N_c \setminus O)|,  V(R \setminus \{ c \}, O \cup N_c) \right\} & \text{if } R \neq \emptyset \\
        0 & \text{if } R = \emptyset
    \end{cases} \\
    & V(R, O) \geq 0.

Modeling in DIDPPy
~~~~~~~~~~~~~~~~~~

Modeling the above DP formulation in DIDPPy is not difficult though it may look complicated.
Assume that the data is preprocessed, and we are given :math:`N_c` for each :math:`c \in C` instead of :math:`P_c`. 

.. code-block:: python

    import didppy as dp

    # Number of customers
    n = 4
    # Neighbors
    neighbors = [[0, 1], [0, 1, 3], [2], [1,    3]]

    model = dp.Model()

    customer = model.add_object_type(number=n)

    # R
    remaining = model.add_set_var(object_type=customer, target=list(range(n)))
    # O
    opened = model.add_set_var(object_type=customer, target=[])

    neighbor_table = model.add_set_table(neighbors, object_type=customer)

    for c in range(n):
        close = dp.Transition(
            name="close {}".format(c),
            cost=dp.max(
                ((opened & remaining) | (neighbor_table[c] - opened)).len(),
                dp.IntExpr.state_cost(),
            ),
            effects=[
                (remaining, remaining.remove(c)),
                (opened, opened | neighbor_table[c]),
            ],
            preconditions=[remaining.contains(c)],
        )
        model.add_transition(close)

    model.add_base_case([remaining.is_empty()])

    model.add_dual_bound(0)

We can take the cardinality of :class:`~didppy.SetVar` and :class:`~didppy.SetExpr` as :class:`~didppy.IntExpr` using :meth:`~didppy.SetExpr.len`.
Now, :code:`cost` is the maximum of an :class:`~didppy.IntExpr` and :meth:`~didppy.IntExpr.state_cost`.
Note that here, :meth:`~didppy.IntExpr.state_cost` is :meth:`didppy.IntExpr.state_cost`, so it is an :class:`~didppy.IntExpr`.
The function :func:`didppy.max` takes the maximum of two :class:`~didppy.IntExpr` and returns an :class:`~didppy.IntExpr`.

Configuring Solvers for General Costs
-------------------------------------

In the above model, the form of :code:`cost` is different from what we observed in the previous models:
it takes the maximum of an :class:`~didppy.IntExpr` and :meth:`~didppy.IntExpr.state_cost` instead of addition.
Still, we can use path-finding based solvers such as :class:`~didppy.CABS` as long as the :class:`~didppy.IntExpr` is independent of :meth:`~didppy.IntExpr.state_cost` if we tell the cost form to the solver.

.. code-block:: python

    solver = dp.CABS(model, f_operator=dp.FOperator.Max)
    solution = solver.search()


The argument :code:`f_operator` takes an instance of :class:`~didppy.FOperator` to specify the form of the cost expression.
Because we take the maximum, we use :attr:`~didppy.FOperator.Max`.
The default value is :attr:`~didppy.FOperator.Plus`, which means that the cost expression is in the form of the addition.

:class:`~didppy.CABS` and other path-finding based solvers can handle the product and minimum as well if we use :attr:`~didppy.FOperator.Product` and :attr:`~didppy.FOperator.Min` for :code:`f_operator`, respectively.

If we use the most generic (but potentially inefficient) solver, :class:`~didppy.ForwardRecursion`, we do not need such a configuration.

.. code-block:: python

    solver = dp.ForwardRecursion(model)
    solution = solver.search()

