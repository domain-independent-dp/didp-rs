Advanced Tutorial
=================

This tutorial covers several features that are not covered in the :doc:`first tutorial <tutorial>`.
We use the following two problems:

* :ref:`Talent scheduling problem <advanced-tutorial:Talent Scheduling>`: we demonstrate how to use forced transitions and show diverse operations on :class:`~didppy.SetExpr`.
* :ref:`Minimization of open stacks problem <advanced-tutorial:MOSP>`: we demonstrate how to handle a more general form of :code:`cost` in :class:`~didppy.Transition`.

Talent Scheduling
-----------------

In a talent scheduling problem, we are given a set of scenes :math:`S = \{ 0, ..., n - 1 \}` and a set of actors :math:`A = \{ 0, ..., m - 1 \}`.
In a scene :math:`s \in S`, a set of actors :math:`A_s \subseteq A` plays for :math:`d_s` days.
An actor comes to the location when the first scene he or she plays starts and leaves when the last scene he or she plays ends.
For each day actor :math:`a` is on location, we need to pay the cost :math:`c_a`.
We want to find a sequence of scenes to shoot such that the total cost is minimized.

DP Formulation for Talent Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DP formulation is based on :cite:t:`GarciaDeLaBanda2011`.
Suppose that a set of scenes :math:`Q` is remaining.
A set of actors :math:`\bigcup_{s \in N \setminus Q} A_s` already came to the location, and :math:`\bigcup_{s \in Q} A_s` is still on location because they need to play on the remaining scenes :math:`Q`.
Therefore, if we shoot a scene :math:`s \in Q` next, the set of actors on location will be

.. math::

    L(s, Q) = A_s \cup \left( \bigcup_{s' \in N \setminus Q} A_{s'} \cap \bigcup_{s' \in Q } A_{s'}  \right).

We need to pay the cost :math:`d_s \sum_{a \in L(s, Q)} c_a` when shooting scene :math:`s`.
Once we shot scene :math:`s`, the remaining problem is to decide the order of the remaining scenes :math:`Q \cup \{ s \}.`
Therefore, a state is defined by the set of remaining scenes :math:`Q`, and the minimum cost to shoot :math:`Q` is represented by :math:`V(Q)`.
Because :math:`A_s`, actors who play in scence :math:`s`, are always on location when :math:`s` is shot, :math:`\sum_{s \in Q} d_s \sum_{a \in A_s} c_a` is a lower bound on :math:`V(Q)`.
We have the following DP formulation.

.. math::

    \text{compute } & V(N) \\
    & V(Q) = \begin{cases}
        \min\limits_{s \in Q} d_s \sum\limits_{a \in L(s, Q)} c_a + V(Q \setminus \{ s \}) & \text{if } Q \neq N \\
        0 & \text{if } Q = \emptyset
    \end{cases} \\
    & V(Q) \geq \sum_{s \in Q} d_s \sum_{a \in A_s} c_a.

Scheduling without Extra Cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :math:`A_s`, the set of actors that play in scence :math:`s`, is equivalent to the set of actors currently on location, we can shoot :math:`s` with the minimum cost:
we just need to pay for the actors who play in :math:`s`.
We should always shoot such a scene first.
In state :math:`Q`, the set of actors on location is

.. math::

    \bigcup_{s \in N \setminus Q} A_{s} \cap \bigcup_{s \in Q} A_{s}.

Therefore, we have the following equation:

.. math::

    V(Q) = d_s \sum\limits_{a \in A_s} c_a + V(Q \setminus \{ s \}) \text{ if } s \in Q \land A_s = \bigcup_{s' \in N \setminus Q} A_{s'} \cap \bigcup_{s' \in Q} A_{s'}.

If multiple scenes satisfy the condition, we can shoot any of them.
This equation helps a solver because it tells that other transitions are not needed to be considered.

DIDPPy for Talent Scheduling 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To model the above equation, we can use forced transitions.
Before defining forced transitions, let's model the other parts of the formulation.

.. code-block:: python

    import didppy as dp


    # Number of scenes
    n = 4
    # Number of actors
    m = 4
    # Duration of scenes
    d = [1, 1, 1, 1]
    # Costs of actors
    c = [1, 3, 1, 2]
    # Actors in each scene
    capital_a = [[0, 1, 3], [1, 2], [0, 2, 3], [0, 1, 2]]

    model = dp.Model()

    scene = model.add_object_type(number=n)
    actor = model.add_object_type(number=m)

    # Q
    remaining = model.add_set_var(object_type=scene, target=list(range(n)))

    scene_to_actors = model.add_set_table(capital_a, object_type=scene)
    actor_to_cost = model.add_int_table(c)

    # Precompute the minimum cost of each scene
    scene_to_min_cost = model.add_int_table(
        [d[s] * sum(c[a] for a in capital_a[s]) for s in range(n)]
    )

    for s in range(n):
        already_shot = remaining.complement()
        came_to_location = scene_to_actors.union(already_shot)
        standby = scene_to_actors.union(remaining)
        on_location = scene_to_actors[s] | (came_to_location & standby)

        shoot = dp.Transition(
            name="shoot {}".format(s),
            cost=d[s] * actor_to_cost[on_location] + dp.IntExpr.state_cost(),
            preconditions=[remaining.contains(s)],
            effects=[(remaining, remaining.remove(s))],
        )
        model.add_transition(shoot)

    model.add_base_case([remaining.is_empty()])

    model.add_dual_bound(scene_to_min_cost[remaining])

The state variable :code:`remaining` represents the set of remaining scenes.
With :func:`~didppy.StateVar.complement`, we can get the complement of :code:`remaining`, which is the set of already shot scenes :math:`N \setminus Q`.

We define a set table :code:`scene_to_actors` to represent the set of actors in each scene using :func:`~didppy.Model.add_set_table`.
When defining a set table, we can use a :class:`list` of :class:`list` or :class:`set`, but we need to specify the object type using :code:`object_type` argument.
Alternately, we can use a list of :class:`~didppy.SetConst`, which does not requore :code:`object_type` as it is specified when created by :func:`~didppy.Model.create_set_const`.

By using the :func:`~didppy.SetTable1D.union` method of a table, we can get the union of sets corresponding to the elements in the set (:class:`~didppy.SetVar`, :class:`~didppy.SetExpr`, or :class:`~didppy.SetConst`) given as an argument.
Therefore, :code:`scene_to_actors.union(remaining)` corresponds to :math:`\bigcup_{s \in Q} A_s`.

The union and intersection of two sets can be represented by the bitwise OR operator :code:`|` and AND operator :code:`&`.

Forced Transition
~~~~~~~~~~~~~~~~~

Now, let's model the following equation using forced transitions.

.. math::

    V(Q) = d_s \sum\limits_{a \in A_s} c_a + V(Q \setminus \{ s \}) \text{ if } s \in Q \land A_s = \bigcup_{s' \in N \setminus Q} A_{s'} \cap \bigcup_{s' \in Q} A_{s'}.

Because which :math:`s` satisfies the condition is unknown, we need to define a transition for each :math:`s`.

.. code-block:: python

    for s in range(n):
        already_shot = remaining.complement()
        came_to_location = scene_to_actors.union(already_shot)
        standby = scene_to_actors.union(remaining)
        on_location = scene_to_actors[s] | (came_to_location & standby)

        shoot = dp.Transition(
            name="forced shoot {}".format(s),
            cost=d[s] * actor_to_cost[scene_to_actors[s]] + dp.IntExpr.state_cost(),
            preconditions=[
                remaining.contains(s),
                scene_to_actors[s] == (came_to_location & standby),
            ],
            effects=[(remaining, remaining.remove(s))],
        )
        model.add_transition(shoot, forced=True)

Now, we have an additional precondition, :code:`scene_to_actors[s] == (came_to_location & standby)`, which corresponds to :math:`A_s = \bigcup_{s' \in N \setminus Q} A_{s'} \cap \bigcup_{s' \in Q} A_{s'}`.
When registering this transition to the model, we use the argument :code:`forced=True` to indicate that this transition is a forced transition.

Ordinarily, DIDPPy takes the minimum (or maximum) :code:`code` over all transitions whose :code:`preconditions` are satisfied. 
However, if :code:`preconditions` of a forced transition are satisfied, DIDPPy ignores other transitions and only considers the forced transition.
If multiple forced transitions are available, DIDPPy selects the first-defined one.
Therefore, **the order to define forced transitions does matter**.

Further optimization
~~~~~~~~~~~~~~~~~~~~

We can further optimize this DP model by considering dominance relations between scenes:
given two scenes :math:`s_1` and :math:`s_2`, when some conditions are satisfied, we can prove that scheduling :math:`s_1` first is always better.
This can be ensured by preconditions: we can add a precondition to the transition for :math:`s_2` that states there is no such :math:`s_1` in :math:`Q`.

We do not go into details here.
If you are interested in this topic, please refer :cite:t:`GarciaDeLaBanda2011` and :cite:t:`DIDPAnytime`.


MOSP
----