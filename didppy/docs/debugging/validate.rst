Validating a Solution
=====================

One way to debug a model is to use a small problem instance and see why an expected solution is not found.
DIDPPy provides :meth:`didppy.Model.validate_solution` to check feasibility and compute the objective value.
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
    pack_id = model.add_transition(pack)

    ignore = dp.Transition(
        name="ignore",
        cost=dp.IntExpr.state_cost(),
        effects=[(i, i + 1)],
        preconditions=[i < n],
    )
    ignore_id = model.add_transition(ignore)

    model.add_base_case([i == n])

    solution = [ignore, pack, pack, ignore]

    try:
        cost = model.validate_solution(solution)
        print(f"Solution is feasible, with cost {cost}.")  # cost is 60
    except dp.ValidationError as error:
        print(f"Invalid solution: {error}")

The method accepts a sequence of :class:`~didppy.Transition` objects or a sequence of :class:`~didppy.TransitionId` objects.
All entries must have the same type; mixing transitions and IDs raises :class:`TypeError`.
For a :class:`didppy.Solution`, including a solver result, pass :code:`solution.transitions`.
It returns an :class:`int` for an integer-cost model and a :class:`float` for a continuous-cost model.
The objective is computed from the final base cost and the transition cost expressions, so nonzero base costs and non-additive objectives are supported.
Validation does not modify the model or the solution.

Each transition must be registered in the model as a transition.
The original transition passed to :meth:`~didppy.Model.add_transition` can be used; internal expression simplification is taken into account.
Preconditions and state constraints must hold, including state constraints at the target.
The sequence must end at the first base state.
An empty sequence is feasible when the target is a base state satisfying the state constraints.
Forced-transition priority, transition dominance, dual bounds, and optimality are not checked.

Using transition IDs
--------------------

:meth:`~didppy.Model.add_transition` returns an ID that can be used directly:

.. code-block:: python

    cost = model.validate_solution([ignore_id, pack_id, pack_id, ignore_id])
    assert cost == 60

IDs resolve directly to the model's registered transitions, avoiding structural membership comparisons and expression simplification.
All feasibility checks and objective evaluation remain the same.
Forward and forward-forced IDs are accepted; backward or out-of-range IDs raise :class:`~didppy.ValidationError` identifying the offending position.

IDs are interpreted relative to the supplied model, just as in :meth:`~didppy.Model.get_transition`.
They do not record model ownership: an ID from another model may refer to a different valid transition in this model.
Use IDs from the model being validated to avoid that ambiguity.
Changing the original :class:`~didppy.Transition` object after registration does not change the transition referenced by its ID.

Understanding failures
----------------------

Validation stops at the first failure and raises :class:`didppy.ValidationError`, a subclass of :class:`ValueError`.
Membership of all transition objects, or resolution of all IDs, is checked in sequence order before feasibility is checked.
An invalid reference later in the sequence is therefore reported before an earlier feasibility failure, including a target-state constraint violation.
Once all references are valid, replay stops at the first feasibility failure.
It does not print ordinary validation failures.
For example, an incomplete sequence produces:

.. code-block:: python

    try:
        model.validate_solution([ignore, pack, pack])
    except dp.ValidationError as error:
        print(error)

.. code-block:: text

    The final state (state 3) does not satisfy any base case.

Transitions and failed preconditions or constraints are numbered from 1 in messages.
State 0 is the target; state 1 is the state after transition 1.
Precondition and constraint numbers refer to the model's internally simplified or grounded conditions, which may differ from their order in the input.
Expression evaluation failures, such as an invalid table index or integer division by zero, also raise :class:`~didppy.ValidationError` with evaluation context.
Rust's installed panic hook may additionally write an evaluation diagnostic to stderr.
Non-finite objectives are rejected; arithmetic otherwise follows the existing DyPDL expression semantics.

Checking a declared cost
------------------------

Feasibility validation and cost comparison are separate operations.
:meth:`~didppy.Model.validate_solution` returns the computed objective without checking a declared cost.
To compare it with the cost of a solver result, use :meth:`~didppy.Model.validate_cost`:

.. code-block:: python

    solution = dp.CABS(model, quiet=True).search()
    cost = model.validate_solution(solution.transitions)
    if solution.cost is not None:
        model.validate_cost(cost, solution.cost)

Cost comparison returns :code:`None` when the costs match and raises :class:`~didppy.ValidationError` otherwise.
It only compares the supplied numbers; it does not replay transitions.

Cost comparison is exact by default.
Continuous costs can differ slightly due to floating-point evaluation order, so an application can choose a tolerance:

.. code-block:: python

    if solution.cost is not None:
        model.validate_cost(cost, solution.cost, rel_tol=1e-9, abs_tol=1e-9)

For continuous costs, the absolute difference must be at most
:code:`max(abs_tol, rel_tol * max(abs(declared), abs(computed)))`.
Both tolerances must be finite and non-negative.
Integer costs are always compared exactly.
Neither operation changes :code:`solution.cost`.

Loading a solution
-------------------

An existing model can load the solution format written by didp-yaml:

.. code-block:: python

    transitions, declared_cost = model.load_solution_from_file("solution.yaml")
    cost = model.validate_solution(transitions)
    if declared_cost is not None:
        model.validate_cost(cost, declared_cost)

For the knapsack model above, the file can contain:

.. code-block:: yaml

    cost: 60
    transitions:
      - name: ignore
      - name: pack
      - name: pack
      - name: ignore

:meth:`~didppy.Model.load_solution_from_str` accepts the same YAML as a string.
Both loading methods return a two-item tuple :code:`(transitions, declared_cost)`, not a :class:`~didppy.Solution`.
The first item is a list of :class:`~didppy.Transition` objects, ready to pass to :meth:`~didppy.Model.validate_solution`.
The second item is an :class:`int` for an integer-cost model, a :class:`float` for a continuous-cost model, or :code:`None` when no cost is declared.
:class:`~didppy.Solution` is reserved for solver results; loading does not create search statistics or solver status flags.

For parameterized YAML transitions, specify the original name and the complete parameter map, for example
:code:`{name: visit, parameters: {i: 3}}`.
For transitions created in Python, use the exact name passed to :class:`~didppy.Transition` and omit parameters or use :code:`parameters: {}`.
The name and parameter map must uniquely identify a transition in the supplied model.

The :code:`cost` field is optional; omitted or null costs load as :code:`None`.
The :code:`transitions` field is required, including for an empty sequence.
Loading resolves transition references but does not check feasibility or compute the objective.
It preserves the declared cost without comparing it to the objective. The return value always has two items, including when the transition list is empty or the declared cost is :code:`None`.
Invalid YAML or transition references raise :class:`ValueError`; file errors raise :class:`OSError` subclasses such as :class:`FileNotFoundError`.

Dumping a solution
------------------

:meth:`~didppy.Model.dump_solution_to_str` and :meth:`~didppy.Model.dump_solution_to_file` write the same solution format:

.. code-block:: python

    transitions = [ignore, pack, pack, ignore]
    text = model.dump_solution_to_str(transitions)
    restored, declared_cost = model.load_solution_from_str(text)
    assert declared_cost is None

    cost = model.validate_solution(restored)
    model.dump_solution_to_file(restored, "checked-solution.yaml", cost=cost)

Both methods accept a homogeneous sequence of :class:`~didppy.Transition` objects or :class:`~didppy.TransitionId` objects.
For a solver result, use :code:`model.dump_solution_to_str(solution.transitions, cost=solution.cost)`.
The optional :code:`cost` is preserved as supplied, not computed or compared; omitted or :code:`None` costs are written as YAML null and load as :code:`None`.
Integer-cost models require a 32-bit integer cost, while continuous-cost models accept finite numbers.

Dumping does not check feasibility or evaluate expressions. It can save incomplete or infeasible candidates for later validation.
Only transition names and parameter maps are serialized, not effects, preconditions, or cost expressions.
Transition-object membership is not checked; the references must uniquely identify forward or forward-forced transitions in the model used to reload them.
Changing an original transition object's expressions does not include those changes in the solution YAML.
IDs are resolved relative to the supplied model; backward and out-of-range IDs raise :class:`ValueError` with the sequence position.
Mixed sequences or incorrectly typed costs raise :class:`TypeError`; non-finite costs raise :class:`ValueError`.

The file method accepts :class:`str` and :class:`os.PathLike` paths and overwrites an existing file.
Serialization errors are reported before opening the file; write failures raise :class:`OSError` subclasses.
To export the model itself instead of a candidate solution, use :meth:`~didppy.Model.dump_to_str` or :meth:`~didppy.Model.dump_to_files`.
