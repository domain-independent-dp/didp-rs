#!/usr/bin/env python3

import argparse

import didppy as dp

import read_mosp


def solve(item_to_patterns, pattern_to_items, time_limit=None):
    m = len(item_to_patterns)
    item_to_neighbors = read_mosp.compute_item_to_neighbors(
        item_to_patterns, pattern_to_items
    )

    model = dp.Model()
    item = model.add_object_type(m)
    remaining = model.add_set_var(item, [i for i in range(m)])
    opened = model.add_set_var(item, [])
    neighbors = model.add_set_table(item_to_neighbors, object_type=item)

    model.add_base_case([remaining.is_empty()])

    name_to_item = {}

    for i in range(m):
        name = "close {}".format(i)
        name_to_item[name] = i
        t = dp.Transition(
            name=name,
            cost=dp.max(
                dp.IntExpr.state_cost(),
                ((opened & remaining) | (neighbors[i] - opened)).len(),
            ),
            effects=[(remaining, remaining.remove(i)), (opened, opened | neighbors[i])],
            preconditions=[remaining.contains(i)],
        )
        model.add_transition(t)

    model.add_dual_bound(0)

    solver = dp.CABS(
        model, f_operator=dp.FOperator.Max, time_limit=time_limit, quiet=False
    )
    solution = solver.search()

    if solution.is_infeasible:
        print("Problem is infeasible")
    elif solution.is_optimal:
        print("Optimally solved")
    elif solution.time_out:
        print("Time out")

    pattern_sequence = []
    produced = set()

    for t in solution.transitions:
        print(t.name)
        item = name_to_item[t.name]

        for pattern in item_to_patterns[item]:
            if pattern not in produced:
                pattern_sequence.append(pattern)
                produced.add(pattern)

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))

    return pattern_sequence, solution.cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    item_to_patterns, pattern_to_items = read_mosp.read(args.input)
    sequence, cost = solve(item_to_patterns, pattern_to_items)

    if cost is not None and read_mosp.validate(
        item_to_patterns, pattern_to_items, sequence, cost
    ):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
