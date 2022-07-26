#!/usr/bin/env python3

import argparse

import dypy as dp

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

    for i in range(m):
        t = dp.Transition(
            name="close {}".format(i),
            cost=dp.max(
                dp.IntExpr.state_cost(),
                ((opened & remaining) | (neighbors[i] - opened)).len(),
            ),
            effects=[(remaining, remaining.remove(i)), (opened, opened | neighbors[i])],
            preconditions=[remaining.contains(i)],
        )
        model.add_transition(t)

    solver = dp.CAASDy(time_limit=time_limit)
    solution = solver.solve(model)

    for t in solution.transitions:
        print(t.name)

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    item_to_patterns, pattern_to_items = read_mosp.read(args.input)
    solve(item_to_patterns, pattern_to_items)
