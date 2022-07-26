#!/usr/bin/env python3

import argparse

import dypy as dp

import read_graph_clear


def solve(n, node_weights, edge_weights, time_limit=None):
    model = dp.Model()
    node = model.add_object_type(n)
    clean = model.add_set_var(node, [])
    all_nodes = model.create_set_const(node, [i for i in range(n)])
    a = model.add_int_table(node_weights)
    b = model.add_int_table(
        [
            [
                edge_weights[i, j]
                if (i, j) in edge_weights
                else edge_weights[j, i]
                if (j, i) in edge_weights
                else 0
                for j in range(n)
            ]
            for i in range(n)
        ]
    )

    model.add_base_case([all_nodes <= clean])

    for i in range(n):
        t = dp.Transition(
            name="sweep {}".format(i),
            cost=dp.max(
                dp.IntExpr.state_cost(),
                a[i] + b[i, all_nodes] + b[clean, clean.complement().remove(i)],
            ),
            effects=[(clean, clean.add(i))],
            preconditions=[~clean.contains(i)],
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

    n, a, b = read_graph_clear.read(args.input)
    solve(n, a, b, time_limit=args.time_out)
