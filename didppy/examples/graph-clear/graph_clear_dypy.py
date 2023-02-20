#!/usr/bin/env python3

import argparse

import didppy as dp

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

    name_to_node = {}

    for i in range(n):
        name = "sweep {}".format(i)
        name_to_node[name] = i
        t = dp.Transition(
            name=name,
            cost=dp.max(
                dp.IntExpr.state_cost(),
                a[i] + b[i, all_nodes] + b[clean, clean.complement().remove(i)],
            ),
            effects=[(clean, clean.add(i))],
            preconditions=[~clean.contains(i)],
        )
        model.add_transition(t)

    model.add_dual_bound(0)

    solver = dp.CAASDy(model, time_limit=time_limit)
    solution = solver.search()

    if solution.is_infeasible:
        print("Problem is infeasible")
    elif solution.is_optimal:
        print("Optimally solved")
    elif solution.time_out:
        print("Time out")

    sequence = []

    for t in solution.transitions:
        print(t.name)
        sequence.append(name_to_node[t.name])

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))

    return sequence, solution.cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    n, a, b = read_graph_clear.read(args.input)
    sequence, cost = solve(n, a, b, time_limit=args.time_out)

    if cost is not None and read_graph_clear.validate(n, a, b, sequence, cost):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
