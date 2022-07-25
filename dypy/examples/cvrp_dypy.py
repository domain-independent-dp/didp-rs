#!/usr/bin/env python3

import argparse
import os
import re

import dypy as dp

import read_tsplib


def solve(n, nodes, edges, capacity, demand, k):
    model = dp.Model()
    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=[i for i in range(1, n)])
    location = model.add_element_var(object_type=customer, target=0)
    load = model.add_int_resource_var(target=0, less_is_better=True)
    vehicles = model.add_int_resource_var(target=1, less_is_better=True)
    demand = model.add_int_table([demand[i] for i in nodes])
    distance = model.add_int_table(
        [[edges[i, j] if (i, j) in edges else 0 for j in nodes] for i in nodes]
    )

    model.add_base_case([unvisited.is_empty(), location == 0])

    for i in range(1, n):
        visit = dp.Transition(
            "visit {}".format(i),
            cost=dp.IntExpr.state_cost() + distance[location, i],
            preconditions=[unvisited.contains(i), load + demand[i] <= capacity],
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (load, load + demand[i]),
            ],
        )
        model.add_transition(visit)
        visit_via_depot = dp.Transition(
            "visit_via_depot {}".format(i),
            preconditions=[unvisited.contains(i), vehicles < k],
            cost=dp.IntExpr.state_cost() + distance[location, 0] + distance[0, i],
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (load, demand[i]),
            ],
        )
        model.add_transition(visit_via_depot)

    return_to_depot = dp.Transition(
        "return",
        cost=dp.IntExpr.state_cost() + distance[location, 0],
        preconditions=[unvisited.is_empty(), location != 0],
        effects=[(location, 0)],
    )
    model.add_transition(return_to_depot)

    solver = dp.CAASDy(time_limit=1800)
    solution = solver.solve(model)

    for t in solution.transitions:
        print(t.name)

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    name = os.path.basename(args.input)
    m = re.match(r".+k(\d+).+", name)
    k = int(m.group(1))

    (
        n,
        nodes,
        edges,
        capacity,
        demand,
        _,
        _,
    ) = read_tsplib.read_cvrp(args.input)
    problem = solve(n, nodes, edges, capacity, demand, k)
