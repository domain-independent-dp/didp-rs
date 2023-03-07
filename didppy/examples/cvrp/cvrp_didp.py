#!/usr/bin/env python3

import argparse
import os
import re

import didppy as dp

import read_tsplib


def solve(n, nodes, edges, capacity, demand, k, time_limit=None):
    model = dp.Model()
    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=[i for i in range(1, n)])
    location = model.add_element_var(object_type=customer, target=0)
    load = model.add_int_resource_var(target=0, less_is_better=True)
    vehicles = model.add_int_resource_var(target=1, less_is_better=True)
    demand = model.add_int_table([demand[i] for i in nodes])
    distance_matrix = [
        [edges[i, j] if (i, j) in edges else 0 for j in nodes] for i in nodes
    ]
    distance = model.add_int_table(distance_matrix)
    distance_via_depot = model.add_int_table(
        [
            [
                edges[i, nodes[0]] + edges[nodes[0], j]
                if (i, nodes[0]) in edges and (nodes[0], j) in edges
                else edges[i, j]
                if (i, j) in edges
                else 0
                for j in nodes
            ]
            for i in nodes
        ]
    )

    model.add_base_case([unvisited.is_empty(), location == 0])
    name_to_partial_tour = {}

    for i in range(1, n):
        name = "visit {}".format(i)
        name_to_partial_tour[name] = (nodes[i],)
        visit = dp.Transition(
            name=name,
            cost=dp.IntExpr.state_cost() + distance[location, i],
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (load, load + demand[i]),
            ],
            preconditions=[unvisited.contains(i), load + demand[i] <= capacity],
        )
        model.add_transition(visit)

    for i in range(1, n):
        name = "visit {} via depot".format(i)
        name_to_partial_tour[name] = (nodes[0], nodes[i])
        visit_via_depot = dp.Transition(
            name=name,
            cost=dp.IntExpr.state_cost() + distance_via_depot[location, i],
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (load, demand[i]),
                (vehicles, vehicles + 1),
            ],
            preconditions=[unvisited.contains(i), vehicles < k],
        )
        model.add_transition(visit_via_depot)

    name = "return"
    name_to_partial_tour[name] = (nodes[0],)
    return_to_depot = dp.Transition(
        name=name,
        cost=dp.IntExpr.state_cost() + distance[location, 0],
        effects=[(location, 0)],
        preconditions=[unvisited.is_empty(), location != 0],
    )
    model.add_transition(return_to_depot)

    model.add_state_constr((k - vehicles + 1) * capacity >= load + demand[unvisited])

    min_distance_to = model.add_int_table(
        [min(distance_matrix[i][j] for i in range(n) if i != j) for j in range(n)]
    )
    model.add_dual_bound(
        min_distance_to[unvisited] + (location != 0).if_then_else(min_distance_to[0], 0)
    )

    min_distance_from = model.add_int_table(
        [min(distance_matrix[i][j] for j in range(n) if i != j) for i in range(n)]
    )
    model.add_dual_bound(
        min_distance_from[unvisited]
        + (location != 0).if_then_else(min_distance_from[location], 0)
    )

    solver = dp.CABS(model, time_limit=time_limit, quiet=False)
    solution = solver.search()

    if solution.is_infeasible:
        print("Problem is infeasible")
    elif solution.is_optimal:
        print("Optimally solved")
    elif solution.time_out:
        print("Time out")

    tour = [1]

    for t in solution.transitions:
        print(t.name)
        tour += list(name_to_partial_tour[t.name])

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))

    return tour, solution.cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
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
        depot,
        _,
    ) = read_tsplib.read_cvrp(args.input)
    tour, cost = solve(n, nodes, edges, capacity, demand, k, time_limit=args.time_out)

    if cost is not None and read_tsplib.validate_cvrp(
        n, nodes, edges, capacity, demand, depot, tour, cost, k
    ):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
