#!/usr/bin/env python3

import argparse
import didppy as dp

import read_tsptw


def solve(n, nodes, edges, a, b, time_limit=None):
    model = dp.Model()
    customer = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=customer, target=[i for i in range(1, n)])
    location = model.add_element_var(object_type=customer, target=0)
    time = model.add_int_resource_var(target=0, less_is_better=True)
    ready_time = model.add_int_table([a[i] for i in nodes])
    due_date = model.add_int_table([b[i] for i in nodes])
    distance_matrix = [
        [edges[i, j] if (i, j) in edges else 0 for i in nodes] for j in nodes
    ]
    distance = model.add_int_table(distance_matrix)

    for i in range(1, n):
        model.add_state_constr(
            ~(unvisited.contains(i)) | (time + distance[location, i] <= due_date[i])
        )

    model.add_base_case([location == 0, unvisited.is_empty()])

    state_cost = dp.IntExpr.state_cost()
    name_to_customer = {}

    for i in range(1, n):
        name = "visit {}".format(i)
        name_to_customer[name] = i
        visit = dp.Transition(
            name=name,
            cost=distance[location, i] + state_cost,
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (time, dp.max(time + distance[location, i], ready_time[i])),
            ],
            preconditions=[unvisited.contains(i)],
        )
        model.add_transition(visit)

    name = "return"
    name_to_customer[name] = 0
    return_to_depot = dp.Transition(
        name=name,
        cost=distance[location, 0] + state_cost,
        effects=[(location, 0), (time, time + distance[location, 0])],
        preconditions=[unvisited.is_empty(), location != 0],
    )
    model.add_transition(return_to_depot)

    min_distance_to = model.add_int_table(
        [min(distance_matrix[i][j] for i in nodes if i != j) for j in nodes]
    )
    model.add_dual_bound(
        min_distance_to[unvisited] + (location != 0).if_then_else(min_distance_to[0], 0)
    )

    min_distance_from = model.add_int_table(
        [min(distance_matrix[i][j] for j in nodes if i != j) for i in nodes]
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

    tour = [0]

    for t in solution.transitions:
        print(t.name)
        tour.append(name_to_customer[t.name])

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))

    return tour, solution.cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    n, nodes, edges, a, b = read_tsptw.read(args.input)
    tour, cost = solve(n, nodes, edges, a, b, time_limit=args.time_out)

    if cost is not None and read_tsptw.validate(n, edges, a, b, tour, cost):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
