#!/usr/bin/env python3

import argparse
import dypy as dp

import read_tsptw


def solve(n, nodes, edges, a, b):
    model = dp.Model()
    city = model.add_object_type(number=n)
    unvisited = model.add_set_var(object_type=city, target=[i for i in range(1, n)])
    location = model.add_element_var(object_type=city, target=0)
    time = model.add_int_resource_var(target=0, less_is_better=True)
    ready_time = model.add_int_table([a[i] for i in nodes])
    due_date = model.add_int_table([b[i] for i in nodes])
    distance = model.add_int_table(
        [[edges[i, j] if (i, j) in edges else 0 for i in nodes] for j in nodes]
    )

    for i in range(1, n):
        model.add_state_constr(
            ~(unvisited.contains(i)) | (time + distance[location, i] <= due_date[i])
        )

    model.add_base_case([location == 0, unvisited.is_empty()])

    state_cost = dp.IntExpr.state_cost()

    for i in range(1, n):
        visit = dp.Transition(
            "visit {}".format(i),
            cost=distance[location, i] + state_cost,
            preconditions=[unvisited.contains(i)],
            effects=[
                (unvisited, unvisited.remove(i)),
                (location, i),
                (time, dp.max(time + distance[location, i], ready_time[i])),
            ],
        )
        model.add_transition(visit)

    return_to_depot = dp.Transition(
        "return",
        cost=distance[location, 0] + state_cost,
        preconditions=[unvisited.is_empty(), location != 0],
        effects=[(location, 0), (time, time + distance[location, 0])],
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

    n, nodes, edges, a, b = read_tsptw.read(args.input)
    pddl_text = solve(n, nodes, edges, a, b)
