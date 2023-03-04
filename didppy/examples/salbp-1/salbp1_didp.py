#! /usr/bin/env python3

import argparse
import math

import didppy as dp

import read_salbp1


def solve(
    number_of_tasks,
    cycle_time,
    task_times,
    predecessors,
    time_limit=None,
):
    model = dp.Model()
    task = model.add_object_type(number=number_of_tasks)
    uncompleted = model.add_set_var(
        object_type=task, target=list(range(number_of_tasks))
    )
    idle_time = model.add_int_resource_var(target=0, less_is_better=False)
    task_time_table = model.add_int_table(
        [task_times[i + 1] for i in range(number_of_tasks)]
    )
    predecessors_table = model.add_set_table(
        [[j - 1 for j in predecessors[i + 1]] for i in range(number_of_tasks)],
        object_type=task,
    )
    lb2_weight1 = model.add_int_table(
        [1 if task_times[i + 1] > cycle_time / 2 else 0 for i in range(number_of_tasks)]
    )
    lb2_weight2 = model.add_float_table(
        [
            0.5 if task_times[i + 1] == cycle_time / 2 else 0
            for i in range(number_of_tasks)
        ]
    )
    lb3_weight = model.add_float_table(
        [
            1.0
            if task_times[i + 1] > cycle_time * 2 / 3
            else 2 / 3 // 0.001 / 1000
            if task_times[i + 1] == cycle_time * 2 / 3
            else 0.5
            if task_times[i + 1] > cycle_time / 3
            else 1 / 3 // 0.001 / 1000
            if task_times[i + 1] == cycle_time / 3
            else 0.0
            for i in range(number_of_tasks)
        ]
    )
    model.add_base_case([uncompleted.is_empty()])

    name_to_task = {}

    for i in range(number_of_tasks):
        name = "schedule {}".format(i)
        name_to_task[name] = i + 1
        t = dp.Transition(
            name=name,
            cost=dp.IntExpr.state_cost(),
            effects=[
                (uncompleted, uncompleted.remove(i)),
                (idle_time, idle_time - task_time_table[i]),
            ],
            preconditions=[
                uncompleted.contains(i),
                task_time_table[i] <= idle_time,
                (uncompleted & predecessors_table[i]).is_empty(),
            ],
        )
        model.add_transition(t)

    t = dp.Transition(
        name="open a new station",
        cost=dp.IntExpr.state_cost() + 1,
        effects=[(idle_time, cycle_time)],
        preconditions=[
            ~uncompleted.contains(i)
            | (task_time_table[i] > idle_time)
            | ((uncompleted & predecessors_table[i]).len() > 0)
            for i in range(number_of_tasks)
        ],
    )
    model.add_transition(t, forced=True)

    model.add_dual_bound(
        math.ceil((task_time_table[uncompleted] - idle_time) / cycle_time)
    )
    model.add_dual_bound(
        lb2_weight1[uncompleted]
        + math.ceil(lb2_weight2[uncompleted])
        - (idle_time >= cycle_time / 2).if_then_else(1, 0)
    )
    model.add_dual_bound(
        math.ceil(lb3_weight[uncompleted])
        - (idle_time >= cycle_time / 3).if_then_else(1, 0)
    )

    solver = dp.CABS(model, time_limit=time_limit, quiet=False)
    solution = solver.search()

    if solution.is_infeasible:
        print("Problem is infeasible")
    elif solution.is_optimal:
        print("Optimally solved")
    elif solution.time_out:
        print("Time out")

    assignment = []

    for t in solution.transitions:
        print(t.name)

        if t.name == "open a new station":
            assignment.append([])
        else:
            assignment[-1].append(name_to_task[t.name])

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))

    return assignment, solution.cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    number_of_tasks, cycle_time, task_times, predecessors, _ = read_salbp1.read(
        args.input
    )
    assignment, cost = solve(
        number_of_tasks,
        cycle_time,
        task_times,
        predecessors,
        time_limit=args.time_out,
    )
    if cost is not None and read_salbp1.validate(
        number_of_tasks, cycle_time, task_times, predecessors, assignment, cost
    ):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
