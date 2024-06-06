import didppy as dp
import math
import pytest


def bpp_model(num_items, weights, capacity):
    model = dp.Model()

    item = model.add_object_type(num_items)
    unpacked = model.add_set_var(item, [i for i in range(num_items)])
    residual = model.add_int_resource_var(0, less_is_better=False)
    bin_number = model.add_element_resource_var(item, 0, less_is_better=True)

    weight_table = model.add_int_table(weights)
    lb2_weight1 = model.add_int_table(
        [1 if weights[i] > capacity / 2 else 0 for i in range(num_items)]
    )
    lb2_weight2 = model.add_float_table(
        [0.5 if weights[i] == capacity / 2 else 0 for i in range(num_items)]
    )
    lb3_weight = model.add_float_table(
        [
            (
                1.0
                if weights[i] > capacity * 2 / 3
                else (
                    2 / 3 // 0.001 / 1000
                    if weights[i] == capacity * 2 / 3
                    else (
                        0.5
                        if weights[i] > capacity / 3
                        else (
                            1 / 3 // 0.001 / 1000 if weights[i] == capacity / 3 else 0.0
                        )
                    )
                )
            )
            for i in range(num_items)
        ]
    )
    model.add_base_case([unpacked.is_empty()])

    name_to_item = {}

    for i in range(num_items):
        name = "pack {}".format(i)
        name_to_item[name] = i
        t = dp.Transition(
            name=name,
            cost=dp.IntExpr.state_cost(),
            effects=[
                (unpacked, unpacked.remove(i)),
                (residual, residual - weight_table[i]),
            ],
            preconditions=[
                unpacked.contains(i),
                weight_table[i] <= residual,
                bin_number <= i + 1,
            ],
        )
        model.add_transition(t)

        name = "open a new bin and pack {}".format(i)
        name_to_item[name] = i
        ft = dp.Transition(
            name=name,
            cost=dp.IntExpr.state_cost() + 1,
            preconditions=[
                bin_number <= i,
                unpacked.contains(i),
                weight_table[i] > residual,
            ]
            + [
                ~unpacked.contains(j) | (weight_table[j] > residual)
                for j in range(num_items)
                if i != j
            ],
            effects=[
                (unpacked, unpacked.remove(i)),
                (residual, capacity - weight_table[i]),
                (bin_number, bin_number + 1),
            ],
        )
        model.add_transition(ft, forced=True)

    model.add_dual_bound(math.ceil((weight_table[unpacked] - residual) / capacity))
    model.add_dual_bound(
        lb2_weight1[unpacked]
        + math.ceil(lb2_weight2[unpacked])
        - (residual >= capacity / 2).if_then_else(1, 0)
    )
    model.add_dual_bound(
        math.ceil(lb3_weight[unpacked]) - (residual >= capacity / 3).if_then_else(1, 0)
    )

    return model


def cvrp_model(num_locations, nodes, edges, capacity, demand, max_vehicles):
    model = dp.Model()
    customer = model.add_object_type(number=num_locations)
    unvisited = model.add_set_var(
        object_type=customer, target=[i for i in range(1, num_locations)]
    )
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
                (
                    edges[i, nodes[0]] + edges[nodes[0], j]
                    if (i, nodes[0]) in edges and (nodes[0], j) in edges
                    else edges[i, j] if (i, j) in edges else 0
                )
                for j in nodes
            ]
            for i in nodes
        ]
    )

    model.add_base_case([unvisited.is_empty(), location == 0])
    name_to_partial_tour = {}

    for i in range(1, num_locations):
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

    for i in range(1, num_locations):
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
            preconditions=[unvisited.contains(i), vehicles < max_vehicles],
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

    model.add_state_constr(
        (max_vehicles - vehicles + 1) * capacity >= load + demand[unvisited]
    )

    min_distance_to = model.add_int_table(
        [
            min(distance_matrix[i][j] for i in range(num_locations) if i != j)
            for j in range(num_locations)
        ]
    )
    model.add_dual_bound(
        min_distance_to[unvisited] + (location != 0).if_then_else(min_distance_to[0], 0)
    )

    min_distance_from = model.add_int_table(
        [
            min(distance_matrix[i][j] for j in range(num_locations) if i != j)
            for i in range(num_locations)
        ]
    )
    model.add_dual_bound(
        min_distance_from[unvisited]
        + (location != 0).if_then_else(min_distance_from[location], 0)
    )

    return model


def graph_clear_model(num_nodes, node_weights, edge_weights):
    model = dp.Model()
    node = model.add_object_type(num_nodes)
    clean = model.add_set_var(node, [])
    all_nodes = model.create_set_const(node, [i for i in range(num_nodes)])
    a = model.add_int_table(node_weights)
    b = model.add_int_table(
        [
            [
                (
                    edge_weights[i, j]
                    if (i, j) in edge_weights
                    else edge_weights[j, i] if (j, i) in edge_weights else 0
                )
                for j in range(num_nodes)
            ]
            for i in range(num_nodes)
        ]
    )

    model.add_base_case([all_nodes <= clean])

    name_to_node = {}

    for i in range(num_nodes):
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

    return model


def knapsack_model(num_items, weights, prices, capacity):
    model = dp.Model(maximize=True)
    items = model.add_object_type(num_items)
    remaining = model.add_int_resource_var(target=capacity, less_is_better=False)
    index = model.add_element_var(target=0, object_type=items)

    weight_table = model.add_int_table(weights)
    price_table = model.add_int_table(prices)

    pack = dp.Transition(
        name="pack",
        cost=dp.IntExpr.state_cost() + price_table[index],
        effects=[(index, index + 1), (remaining, remaining - weight_table[index])],
        preconditions=[index < num_items, remaining >= weight_table[index]],
    )
    model.add_transition(pack)

    ignore = dp.Transition(
        name="ignore",
        cost=dp.IntExpr.state_cost(),
        effects=[(index, index + 1)],
        preconditions=[index < num_items],
    )
    model.add_transition(ignore)

    model.add_base_case([index == num_items])

    return model


models = []
models.append(bpp_model(3, [1, 2, 3], 3))
models.append(
    cvrp_model(
        4,
        [0, 1, 2, 3],
        {
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 3,
            (1, 0): 1,
            (1, 2): 1,
            (1, 3): 2,
            (2, 0): 2,
            (2, 1): 1,
            (2, 3): 1,
            (3, 0): 3,
            (3, 1): 2,
            (3, 2): 1,
        },
        10,
        [4, 3, 3, 3],
        3,
    )
)
models.append(graph_clear_model(3, [1, 2, 3], {(0, 1): 1, (0, 2): 2, (1, 2): 1}))
models.append(knapsack_model(4, [4, 3, 3, 3], [4, 2, 5, 3], 10))


@pytest.mark.parametrize("model", models)
def test_load_and_dump(model):
    domain, problem = model.dump_to_str()
    reloaded_model = dp.Model.load_from_str(domain, problem)

    target_state = model.target_state
    reloaded_target_state = reloaded_model.target_state

    assert model.is_base(target_state) == reloaded_model.is_base(reloaded_target_state)
    assert model.eval_dual_bound(target_state) == reloaded_model.eval_dual_bound(
        reloaded_target_state
    )

    transitions = model.get_transitions()
    reloaded_transitions = reloaded_model.get_transitions()
    transitions.extend(model.get_transitions(forced=True))
    reloaded_transitions.extend(reloaded_model.get_transitions(forced=True))

    assert len(transitions) == len(reloaded_transitions)

    curr_state = None
    curr_reloaded_state = None
    next_state = target_state
    next_reloaded_state = reloaded_target_state
    while curr_state != next_state:
        curr_state = next_state
        curr_reloaded_state = next_reloaded_state

        for transition, reloaded_transition in zip(transitions, reloaded_transitions):
            assert transition.is_applicable(
                curr_state, model
            ) == reloaded_transition.is_applicable(curr_reloaded_state, reloaded_model)

            if transition.is_applicable(curr_state, model):
                assert transition.eval_cost(
                    0, curr_state, model
                ) == reloaded_transition.eval_cost(
                    0, curr_reloaded_state, reloaded_model
                )

                next_state = transition.apply(curr_state, model)
                next_reloaded_state = reloaded_transition.apply(
                    curr_reloaded_state, reloaded_model
                )

                assert model.is_base(next_state) == reloaded_model.is_base(
                    next_reloaded_state
                )
                assert model.eval_dual_bound(
                    next_state
                ) == reloaded_model.eval_dual_bound(next_reloaded_state)
