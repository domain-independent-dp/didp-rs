def read(filename):
    with open(filename) as f:
        values = f.read().split()

    position = 0
    n = int(values[position])
    position += 1
    nodes = list(range(n))
    edges = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    edges[i, j] = int(values[position])
                except ValueError:
                    edges[i, j] = float(values[position])
            position += 1
    a = {}
    b = {}
    for i in range(n):
        try:
            a[i] = int(values[position])
        except ValueError:
            a[i] = float(values[position])
        position += 1
        try:
            b[i] = int(values[position])
        except ValueError:
            b[i] = float(values[position])
        position += 1

    return n, nodes, edges, a, b


def reduce_time_window(nodes, edges, a, b):
    reduced_a = dict(a)
    reduced_b = dict(b)
    change = True
    while change:
        change = False
        for i in nodes:
            a_i = min(
                reduced_b[i],
                min((reduced_a[j] + edges[j, i]) for j in nodes if (j, i) in edges),
                min((reduced_a[j] - edges[i, j]) for j in nodes if (i, j) in edges),
            )
            if a_i > reduced_a[i]:
                if not change:
                    change = True
                reduced_a[i] = a_i
            b_i = max(
                reduced_a[i],
                max((reduced_b[j] + edges[j, i]) for j in nodes if (j, i) in edges),
                max((reduced_b[j] - edges[i, j]) for j in nodes if (i, j) in edges),
            )
            if b_i < reduced_b[i]:
                if not change:
                    change = True
                reduced_b[i] = b_i

    return reduced_a, reduced_b


def reduce_edges(nodes, edges, a, b):
    print("edges: {}".format(len(edges)))
    forward_dependent = []
    for (i, j) in edges.keys():
        if i == 0 or j == 0 or a[i] <= b[j]:
            forward_dependent.append((i, j))

    direct_forward_dependent = {}
    for (i, j) in forward_dependent:
        if (
            i == 0
            or j == 0
            or all(b[i] > a[k] or b[k] > a[j] for k in nodes if k != 0)
            or (a[i] == a[j] and b[i] == b[j])
        ):
            direct_forward_dependent[i, j] = edges[i, j]

    print("reduced edges: {}".format(len(direct_forward_dependent)))
    return direct_forward_dependent


def validate(n, edges, a, b, solution, cost, tolerance=1e-4):
    previous = solution[0]
    if previous != 0:
        print(
            "The tour does not start from the depot {} but from {}".format(0, previous)
        )
        return False

    time = 0
    actual_cost = 0
    visited = set([0])

    for i in solution[1:-1]:
        if i < 0 or i > n - 1:
            print("Customer {} does not exist".format(i))
            return False
        if i in visited:
            print("Customer {} is already visited".format(i))
            return False
        visited.add(i)

        actual_cost += edges[previous, i]
        time = max(a[i], time + edges[previous, i])
        if time > b[i]:
            print("The time {} exceeds the deadline {} for {}".format(time, b[i], i))

        previous = i

    if solution[-1] != 0:
        print(
            "The tour does not return to the depot {}, but to {}".format(
                0, solution[-1]
            )
        )
        return False

    actual_cost += edges[previous, 0]

    if len(visited) != n:
        print(
            "The number of visited customers is {}, but should be {}".format(
                len(visited), n
            )
        )
        return False

    if abs(actual_cost - cost) > tolerance:
        print(
            "The cost of the solution {} mismatches the actual cost {}".format(
                cost, actual_cost
            )
        )
        return False

    return True
