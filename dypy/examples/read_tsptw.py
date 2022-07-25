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
